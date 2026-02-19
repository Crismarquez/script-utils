"""
Synthetic Dataset Generator.

Generates labeled datasets for document classification/segmentation evaluation
by combining individual PDFs (organized by file_type) into merged "loan packages"
with ground-truth segment labels.

Usage:
    # Generate exactly 50 merged documents
    python -m scripts.synthetic_dataset_generator \
      --source-dir data/source_docs \
      --output-dir data/synthetic_dataset_01 \
      --count 50 --seed 42

    # Generate until all source PDFs are used at least once
    python -m scripts.synthetic_dataset_generator \
      --source-dir data/source_docs \
      --output-dir data/synthetic_dataset_01 \
      --exhaust --seed 42

    # Use all PDFs with minimal reuse (balanced drain)
    python -m scripts.synthetic_dataset_generator \
      --source-dir data/source_docs \
      --output-dir data/synthetic_dataset_01 \
      --balanced --seed 42
"""

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse existing PDF utilities
from app.utils.pdf_utils import get_page_count, is_pdf, merge_pdfs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# file_type -> human-readable category mapping
# ---------------------------------------------------------------------------
FILE_TYPE_CATEGORY_MAP = {
    "1008": "1008 Fields",
    "LOAN": "1003 Lender Loan Information",
    "BRWR": "1003 Borrower",
    "ABOR": "1003 Additional Borrower",
    "CRPT": "Credit Report",
    "BAST": "Bank Statement",
    "BDIS": "Borrower Closing Disclosure",
    "DUFI": "DU Findings",
    "LPFB": "LPA Findings",
    "PSTB": "Pay Stub",
    "TAXR": "Tax Return",
    "TRAN": "Tax Transcript",
    "W2FM": "W-2 Form",
    "1004": "Uniform Residential Appraisal Report",
    "1073": "Condominium Unit Appraisal",
    "1025": "2-4 Unit Residential Income Property",
    "MAPR": "Manufactured Home Appraisal Report",
    "PROP": "Property Appraisal Report",
    "HYAP": "Hybrid Appraisal Report",
    "COMP": "1004D (Completion / Update)",
}


def get_category(file_type: str) -> str:
    """Resolve the category label for a file_type, falling back to the folder name."""
    if file_type in FILE_TYPE_CATEGORY_MAP:
        return FILE_TYPE_CATEGORY_MAP[file_type]
    logger.warning(
        "file_type '%s' not in FILE_TYPE_CATEGORY_MAP; using folder name as category",
        file_type,
    )
    return file_type


# ---------------------------------------------------------------------------
# Source directory scanning
# ---------------------------------------------------------------------------

def scan_source_dir(source_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan source directory and return {file_type: [pdf_paths]} for each subfolder.

    Only includes subfolders that contain at least one .pdf file.
    """
    file_type_map: Dict[str, List[Path]] = {}

    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        pdfs = sorted(child.glob("*.pdf"))
        if pdfs:
            file_type_map[child.name] = pdfs

    return file_type_map


# ---------------------------------------------------------------------------
# Cyclic pool – shuffled queue per file_type with automatic re-shuffle
# ---------------------------------------------------------------------------

class FileTypePool:
    """
    Manages a shuffled queue of PDFs for each file_type.

    When the queue for a type is exhausted it is re-shuffled (cyclic reuse).
    Tracks which PDFs have been consumed at least once per type so that
    ``--exhaust`` mode knows when full coverage is reached.
    """

    def __init__(self, file_type_map: Dict[str, List[Path]], rng: random.Random):
        self._rng = rng
        self._sources: Dict[str, List[Path]] = file_type_map
        self._queues: Dict[str, List[Path]] = {}
        self._used: Dict[str, set] = defaultdict(set)

        for ft, paths in file_type_map.items():
            q = list(paths)
            self._rng.shuffle(q)
            self._queues[ft] = q

    @property
    def file_types(self) -> List[str]:
        return list(self._sources.keys())

    def take(self, file_type: str) -> Path:
        """Return the next PDF path for *file_type*, re-shuffling if the queue is empty."""
        q = self._queues[file_type]
        if not q:
            q = list(self._sources[file_type])
            self._rng.shuffle(q)
            self._queues[file_type] = q
        path = q.pop()
        self._used[file_type].add(path)
        return path

    def all_used_at_least_once(self) -> bool:
        """True when every source PDF across all types has been consumed >= 1 time."""
        for ft, paths in self._sources.items():
            if len(self._used[ft]) < len(paths):
                return False
        return True

    def remaining_unused(self, file_type: str) -> int:
        """Number of source PDFs for *file_type* that haven't been used yet."""
        return len(self._sources[file_type]) - len(self._used[file_type])

    def remaining_unused_all(self) -> Dict[str, int]:
        """Return {file_type: remaining_unused_count} for every type."""
        return {ft: self.remaining_unused(ft) for ft in self._sources}

    def usage_counts(self) -> Dict[str, int]:
        """Return {file_type: number_of_times_a_pdf_was_taken}."""
        return {ft: len(used) for ft, used in self._used.items()}


# ---------------------------------------------------------------------------
# Round-robin type selector (guarantees coverage)
# ---------------------------------------------------------------------------

class TypeSelector:
    """
    Yields groups of file_types for each merged document, guaranteeing
    that every type appears at least once before any type repeats in the
    selection rotation.
    """

    def __init__(
        self,
        file_types: List[str],
        min_types: int,
        max_types: int,
        rng: random.Random,
    ):
        self._file_types = list(file_types)
        self._min = min_types
        self._max = max_types
        self._rng = rng
        self._deck: List[str] = []

    def _refill(self) -> None:
        deck = list(self._file_types)
        self._rng.shuffle(deck)
        self._deck = deck

    def next_group(self) -> List[str]:
        """Return a list of distinct file_types for the next merged document."""
        size = self._rng.randint(self._min, self._max)
        size = min(size, len(self._file_types))  # can't exceed available types

        group: List[str] = []
        while len(group) < size:
            if not self._deck:
                self._refill()
            candidate = self._deck.pop()
            if candidate not in group:
                group.append(candidate)

        # Shuffle the group so the order inside a document is random
        self._rng.shuffle(group)
        return group


# ---------------------------------------------------------------------------
# Balanced type selector (minimises reuse)
# ---------------------------------------------------------------------------

def plan_balanced_assignments(
    file_type_map: Dict[str, List[Path]],
    min_types: int,
    max_types: int,
    rng: random.Random,
) -> List[List[Tuple[str, Path]]]:
    """
    Pre-plan document assignments so each source PDF is used exactly once,
    producing the minimum number of merged documents with zero reuse.

    Algorithm:
        1. Compute ``num_docs = max(ceil(total_pdfs / max_types), max_type_count)``
           — the theoretical minimum number of documents.
        2. Create ``num_docs`` empty plans.
        3. Process file_types from largest to smallest PDF count.
        4. For each PDF, assign to the document with the fewest items that
           does not already contain that file_type and is not full.
        5. Merge sub-minimum documents (< min_types items) by redistributing
           their items into other documents that have room.
        6. Shuffle internal order of each document + overall document order.

    Returns:
        List of plans, each plan is a list of (file_type, Path) tuples.
    """
    total_pdfs = sum(len(paths) for paths in file_type_map.values())
    max_type_count = max(len(paths) for paths in file_type_map.values())
    num_docs = max(math.ceil(total_pdfs / max_types), max_type_count)

    # Initialize empty plans and type-tracking sets per document
    plans: List[List[Tuple[str, Path]]] = [[] for _ in range(num_docs)]
    types_in_doc: List[set] = [set() for _ in range(num_docs)]

    # Process types from largest to smallest count for best packing
    sorted_types = sorted(
        file_type_map.keys(), key=lambda ft: len(file_type_map[ft]), reverse=True
    )

    for ft in sorted_types:
        pdfs = list(file_type_map[ft])
        rng.shuffle(pdfs)
        for pdf_path in pdfs:
            # Find candidate documents: don't already have this type and not full
            best_idx = None
            best_size = float("inf")
            for i in range(num_docs):
                if ft in types_in_doc[i]:
                    continue
                if len(plans[i]) >= max_types:
                    continue
                if len(plans[i]) < best_size:
                    best_size = len(plans[i])
                    best_idx = i
            if best_idx is None:
                # All docs either have this type or are full — add a new doc
                best_idx = len(plans)
                plans.append([])
                types_in_doc.append(set())
                num_docs += 1
            plans[best_idx].append((ft, pdf_path))
            types_in_doc[best_idx].add(ft)

    # Merge sub-minimum documents by redistributing their items
    stable = False
    while not stable:
        stable = True
        for i in range(len(plans) - 1, -1, -1):
            if not plans[i] or len(plans[i]) >= min_types:
                continue
            # Try to redistribute items from this undersized doc
            items_to_move = list(plans[i])
            all_moved = True
            for ft, pdf_path in items_to_move:
                moved = False
                # Find another doc that can accept this item
                for j in range(len(plans)):
                    if j == i:
                        continue
                    if ft in types_in_doc[j]:
                        continue
                    if len(plans[j]) >= max_types:
                        continue
                    plans[j].append((ft, pdf_path))
                    types_in_doc[j].add(ft)
                    moved = True
                    break
                if not moved:
                    all_moved = False
            if all_moved:
                # Successfully redistributed all items — remove this doc
                plans[i] = []
                types_in_doc[i] = set()
                stable = False

    # Remove empty plans
    plans = [p for p in plans if p]

    # Shuffle internal order of each document and overall document order
    for p in plans:
        rng.shuffle(p)
    rng.shuffle(plans)

    return plans


# ---------------------------------------------------------------------------
# Single merged-document generation
# ---------------------------------------------------------------------------

def generate_merged_document(
    pool: FileTypePool,
    selected_types: List[str],
    doc_index: int,
    output_dir_name: str,
) -> Tuple[bytes, dict]:
    """
    Build one merged PDF and its label dict.

    Returns:
        (merged_pdf_bytes, label_dict)
    """
    doc_name = f"merged_{doc_index:04d}.pdf"

    pdf_parts: List[bytes] = []
    segment_infos: List[dict] = []

    for ft in selected_types:
        src_path = pool.take(ft)
        pdf_bytes = src_path.read_bytes()
        if not is_pdf(pdf_bytes):
            logger.warning("Skipping non-PDF file: %s", src_path)
            continue
        page_count = get_page_count(pdf_bytes)
        pdf_parts.append(pdf_bytes)
        segment_infos.append({
            "file_type": ft,
            "category": get_category(ft),
            "page_count": page_count,
            "source_file": f"{ft}/{src_path.name}",
        })

    merged_bytes = merge_pdfs(pdf_parts)

    # Build segments with cumulative page ranges
    segments = []
    current_page = 1
    for idx, info in enumerate(segment_infos, start=1):
        start = current_page
        end = current_page + info["page_count"] - 1
        segments.append({
            "segment_id": f"segment_{idx}",
            "category": info["category"],
            "file_type": info["file_type"],
            "start_page_number": start,
            "end_page_number": end,
            "source_file": info["source_file"],
        })
        current_page = end + 1

    total_pages = current_page - 1

    label = {
        "document_name": doc_name,
        "source_uri": f"{output_dir_name}/{doc_name}",
        "segments": segments,
        "total_pages": total_pages,
    }

    return merged_bytes, label


def generate_merged_document_from_plan(
    plan: List[Tuple[str, Path]],
    doc_index: int,
    output_dir_name: str,
) -> Tuple[bytes, dict]:
    """
    Build one merged PDF from pre-assigned (file_type, path) pairs.

    Same merge/label logic as ``generate_merged_document`` but uses
    pre-planned assignments instead of drawing from a FileTypePool.

    Returns:
        (merged_pdf_bytes, label_dict)
    """
    doc_name = f"merged_{doc_index:04d}.pdf"

    pdf_parts: List[bytes] = []
    segment_infos: List[dict] = []

    for ft, src_path in plan:
        pdf_bytes = src_path.read_bytes()
        if not is_pdf(pdf_bytes):
            logger.warning("Skipping non-PDF file: %s", src_path)
            continue
        page_count = get_page_count(pdf_bytes)
        pdf_parts.append(pdf_bytes)
        segment_infos.append({
            "file_type": ft,
            "category": get_category(ft),
            "page_count": page_count,
            "source_file": f"{ft}/{src_path.name}",
        })

    merged_bytes = merge_pdfs(pdf_parts)

    # Build segments with cumulative page ranges
    segments = []
    current_page = 1
    for idx, info in enumerate(segment_infos, start=1):
        start = current_page
        end = current_page + info["page_count"] - 1
        segments.append({
            "segment_id": f"segment_{idx}",
            "category": info["category"],
            "file_type": info["file_type"],
            "start_page_number": start,
            "end_page_number": end,
            "source_file": info["source_file"],
        })
        current_page = end + 1

    total_pages = current_page - 1

    label = {
        "document_name": doc_name,
        "source_uri": f"{output_dir_name}/{doc_name}",
        "segments": segments,
        "total_pages": total_pages,
    }

    return merged_bytes, label


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(args: argparse.Namespace) -> dict:
    """Execute the full generation pipeline; returns the manifest dict."""
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.is_dir():
        print(f"Error: source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Scan sources
    file_type_map = scan_source_dir(source_dir)
    available_types = list(file_type_map.keys())

    if len(available_types) < args.min_types:
        print(
            f"Error: found {len(available_types)} file_types but --min-types is "
            f"{args.min_types}. Need at least {args.min_types} types."
        )
        sys.exit(1)

    total_source_pdfs = sum(len(v) for v in file_type_map.values())
    print(f"Source scan: {len(available_types)} file_types, {total_source_pdfs} PDFs")
    for ft, paths in sorted(file_type_map.items()):
        print(f"  {ft}: {len(paths)} PDFs")

    # Seed
    rng = random.Random(args.seed)

    # Determine mode
    mode = "exhaust"
    target_count: Optional[int] = None
    if args.count is not None:
        mode = "count"
        target_count = args.count
    elif getattr(args, "balanced", False):
        mode = "balanced"

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_name = output_dir.name

    generated = 0
    total_pages = 0
    type_distribution: Dict[str, int] = defaultdict(int)

    print(f"\nGenerating merged documents ({mode} mode) ...")

    if mode == "balanced":
        plans = plan_balanced_assignments(
            file_type_map, args.min_types, args.max_types, rng
        )
        print(f"  Pre-planned {len(plans)} documents (zero PDF reuse)")

        for doc_index, plan in enumerate(plans, start=1):
            merged_bytes, label = generate_merged_document_from_plan(
                plan, doc_index, output_dir_name
            )

            # Write PDF
            pdf_path = output_dir / label["document_name"]
            pdf_path.write_bytes(merged_bytes)

            # Write label JSON
            json_name = label["document_name"].replace(".pdf", ".json")
            json_path = output_dir / json_name
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(label, f, indent=2, ensure_ascii=False)

            # Track stats
            generated += 1
            total_pages += label["total_pages"]
            for seg in label["segments"]:
                type_distribution[seg["file_type"]] += 1

            if generated % 10 == 0 or generated == 1:
                print(f"  [{generated}] {label['document_name']}  ({label['total_pages']} pages)")
    else:
        pool = FileTypePool(file_type_map, rng)
        selector = TypeSelector(available_types, args.min_types, args.max_types, rng)

        while True:
            # Stop conditions
            if mode == "count" and generated >= target_count:
                break
            if mode == "exhaust" and pool.all_used_at_least_once():
                break

            doc_index = generated + 1
            group = selector.next_group()

            merged_bytes, label = generate_merged_document(
                pool, group, doc_index, output_dir_name
            )

            # Write PDF
            pdf_path = output_dir / label["document_name"]
            pdf_path.write_bytes(merged_bytes)

            # Write label JSON
            json_name = label["document_name"].replace(".pdf", ".json")
            json_path = output_dir / json_name
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(label, f, indent=2, ensure_ascii=False)

            # Track stats
            generated += 1
            total_pages += label["total_pages"]
            for seg in label["segments"]:
                type_distribution[seg["file_type"]] += 1

            if generated % 10 == 0 or generated == 1:
                print(f"  [{generated}] {label['document_name']}  ({label['total_pages']} pages)")

    # Build manifest
    manifest = {
        "dataset_name": output_dir_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": generated,
        "total_pages": total_pages,
        "total_source_pdfs": total_source_pdfs,
        "file_types_used": sorted(type_distribution.keys()),
        "file_type_distribution": dict(sorted(type_distribution.items())),
        "config": {
            "min_types": args.min_types,
            "max_types": args.max_types,
            "seed": args.seed,
            "mode": mode,
            "count": target_count,
        },
    }
    if mode == "balanced":
        manifest["source_pdf_reuse"] = 0

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Dataset generated: {output_dir}")
    print(f"  Documents: {generated}")
    print(f"  Total pages: {total_pages}")
    print(f"  File types: {len(type_distribution)}")
    for ft, count in sorted(type_distribution.items()):
        print(f"    {ft}: {count} segments")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic merged PDF datasets with segment labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.synthetic_dataset_generator \\
    --source-dir data/source_docs --output-dir data/syn_01 --count 50 --seed 42

  python -m scripts.synthetic_dataset_generator \\
    --source-dir data/source_docs --output-dir data/syn_01 --exhaust

  python -m scripts.synthetic_dataset_generator \\
    --source-dir data/source_docs --output-dir data/syn_01 --balanced --seed 42
        """,
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Root directory with one subfolder per file_type containing PDFs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged PDFs, labels, and manifest will be written",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--count",
        type=int,
        default=None,
        help="Generate exactly N merged documents",
    )
    mode_group.add_argument(
        "--exhaust",
        action="store_true",
        default=False,
        help="Generate until every source PDF has been used at least once (default)",
    )
    mode_group.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help=(
            "Pre-plan document assignments so each source PDF is used exactly once, "
            "producing the minimum number of merged documents. No PDF reuse."
        ),
    )

    parser.add_argument(
        "--min-types",
        type=int,
        default=3,
        help="Minimum number of distinct file_types per merged document (default: 3)",
    )
    parser.add_argument(
        "--max-types",
        type=int,
        default=5,
        help="Maximum number of distinct file_types per merged document (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Default to exhaust when no mode is specified
    if args.count is None and not args.exhaust and not args.balanced:
        args.exhaust = True

    # Validation
    if args.min_types < 1:
        parser.error("--min-types must be >= 1")
    if args.max_types < args.min_types:
        parser.error("--max-types must be >= --min-types")
    if args.count is not None and args.count < 1:
        parser.error("--count must be >= 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_generation(args)


if __name__ == "__main__":
    main()
