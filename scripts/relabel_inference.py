"""
Inference Relabeler.

Replaces predicted segments in a platform-exported inference result with
ground-truth labels from the synthetic dataset, producing a corrected file
ready for re-upload.

The platform often over-segments documents (many "Other/NONE" single-page
segments).  This script matches each document by name against its synthetic
label and swaps in the correct segments while preserving the batch envelope.

Usage:
    python -m scripts.relabel_inference \
      --inferred data/inferred-result.json \
      --labels-dir data/synthetic_dataset_01 \
      --output data/relabeled-result.json

    # Dry-run: only print the diff without writing
    python -m scripts.relabel_inference \
      --inferred data/inferred-result.json \
      --labels-dir data/synthetic_dataset_01 \
      --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(labels_dir: Path) -> Dict[str, dict]:
    """
    Load all merged_*.json label files from *labels_dir*.

    Returns:
        {document_name: label_dict}  e.g. {"merged_0001.pdf": {...}}
    """
    labels: Dict[str, dict] = {}
    for label_path in sorted(labels_dir.glob("merged_*.json")):
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        doc_name = label.get("document_name")
        if doc_name:
            labels[doc_name] = label
    return labels


# ---------------------------------------------------------------------------
# Segment conversion: ground-truth format -> inferred format
# ---------------------------------------------------------------------------

def convert_segments(
    gt_segments: List[dict],
    source_classifier_id: Optional[str] = None,
) -> List[dict]:
    """
    Convert ground-truth segments into the platform's inferred segment format.

    Ground-truth segment keys:
        segment_id, category, file_type, start_page_number,
        end_page_number, source_file

    Inferred segment keys:
        segment_id, category, file_type, start_page_number,
        end_page_number, confidence, sub_segments, depth,
        source_classifier_id
    """
    converted = []
    for idx, gt in enumerate(gt_segments, start=1):
        converted.append({
            "segment_id": f"segment{idx}",
            "category": gt["category"],
            "file_type": gt["file_type"],
            "start_page_number": gt["start_page_number"],
            "end_page_number": gt["end_page_number"],
            "confidence": None,
            "sub_segments": None,
            "depth": 0,
            "source_classifier_id": source_classifier_id,
        })
    return converted


# ---------------------------------------------------------------------------
# Relabeling
# ---------------------------------------------------------------------------

def relabel_document(doc: dict, gt_label: dict) -> dict:
    """
    Replace segments in an inferred document with ground-truth segments.

    Preserves all envelope fields (id, source_uri, total_pages, …) and
    carries over ``source_classifier_id`` from the first original segment.
    """
    # Carry over classifier id from original prediction
    source_classifier_id = None
    if doc.get("segments"):
        source_classifier_id = doc["segments"][0].get("source_classifier_id")

    new_segments = convert_segments(gt_label["segments"], source_classifier_id)

    relabeled = dict(doc)
    relabeled["segments"] = new_segments
    relabeled["segmentCount"] = len(new_segments)
    return relabeled


def relabel_batch(
    inferred: dict,
    labels: Dict[str, dict],
) -> tuple:
    """
    Relabel all documents in an inferred batch.

    Returns:
        (relabeled_batch, stats) where stats is a summary dict.
    """
    result = dict(inferred)
    documents = list(inferred.get("documents", []))

    matched = 0
    skipped = 0
    skipped_names: List[str] = []
    details: List[dict] = []

    new_documents = []
    for doc in documents:
        doc_name = doc.get("document_name", "")
        gt_label = labels.get(doc_name)

        if gt_label is None:
            logger.warning("No label found for '%s' — keeping original segments", doc_name)
            new_documents.append(doc)
            skipped += 1
            skipped_names.append(doc_name)
            continue

        old_count = len(doc.get("segments", []))
        relabeled = relabel_document(doc, gt_label)
        new_count = len(relabeled["segments"])
        new_documents.append(relabeled)
        matched += 1

        details.append({
            "document_name": doc_name,
            "segments_before": old_count,
            "segments_after": new_count,
        })

    result["documents"] = new_documents

    stats = {
        "total_documents": len(documents),
        "matched": matched,
        "skipped": skipped,
        "skipped_names": skipped_names,
        "details": details,
    }
    return result, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(stats: dict) -> None:
    """Print a human-readable correction report."""
    total = stats["total_documents"]
    matched = stats["matched"]
    skipped = stats["skipped"]

    print(f"\n{'=' * 60}")
    print(f"Relabeling report")
    print(f"{'=' * 60}")
    print(f"  Documents in batch : {total}")
    print(f"  Matched & relabeled: {matched}")
    print(f"  Skipped (no label) : {skipped}")

    if stats["skipped_names"]:
        print(f"\n  Skipped documents:")
        for name in stats["skipped_names"]:
            print(f"    - {name}")

    if stats["details"]:
        print(f"\n  Corrections:")
        for d in stats["details"]:
            before = d["segments_before"]
            after = d["segments_after"]
            delta = before - after
            arrow = f"{before} -> {after}"
            if delta > 0:
                arrow += f"  (consolidated {delta} segments)"
            elif delta < 0:
                arrow += f"  (expanded by {abs(delta)} segments)"
            print(f"    {d['document_name']:30s}  segments: {arrow}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replace inferred segments with ground-truth labels from synthetic dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.relabel_inference \\
    --inferred data/inferred-result.json \\
    --labels-dir data/synthetic_dataset_01 \\
    --output data/relabeled-result.json

  python -m scripts.relabel_inference \\
    --inferred data/inferred-result.json \\
    --labels-dir data/synthetic_dataset_01 \\
    --dry-run
        """,
    )
    parser.add_argument(
        "--inferred",
        required=True,
        help="Path to the platform-exported inference result JSON",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing merged_*.json ground-truth labels",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the relabeled JSON (default: <inferred>_relabeled.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report without writing the output file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    inferred_path = Path(args.inferred)
    labels_dir = Path(args.labels_dir)

    if not inferred_path.is_file():
        print(f"Error: inferred file '{inferred_path}' does not exist.")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"Error: labels directory '{labels_dir}' does not exist.")
        sys.exit(1)

    # Load inputs
    with open(inferred_path, "r", encoding="utf-8") as f:
        inferred = json.load(f)

    labels = load_labels(labels_dir)
    print(f"Loaded {len(labels)} ground-truth labels from {labels_dir}")

    # Relabel
    relabeled, stats = relabel_batch(inferred, labels)
    print_report(stats)

    if args.dry_run:
        print("\n(dry-run — no file written)")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = inferred_path.with_name(
            inferred_path.stem + "_relabeled" + inferred_path.suffix
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relabeled, f, indent=2, ensure_ascii=False)

    print(f"\nRelabeled result written to: {output_path}")


if __name__ == "__main__":
    main()
