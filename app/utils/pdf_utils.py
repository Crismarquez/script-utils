"""
PDF Utilities.

Provides functions for PDF manipulation, specifically page slicing
to support isolated processing of document segments.

This module enables the V2 execution mode for composite classifiers
and extractors, where each child analyzer only processes the pages
that belong to its parent segment, preventing cross-contamination.
"""

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PageRange:
    """
    Represents a range of pages within a document.
    
    Attributes:
        start: Starting page number (1-based, inclusive)
        end: Ending page number (1-based, inclusive)
    """
    start: int
    end: int
    
    def __post_init__(self):
        if self.start < 1:
            raise ValueError(f"start must be >= 1, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
    
    @property
    def page_count(self) -> int:
        """Number of pages in this range."""
        return self.end - self.start + 1
    
    def to_zero_based(self) -> Tuple[int, int]:
        """Convert to 0-based indices for pypdf."""
        return (self.start - 1, self.end - 1)


def is_pdf(data: bytes) -> bool:
    """
    Check if the given bytes represent a PDF document.
    
    Args:
        data: Document bytes to check
        
    Returns:
        True if the data starts with PDF magic bytes
    """
    return data[:4] == b'%PDF'


def get_page_count(pdf_bytes: bytes) -> int:
    """
    Get the total number of pages in a PDF document.
    
    Args:
        pdf_bytes: Raw PDF bytes
        
    Returns:
        Number of pages in the document
        
    Raises:
        ValueError: If the input is not a valid PDF
    """
    if not is_pdf(pdf_bytes):
        raise ValueError("Input is not a valid PDF document")
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        raise ValueError(f"Failed to read PDF: {e}") from e


def slice_pdf_pages(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
) -> bytes:
    """
    Extract a range of pages from a PDF document.
    
    Creates a new PDF containing only the specified pages.
    Page numbers are 1-based and inclusive.
    
    Args:
        pdf_bytes: Raw bytes of the source PDF
        start_page: First page to include (1-based)
        end_page: Last page to include (1-based)
        
    Returns:
        Bytes of the new PDF containing only the requested pages
        
    Raises:
        ValueError: If input is not a valid PDF or page range is invalid
        
    Example:
        >>> # Extract pages 3-7 from a PDF
        >>> sliced = slice_pdf_pages(original_pdf, 3, 7)
    """
    if not is_pdf(pdf_bytes):
        raise ValueError("Input is not a valid PDF document")
    
    page_range = PageRange(start=start_page, end=end_page)
    
    try:
        from pypdf import PdfReader, PdfWriter
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        
        # Validate range against actual document
        if page_range.start > total_pages:
            raise ValueError(
                f"start_page ({page_range.start}) exceeds document pages ({total_pages})"
            )
        
        # Clamp end_page to document length
        actual_end = min(page_range.end, total_pages)
        if actual_end != page_range.end:
            logger.warning(
                f"end_page ({page_range.end}) exceeds document pages ({total_pages}), "
                f"clamping to {actual_end}"
            )
        
        # Create new PDF with selected pages
        writer = PdfWriter()
        start_idx, _ = page_range.to_zero_based()
        end_idx = actual_end - 1  # Use clamped end
        
        for page_idx in range(start_idx, end_idx + 1):
            writer.add_page(reader.pages[page_idx])
        
        # Write to bytes
        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        
        logger.debug(
            f"Sliced PDF: pages {start_page}-{actual_end} "
            f"({end_idx - start_idx + 1} pages) from {total_pages} total"
        )
        
        return output.read()
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to slice PDF: {e}")
        raise ValueError(f"Failed to slice PDF: {e}") from e


def slice_pdf_for_segment(
    pdf_bytes: bytes,
    segment_start: int,
    segment_end: int,
    total_pages: Optional[int] = None,
) -> Tuple[bytes, int]:
    """
    Extract pages for a document segment with offset information.
    
    This is the primary function for V2 execution mode. It creates a
    sliced PDF for a segment and returns the page offset needed to
    translate results back to original document coordinates.
    
    Args:
        pdf_bytes: Raw bytes of the source PDF
        segment_start: Segment's start page (1-based)
        segment_end: Segment's end page (1-based)
        total_pages: Optional total page count (computed if not provided)
        
    Returns:
        Tuple of (sliced_pdf_bytes, page_offset)
        - sliced_pdf_bytes: New PDF with only segment pages
        - page_offset: Number to add to results to get original page numbers
                       (equals segment_start - 1)
        
    Example:
        >>> # Segment covers pages 5-10 of original document
        >>> sliced, offset = slice_pdf_for_segment(pdf, 5, 10)
        >>> # offset = 4
        >>> # If child reports page 1, original page = 1 + 4 = 5
    """
    sliced_bytes = slice_pdf_pages(pdf_bytes, segment_start, segment_end)
    page_offset = segment_start - 1
    
    return sliced_bytes, page_offset


def adjust_page_numbers(
    page_numbers: List[int],
    offset: int,
) -> List[int]:
    """
    Adjust page numbers by adding an offset.
    
    Used to translate page numbers from a sliced PDF back to
    coordinates in the original document.
    
    Args:
        page_numbers: List of page numbers (1-based)
        offset: Offset to add (typically segment_start - 1)
        
    Returns:
        List of adjusted page numbers
        
    Example:
        >>> adjust_page_numbers([1, 2, 3], offset=4)
        [5, 6, 7]
    """
    return [p + offset for p in page_numbers]


def merge_pdfs(pdf_list: List[bytes]) -> bytes:
    """
    Merge multiple PDF documents into a single PDF.
    
    Useful for recombining processed segments if needed.
    
    Args:
        pdf_list: List of PDF bytes to merge in order
        
    Returns:
        Bytes of the merged PDF
        
    Raises:
        ValueError: If any input is not a valid PDF
    """
    if not pdf_list:
        raise ValueError("Cannot merge empty list of PDFs")
    
    try:
        from pypdf import PdfWriter
        
        writer = PdfWriter()
        
        for idx, pdf_bytes in enumerate(pdf_list):
            if not is_pdf(pdf_bytes):
                raise ValueError(f"Input at index {idx} is not a valid PDF")
            
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                writer.add_page(page)
        
        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        
        return output.read()
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to merge PDFs: {e}")
        raise ValueError(f"Failed to merge PDFs: {e}") from e


def get_document_info(pdf_bytes: bytes) -> dict:
    """
    Get metadata and basic info about a PDF document.
    
    Args:
        pdf_bytes: Raw PDF bytes
        
    Returns:
        Dictionary with document information:
        - page_count: Number of pages
        - metadata: PDF metadata dictionary
        - is_encrypted: Whether the PDF is encrypted
    """
    if not is_pdf(pdf_bytes):
        raise ValueError("Input is not a valid PDF document")
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        
        return {
            "page_count": len(reader.pages),
            "metadata": dict(reader.metadata) if reader.metadata else {},
            "is_encrypted": reader.is_encrypted,
        }
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        raise ValueError(f"Failed to get PDF info: {e}") from e
