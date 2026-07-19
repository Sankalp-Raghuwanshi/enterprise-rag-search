"""
extract.py
----------
Extracts raw text from uploaded documents, keeping track of page/section
metadata so we can cite sources later.

Phase 1 supports: PDF, DOCX
(PPTX / CSV / Markdown are added in Phase 6 - see README)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document


@dataclass
class ExtractedPage:
    """One 'page' or 'section' of extracted text, with metadata for citations."""
    text: str
    source_file: str
    page_number: int  # 1-indexed. For docx, this is a synthetic paragraph-block number.


def extract_pdf(file_path: str, display_name: str = None) -> List[ExtractedPage]:
    """Extract text from a PDF, one ExtractedPage per PDF page."""
    reader = PdfReader(file_path)
    filename = display_name or Path(file_path).name
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:  # skip blank pages
            pages.append(ExtractedPage(text=text, source_file=filename, page_number=i + 1))

    return pages


def extract_docx(file_path: str, paragraphs_per_block: int = 10, display_name: str = None) -> List[ExtractedPage]:
    """
    Extract text from a Word doc. DOCX has no native 'pages', so we group
    paragraphs into synthetic blocks (default: 10 paragraphs per block)
    and treat each block like a 'page' for citation purposes.
    """
    doc = Document(file_path)
    filename = display_name or Path(file_path).name
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    pages = []
    for i in range(0, len(paragraphs), paragraphs_per_block):
        block = paragraphs[i:i + paragraphs_per_block]
        text = "\n".join(block)
        block_number = (i // paragraphs_per_block) + 1
        pages.append(ExtractedPage(text=text, source_file=filename, page_number=block_number))

    return pages


def extract_document(file_path: str, display_name: str = None) -> List[ExtractedPage]:
    """
    Dispatch to the right extractor based on file extension.

    display_name: the filename to use for citations, if it differs from
    file_path (e.g. Streamlit saves uploads to a random temp path - we
    still want citations to show the user's original filename).
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_path, display_name=display_name)
    elif suffix == ".docx":
        return extract_docx(file_path, display_name=display_name)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Phase 1 supports .pdf and .docx only."
        )


if __name__ == "__main__":
    # Quick manual test: `python src/extract.py path/to/file.pdf`
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract.py <file_path>")
        sys.exit(1)

    result = extract_document(sys.argv[1])
    print(f"Extracted {len(result)} pages/blocks from {sys.argv[1]}")
    for p in result[:2]:
        print(f"\n--- Page {p.page_number} ---")
        print(p.text[:300])
