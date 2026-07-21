"""
image_extractor.py
-------------------
Extracts embedded images from PDF pages using PyMuPDF (fitz) - a more
reliable choice than pypdf for this specifically, since pypdf's image
extraction is inconsistent across different PDF encoders.

Why this is a separate module from extract.py: text extraction and
image extraction are genuinely different operations (different library,
different output type), and keeping them separate means either can be
disabled/swapped independently - e.g. skipping image extraction entirely
for a text-only PDF, without touching the text pipeline at all.

Note: this only pulls out images that are embedded as raster images
(photos, screenshots, exported charts saved as PNG/JPEG inside the PDF).
It does NOT capture vector graphics drawn directly in the PDF (e.g. a
chart built from PDF drawing commands, not an embedded image) - that
would require rasterizing the whole page instead, a reasonable future
extension if needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class ExtractedImage:
    image_bytes: bytes
    image_ext: str          # e.g. "png", "jpeg" - needed for the vision API's data URL
    source_file: str
    page_number: int        # 1-indexed, matches extract.py's page numbering
    image_index: int        # position of this image within its page (0, 1, 2...)


def extract_images(file_path: str, display_name: str = None) -> List[ExtractedImage]:
    """
    Pull every embedded image out of a PDF, with page-level metadata so
    each image can be cited back to its source page, same as text chunks.
    """
    filename = display_name or Path(file_path).name
    doc = fitz.open(file_path)

    images = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            images.append(ExtractedImage(
                image_bytes=base_image["image"],
                image_ext=base_image["ext"],
                source_file=filename,
                page_number=page_index + 1,
                image_index=img_idx,
            ))

    doc.close()
    return images


if __name__ == "__main__":
    # Quick manual test: `python image_extractor.py path/to/file.pdf`
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_extractor.py <file_path>")
        sys.exit(1)

    result = extract_images(sys.argv[1])
    print(f"Found {len(result)} embedded images in {sys.argv[1]}")
    for img in result:
        print(f"  - page {img.page_number}, image {img.image_index}, "
              f"format {img.image_ext}, {len(img.image_bytes)} bytes")
