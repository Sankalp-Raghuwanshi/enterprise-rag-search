"""
vision.py
---------
Describes an image using a vision-capable LLM, so the description text
can be embedded and searched exactly like any other chunk - the vector
store and BM25 index don't need to know an "image" is involved at all,
they just see more text with source_file/page_number metadata.

IMPORTANT: verify VISION_MODEL against console.groq.com/docs/models
before running this - vision-capable model availability and naming on
Groq changes over time, and the placeholder below may not be current.
Groq's chat completions endpoint accepts image inputs in the same
OpenAI-compatible format used elsewhere in this project (see llm.py).
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "qwen/qwen3.6-27b"  # confirmed vision-capable per console.groq.com/docs/vision as of July 2026 - re-verify if this breaks

DESCRIBE_IMAGE_PROMPT = (
    "Describe this image in detail, as if writing a caption for someone who "
    "cannot see it. If it contains a chart, diagram, table, or text, transcribe "
    "or summarize the key information precisely. Be factual and specific - this "
    "description will be used to help someone search for this image later."
)


def describe_image(image_bytes: bytes, image_ext: str = "png", model: str = VISION_MODEL) -> str:
    """
    Send an image to a vision-capable LLM and return a text description.
    The description is what actually gets embedded/indexed - the image
    itself is never stored in the vector store or BM25 index, only its
    textual description is.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{image_ext};base64,{base64_image}"

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DESCRIBE_IMAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # Quick manual test: `python vision.py path/to/image.png`
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vision.py <image_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    ext = sys.argv[1].split(".")[-1]
    description = describe_image(image_bytes, image_ext=ext)
    print(description)
