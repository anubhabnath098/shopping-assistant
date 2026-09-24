"""
Shared image-persistence helpers used by both the REST upload endpoint and
the WebSocket base64-frame endpoint, so the "save an uploaded image to disk"
logic lives in exactly one place.
"""
import os
import re
import base64
import uuid
from fastapi import UploadFile

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_DATA_URL_PATTERN = re.compile(r"^data:image/(?P<ext>\w+);base64,(?P<data>.+)$", re.DOTALL)


def validate_extension(extension: str) -> None:
    if extension.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type '{extension}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}")


async def save_uploaded_file(image: UploadFile, upload_dir: str) -> str:
    """Used by the REST /api/chat endpoint (multipart file upload)."""
    os.makedirs(upload_dir, exist_ok=True)
    extension = os.path.splitext(image.filename or "")[1].lower() or ".png"
    validate_extension(extension)
    file_name = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(upload_dir, file_name)
    contents = await image.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    return file_path


def save_base64_image(data_url: str, upload_dir: str) -> str:
    """
    Used by the WebSocket endpoint: decodes a base64 image (typically a
    'data:image/jpeg;base64,...' string from a browser canvas capture) and
    saves it to disk, returning the file path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    cleaned = data_url.strip()
    match = _DATA_URL_PATTERN.match(cleaned)

    if match:
        extension = match.group("ext").lower()
        raw_b64 = match.group("data")
    else:
        # No data-url prefix present; assume raw base64-encoded JPEG bytes.
        extension = "jpeg"
        raw_b64 = cleaned

    if extension == "jpg":
        extension = "jpeg"
    validate_extension(f".{extension}")

    file_name = f"{uuid.uuid4().hex}.{extension}"
    file_path = os.path.join(upload_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(raw_b64))
    return file_path