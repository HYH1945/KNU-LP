"""Shared pipeline types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadedImage:
    """Input: bytes data, str content_type. Output: immutable UploadedImage. Purpose: carry uploaded image bytes and original MIME type across pipeline boundaries."""

    data: bytes
    content_type: str
