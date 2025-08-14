"""Client script to request ball spin calculation from the server via HTTPS."""

import base64
import json
import requests
from pathlib import Path
from typing import Any

API_URL = "https://api.ddxcr.com/backspin"  # Example HTTPS endpoint


def _encode_image(path: str | Path) -> str:
    """Read an image file and return a base64-encoded string."""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def request_backspin(image1_path: str, image2_path: str) -> Any:
    """Send two ball images to the server and return the spin calculation."""
    payload = {
        "image1": _encode_image(image1_path),
        "image2": _encode_image(image2_path),
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Example usage for requesting a spin calculation."""
    image1_path = "ball_frame1.jpg"
    image2_path = "ball_frame2.jpg"
    result = request_backspin(image1_path, image2_path)
    print("Spin result:", result)


if __name__ == "__main__":
    main()
