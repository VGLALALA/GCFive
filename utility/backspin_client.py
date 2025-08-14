"""Client script to request ball spin calculation from the server."""

import base64
import json
import socket
from pathlib import Path
from typing import Any

HOST = "api.ddxcr.com"  # Server hostname or IP address
PORT = 65432  # Server port to connect to


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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(payload).encode("utf-8"))
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk

    return json.loads(data.decode("utf-8"))


def main() -> None:
    """Example usage for requesting a spin calculation."""
    image1_path = "ball_frame1.jpg"
    image2_path = "ball_frame2.jpg"
    result = request_backspin(image1_path, image2_path)
    print("Spin result:", result)


if __name__ == "__main__":
    main()
