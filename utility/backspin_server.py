"""Simple socket server for ball spin calculation using image pairs."""

import base64
import json
import socket
from typing import Any

import cv2
import numpy as np

from spin.GetBallRotation import get_fine_ball_rotation

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


def _decode_image(b64: str) -> np.ndarray:
    """Decode a base64-encoded image string into a NumPy array."""
    img_bytes = base64.b64decode(b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def handle_client(conn: socket.socket) -> None:
    """Receive image data from *conn*, compute spin and return result.

    Parameters
    ----------
    conn:
        Established socket connection to the client.
    """
    with conn:
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        if not data:
            return

        payload: Any = json.loads(data.decode("utf-8"))
        image1_b64: str = payload["image1"]
        image2_b64: str = payload["image2"]

        ball_image1 = _decode_image(image1_b64)
        ball_image2 = _decode_image(image2_b64)

        spin_x, spin_y, spin_z = get_fine_ball_rotation(ball_image1, ball_image2)

        result = {
            "side_spin_deg": spin_x,
            "back_spin_deg": spin_y,
            "axial_spin_deg": spin_z,
        }

        conn.sendall(json.dumps(result).encode("utf-8"))


def main() -> None:
    """Run the backspin calculation server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Backspin server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            handle_client(conn)


if __name__ == "__main__":
    main()
