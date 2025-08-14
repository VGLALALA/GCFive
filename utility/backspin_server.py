"""Flask API server for ball spin calculation using image pairs."""

import base64
import json
from typing import Any

import cv2
import numpy as np
from flask import Flask, request, jsonify

from spin.GetBallRotation import get_fine_ball_rotation

app = Flask(__name__)


def _decode_image(b64: str) -> np.ndarray:
    """Decode a base64-encoded image string into a NumPy array."""
    img_bytes = base64.b64decode(b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@app.route("/backspin", methods=["POST"])
def backspin_endpoint():
    """API endpoint to receive image data, compute spin, and return result."""
    payload: Any = request.get_json(force=True)
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

    return jsonify(result)


def main() -> None:
    """Run the backspin calculation API server."""
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
