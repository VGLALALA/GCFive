from __future__ import annotations

import io
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from .range import RangeSim

app = FastAPI(title="Golf Shot Visualizer")

_last_image: Optional[bytes] = None
_last_data: Optional[Dict[str, float]] = None


def _simulate_shot(ball_data: Dict[str, float]):
    sim = RangeSim()
    sim.ball.hit_from_data(ball_data)
    sim.track_points = True
    sim.trail.add_point(sim.ball.position.copy())

    for _ in range(4000):
        sim.step(1 / 240.0)
        if np.linalg.norm(sim.ball.velocity) < 0.1 and sim.ball.position[1] <= 0:
            break

    positions = np.array(sim.ball.total_position_list)
    carry_yards = None
    if sim.ball.position_list:
        p = sim.ball.position_list[0]
        carry_yards = float((p[[0, 2]] ** 2).sum() ** 0.5 * 1.09361)
    total_yards = sim.distance_yards
    return positions, carry_yards, total_yards


def _plot_trajectory(positions: np.ndarray) -> bytes:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0] * 1.09361, positions[:, 2] * 1.09361, positions[:, 1] * 3.28084)
    ax.set_xlabel("Forward (yd)")
    ax.set_ylabel("Lateral (yd)")
    ax.set_zlabel("Height (ft)")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.post("/shot")
async def shot(shot: Dict):
    global _last_image, _last_data
    ball_data = shot.get("BallData") or shot
    positions, carry, total = _simulate_shot(ball_data)
    _last_image = _plot_trajectory(positions)
    _last_data = {
        "speed": ball_data.get("Speed", 0.0),
        "launch_angle": ball_data.get("VLA", 0.0),
        "backspin": ball_data.get("TotalSpin", 0.0),
        "carry": carry,
        "total": total,
    }
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    if not _last_data:
        return "<h2>No shot data received.</h2>"
    img_tag = '<img src="/latest.png" alt="trajectory">'
    html = f"""
    <html>
      <body>
        <h2>Shot Data</h2>
        <ul>
          <li>Ball Speed: {_last_data['speed']:.1f} mph</li>
          <li>Launch Angle: {_last_data['launch_angle']:.1f}°</li>
          <li>Backspin: {_last_data['backspin']:.1f} rpm</li>
          <li>Carry: {_last_data['carry']:.1f} yd</li>
          <li>Total Distance: {_last_data['total']:.1f} yd</li>
        </ul>
        {img_tag}
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/latest.png")
async def latest_png():
    if _last_image is None:
        return Response(status_code=404)
    return Response(content=_last_image, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
