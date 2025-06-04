import json
import socket
from dataclasses import dataclass, field
from typing import Optional

from .ball import Ball
from .ball_trail import BallTrail


@dataclass
class RangeSim:
    host: str = "0.0.0.0"
    port: int = 49152
    track_points: bool = False
    trail_resolution: float = 0.1
    _trail_timer: float = 0.0
    _apex: float = 0.0
    ball: Ball = field(default_factory=Ball)
    trail: BallTrail = field(default_factory=BallTrail)
    server: socket.socket = field(init=False)
    conn: Optional[socket.socket] = field(default=None, init=False)

    def __post_init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)

    def poll_network(self):
        if self.conn is None:
            try:
                self.conn, _ = self.server.accept()
                self.conn.setblocking(False)
                print("TCP connection accepted")
            except BlockingIOError:
                return
        try:
            data = self.conn.recv(4096)
        except BlockingIOError:
            return
        if not data:
            self.conn.close()
            self.conn = None
            return
        try:
            shot = json.loads(data.decode())
        except json.JSONDecodeError:
            return
        if shot.get("ShotDataOptions", {}).get("ContainsBallData"):
            self.trail.clear_points()
            self.ball.hit_from_data(shot["BallData"])
            self.track_points = True
            self.trail.add_point(self.ball.position.copy())

    def step(self, delta: float):
        self.poll_network()
        self.ball.update(delta)
        if self.track_points:
            self._trail_timer += delta
            if self._trail_timer >= self.trail_resolution:
                self.trail.add_point(self.ball.position.copy())
                self._trail_timer = 0.0
        self._apex = max(self._apex, self.ball.position[1])

    @property
    def distance_yards(self):
        return (self.ball.position[[0,2]] ** 2).sum() ** 0.5 * 1.09361

    @property
    def apex_feet(self):
        return self._apex * 3.0
