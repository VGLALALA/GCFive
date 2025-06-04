# Golf Range Visualization Server

This directory contains a small FastAPI application that exposes the physics model
from `JaySimG_translation` via a web interface. The server computes the trajectory
of a golf ball and streams the positions to a simple Three.js viewer.

## Usage

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
python -m webserver.main
```

Open `http://localhost:8000` in your browser and click **Hit Ball** to simulate a
shot using the default parameters. The ball path will be animated in 3D.
