from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from JaySimG_translation.ball import Ball

app = FastAPI(title="Golf Range Visualizer")

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.post('/simulate')
async def simulate(request: Request):
    data = await request.json()
    ball = Ball()
    if data:
        ball.hit_from_data(data)
    else:
        ball.hit()
    positions = []
    delta = 0.01
    max_time = 20.0
    t = 0.0
    while t < max_time:
        ball.update(delta)
        positions.append(ball.position.tolist())
        t += delta
        if ball.position[1] <= 0.0 and np.linalg.norm(ball.velocity) < 0.01:
            break
    return {"positions": positions}


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    uvicorn.run(app, host='0.0.0.0', port=port)
