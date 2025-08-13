#!/usr/bin/env python3
import queue
import threading
import time

import mvsdk
from config_reader import CONFIG

ROI_W = CONFIG.getint("Camera", "roi_w", fallback=640)
ROI_H = CONFIG.getint("Camera", "roi_h", fallback=280)
ROI_X = CONFIG.getint("Camera", "roi_x", fallback=0)
ROI_Y = CONFIG.getint("Camera", "roi_y", fallback=120)

# how many frames total to grab
TOTAL_FRAMES = CONFIG.getint("Benchmark", "total_frames", fallback=1600)
# how many worker threads you want
WORKERS = CONFIG.getint("Benchmark", "workers", fallback=4)


def worker(hCamera, work_q):
    """
    Pull a (pRaw, head) off the queue, call CameraImageProcess,
    then release/recycle and loop.
    """
    while True:
        pRaw, head = work_q.get()
        if pRaw is None:
            work_q.task_done()
            break

        # each worker allocates its own aligned buffer
        buf = mvsdk.CameraAlignMalloc(head.uBytes, 16)
        try:
            mvsdk.CameraImageProcess(hCamera, pRaw, buf, head)
        finally:
            # always release the raw buffer back to SDK
            mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)
            # free our local processing buffer
            mvsdk.CameraAlignFree(buf)
            work_q.task_done()


def main():
    # 1) camera init & ROI/exposure exactly as before
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found")
        return
    hCam = mvsdk.CameraInit(devs[0], -1, -1)

    try:
        cap = mvsdk.CameraGetCapability(hCam)
        mono = bool(cap.sIspCapacity.bMonoSensor)
        fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(hCam, fmt)

        res = mvsdk.CameraGetImageResolution(hCam)
        res.iIndex = 0xFF
        res.iHOffsetFOV = ROI_X
        res.iVOffsetFOV = ROI_Y
        res.iWidthFOV = ROI_W
        res.iHeightFOV = ROI_H
        res.iWidth = ROI_W
        res.iHeight = ROI_H
        mvsdk.CameraSetImageResolution(hCam, res)

        mvsdk.CameraSetTriggerMode(hCam, 0)
        mvsdk.CameraSetAeState(hCam, 0)
        mvsdk.CameraSetExposureTime(
            hCam, CONFIG.getint("Camera", "exposure_us", fallback=50)
        )  # 0.05 ms

        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCam)
        mvsdk.CameraSetAnalogGainX(hCam, max(gmin, min(1000, gmax)))
        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(hCam, int(0.25 * gamma_max))

        mvsdk.CameraPlay(hCam)

        # 2) set up our frame‐processing thread pool
        work_q = queue.Queue(maxsize=WORKERS * 2)
        threads = []
        for _ in range(WORKERS):
            t = threading.Thread(target=worker, args=(hCam, work_q), daemon=True)
            t.start()
            threads.append(t)

        # 3) grab loop never blocks on processing
        start = time.perf_counter()
        for _ in range(TOTAL_FRAMES):
            pRaw, head = mvsdk.CameraGetImageBuffer(hCam, 2000)
            work_q.put((pRaw, head))

        # 4) tell workers to exit
        for _ in threads:
            work_q.put((None, None))
        work_q.join()
        elapsed = time.perf_counter() - start

        print(
            f"Captured+processed {TOTAL_FRAMES} frames in {elapsed:.3f}s → {TOTAL_FRAMES/elapsed:.2f} FPS"
        )

    finally:
        mvsdk.CameraUnInit(hCam)


if __name__ == "__main__":
    main()
