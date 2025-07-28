import ctypes
import platform

import numpy as np

import camera.mvsdk as mvsdk


class MVSCamera:
    def __init__(
        self,
        roi_w,
        roi_h,
        roi_x,
        roi_y,
        exposure_us,
        desired_analog_gain,
        desired_gamma,
    ):
        self.roi_w, self.roi_h = roi_w, roi_h
        self.roi_x, self.roi_y = roi_x, roi_y
        self.exposure_us = exposure_us
        self.desired_analog_gain = desired_analog_gain
        self.desired_gamma = desired_gamma
        self.hCamera = None
        self.mono = True
        self.buf = None
        self.buf_sz = 0

    def open(self):
        devs = mvsdk.CameraEnumerateDevice()
        if not devs:
            raise RuntimeError("No camera found")
        self.hCamera = mvsdk.CameraInit(devs[0], -1, -1)
        cap = mvsdk.CameraGetCapability(self.hCamera)
        self.mono = cap.sIspCapacity.bMonoSensor != 0
        fmt = (
            mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self.mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        )
        mvsdk.CameraSetIspOutFormat(self.hCamera, fmt)
        res = mvsdk.CameraGetImageResolution(self.hCamera)
        res.iIndex = 0xFF
        res.iHOffsetFOV = self.roi_x
        res.iVOffsetFOV = self.roi_y
        res.iWidthFOV = self.roi_w
        res.iHeightFOV = self.roi_h
        res.iWidth = self.roi_w
        res.iHeight = self.roi_h
        mvsdk.CameraSetImageResolution(self.hCamera, res)
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, self.exposure_us)

        # Set analog gain
        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(self.hCamera)
        mvsdk.CameraSetAnalogGainX(
            self.hCamera, max(gmin, min(self.desired_analog_gain, gmax))
        )

        # Set gamma
        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(self.hCamera, int(self.desired_gamma * gamma_max))

        mvsdk.CameraPlay(self.hCamera)

    def grab(self, timeout_ms=2000):
        pRaw, head = mvsdk.CameraGetImageBuffer(self.hCamera, timeout_ms)
        if (self.buf is None) or (self.buf_sz != head.uBytes):
            if self.buf:
                mvsdk.CameraAlignFree(self.buf)
            self.buf = mvsdk.CameraAlignMalloc(head.uBytes, 16)
            self.buf_sz = head.uBytes
        mvsdk.CameraImageProcess(self.hCamera, pRaw, self.buf, head)
        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(self.buf, head, 1)
        mvsdk.CameraReleaseImageBuffer(self.hCamera, pRaw)
        frame_data = (ctypes.c_ubyte * head.uBytes).from_address(self.buf)
        if self.mono:
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (head.iHeight, head.iWidth)
            )
        else:
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (head.iHeight, head.iWidth, 3)
            )
        return frame

    def close(self):
        if self.buf:
            mvsdk.CameraAlignFree(self.buf)
            self.buf = None
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None
