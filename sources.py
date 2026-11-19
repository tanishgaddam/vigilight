"""Video input. One reader thread per camera keeps only the newest frame, so a slow
detector never builds up lag. Works with USB indexes, HTTP/MJPEG, RTSP and video files."""
import os
import threading
import time

import cv2


class VideoSource:
    def __init__(self, source):
        self.source = source
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.frame = None
        self.frame_id = 0
        self.online = False
        self.running = False
        self._lock = threading.Lock()
        self._cap = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.running = True
        self._thread.start()

    def stop(self):
        self.running = False

    def get(self):
        with self._lock:
            return self.frame, self.frame_id

    def _set(self, frame):
        with self._lock:
            self.frame = frame
            if frame is not None:
                self.frame_id += 1
            self.online = frame is not None

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        delay = 0.0
        while self.running:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self._set(None)
                    time.sleep(2.0)
                    continue
                if self.is_file:
                    delay = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file:  # loop recorded clips
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                self._cap.release()
                self._cap = None
                self._set(None)
                time.sleep(1.0)
                continue
            self._set(frame)
            if delay:
                time.sleep(delay)
        if self._cap is not None:
            self._cap.release()


def scan_usb(max_index=6, skip=()):
    """Probe USB camera indexes. Cameras already in use by VigiLight are skipped."""
    found = []
    for i in range(max_index):
        if i in skip:
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(i)
    return found
