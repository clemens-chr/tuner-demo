from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import threading
import time
from handtracker import MediaPipeTracker
import numpy as np

app = FastAPI()

# Initialize the MediaPipeTracker
tracker = MediaPipeTracker(show=True, camera=1)
tracker.start()

if not tracker.running:
    raise RuntimeError("Failed to start the tracker")

print("Tracker started")
# Shared variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()

def capture_frames():
    global latest_frame
    desired_fps = 30
    interval = 1 / desired_fps
    while tracker.running:
        frame = tracker.frame  # Capture frame from tracker
        if frame is not None:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(interval)

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()
print("Capture thread started")

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                # Convert BGR to RGB if needed
                frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # Yield frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # Approximately 30 fps

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/start_recording")
def start_recording():
    tracker.start_recording()
    return "Recording started"

@app.get("/stop_recording")
def stop_recording():
    tracker.stop_recording()
    return "Recording stopped"

@app.on_event("shutdown")
def shutdown_event():
    tracker.stop()
    capture_thread.join()
