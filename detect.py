# live_crop_health_streaming_v6.py
import os
import cv2
import json
import time
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from queue import Queue

# -------------------- Paths / Config --------------------
RESULTS_DIR = "results"
DETECTIONS_DIR = os.path.join(RESULTS_DIR, "detections")
LOG_CSV = os.path.join(RESULTS_DIR, "predictions_log.csv")
CLASS_MAP_JSON = os.path.join(RESULTS_DIR, "class_indices.json")
MODEL_PATH = os.path.join(RESULTS_DIR, "sesame_densenet_best.keras")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# --- Model & Display Configuration ---
IMG_SIZE = (300, 300)
CONF_THRESH = 0.75
UNHEALTHY_SNAPSHOT_THRESH = 0.80
CAP_WIDTH, CAP_HEIGHT = 1280, 720

EXTERNAL_PREPROCESS = False
if EXTERNAL_PREPROCESS:
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
else:
    densenet_preprocess = None

# -------------------- Load Model & Classes --------------------
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
_ = model(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32), training=False)

num_classes = int(model.outputs[0].shape[-1])
try:
    with open(CLASS_MAP_JSON, "r") as f:
        name_to_idx = json.load(f)
    idx_to_name = {int(v): k for k, v in name_to_idx.items()}
    if len(idx_to_name) != num_classes:
        raise ValueError("Class map length doesn't match model output.")
except Exception as e:
    print(f"Warning: Could not load class_indices.json. Using generic labels. Error: {e}")
    idx_to_name = {i: f"Class_{i}" for i in range(num_classes)}

# --- UI Colors ---
COLORS = {}
for i, name in idx_to_name.items():
    clean_name = name.lower().strip().replace("_", " ")
    if clean_name == "healthy":
        COLORS[i] = (0, 255, 0)
    elif clean_name == "unhealthy":
        COLORS[i] = (0, 0, 255)
    elif clean_name == "not a crop":
        COLORS[i] = (255, 191, 0)
    else:
        COLORS[i] = (255, 255, 255)


# -------------------- Camera Utilities --------------------
def probe_cameras(max_index=10):  # <-- CHANGE: Increased search range
    """Finds and returns a list of available camera indices."""
    found = []
    print("👀 Probing for available cameras...")
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def open_camera(index, width=1280, height=720):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# -------------------- Non-blocking Inference Worker --------------------
class Predictor:
    def __init__(self, model, img_size, preprocess_func=None):
        self.model = model
        self.img_size = img_size
        self.preprocess = preprocess_func
        self.in_q = Queue(maxsize=1)
        self.out = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.running = False

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def submit(self, frame, ts_frame):
        if not self.in_q.empty():
            try:
                self.in_q.get_nowait()
            except Exception:
                pass
        self.in_q.put((frame, ts_frame))

    def _loop(self):
        while self.running:
            try:
                frame, ts_frame = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32)
            if self.preprocess is not None:
                x = self.preprocess(x)
            x = np.expand_dims(x, 0)
            preds = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            self.out = (idx, conf, ts_frame)


# -------------------- UI Utilities --------------------
def put_text(img, text, org, scale=0.9, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# -------------------- Main LIVE Loop --------------------
available_cams = probe_cameras()
# <-- CHANGE: Added a clear message to show what cameras were found
if not available_cams:
    print("❌ No usable cameras found. Please check connections and drivers.")
    raise SystemExit(1)

print(f"Found {len(available_cams)} usable camera(s): {available_cams}")

cam_pos = 0
cur_cam_idx = available_cams[cam_pos]
cap = open_camera(cur_cam_idx, CAP_WIDTH, CAP_HEIGHT)
if cap is None:
    print(f"Failed to open initial camera index {cur_cam_idx}.")
    raise SystemExit(1)

pred = Predictor(model, IMG_SIZE, preprocess_func=densenet_preprocess)
pred.start()

print(f"🎥 LIVE on camera {cur_cam_idx}")
print("Press [c] to switch camera | [s] to save snapshot | [q] to quit")

log_rows = []
frame_id = 0
t_prev = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed. Attempting to reconnect...")
            cap.release();
            time.sleep(1)
            cap = open_camera(cur_cam_idx, CAP_WIDTH, CAP_HEIGHT)
            if cap is None: break
            continue

        frame_id += 1
        pred.submit(frame, frame_id)
        out = pred.out

        status_text, status_color, predicted_label = "Initializing...", (200, 200, 200), ""

        if out is not None:
            idx, conf, _ = out
            predicted_label = idx_to_name.get(idx, "Unknown")
            clean_label = predicted_label.lower().strip().replace("_", " ")

            if conf < CONF_THRESH:
                status_text = f"UNCERTAIN ({predicted_label}: {conf:.2f})"
                status_color = (0, 215, 255)
            else:
                status_color = COLORS.get(idx, (255, 255, 255))
                if clean_label == "not a crop":
                    status_text = f"Status: Not a Crop ({conf:.2f})"
                elif clean_label == "healthy":
                    status_text = f"Status: Healthy ({conf:.2f})"
                elif clean_label == "unhealthy":
                    status_text = f"Status: Unhealthy ({conf:.2f})"
                else:
                    status_text = f"Status: {predicted_label} ({conf:.2f})"

        if "unhealthy" in predicted_label.lower() and out and out[1] >= UNHEALTHY_SNAPSHOT_THRESH:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(DETECTIONS_DIR, f"UnHealthy_{ts}.jpg")
            cv2.imwrite(path, frame)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - t_prev)))
        t_prev = now

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        put_text(frame, f"LIVE | CAM: {cur_cam_idx} | FPS: {fps:.1f}", (10, 28), 0.7, (255, 255, 255))

        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        put_text(frame, status_text, (10, h - 20), 0.9, status_color)

        cv2.imshow("Crop Health Analysis", frame)

        if out is not None:
            log_rows.append({
                "Timestamp": datetime.now().strftime("%Y%m%d %H%M%S"), "FrameID": frame_id,
                "CameraIndex": cur_cam_idx, "PredIndex": out[0], "PredLabel": predicted_label,
                "Confidence": round(out[1], 6)
            })

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cap.release()
            cam_pos = (cam_pos + 1) % len(available_cams)
            cur_cam_idx = available_cams[cam_pos]
            print(f"Switching to camera index: {cur_cam_idx} (List position: {cam_pos})")
            cap = open_camera(cur_cam_idx, CAP_WIDTH, CAP_HEIGHT)
            if cap is None:
                print(f"Failed to switch to camera {cur_cam_idx}. Exiting.")
                break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(DETECTIONS_DIR, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"Snapshot saved to {path}")

finally:
    print("\nShutting down...")
    if 'cap' in locals() and cap is not None: cap.release()
    if 'pred' in locals(): pred.stop()
    cv2.destroyAllWindows()

    if log_rows:
        df = pd.DataFrame(log_rows)
        df.to_csv(LOG_CSV, index=False)
        print(f"Predictions log saved: {LOG_CSV}")

print("Goodbye.")