import os
import pickle
import sys
import time

import cv2
import numpy as np

# from faster_whisper import WhisperModel 

import queue 
import json 
import sounddevice as sd 
from vosk import Model, KaldiRecognizer 

# Speed 
import multiprocessing as mp
import onnxruntime as ort
session = ort.InferenceSession("/Users/juha/Documents/PYTHON/Fallout/models/face_detection_yunet_2023mar.onnx", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])

# Multiprocessing 
def _transcription_worker(model_path: str, sample_rate: int, block_size: int,
                           result_queue: mp.Queue, stop_event: mp.Event):
    """Runs entirely inside the child process. Imports are done here so the
    parent process doesn't need vosk/sounddevice loaded if it doesn't use them
    directly, and so each process gets its own clean state."""
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
 
    audio_queue: "queue.Queue[bytes]" = queue.Queue()
 
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(bytes(indata))
 
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)
 
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
 
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    result_queue.put(("final", text))
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    result_queue.put(("partial", partial_text))

class TranscriptionProcess:
 
    def __init__(self, model_path: str = "model", sample_rate: int = 16000,
                 block_size: int = 8000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._result_queue: mp.Queue = mp.Queue()
        self._stop_event = mp.Event()
        self._process: mp.Process | None = None
 
    def start(self):
        self._process = mp.Process(
            target=_transcription_worker,
            args=(self.model_path, self.sample_rate, self.block_size,
                  self._result_queue, self._stop_event),
            daemon=True,
        )
        self._process.start()
 
    def poll(self):
        """Non-blocking generator yielding (kind, text) tuples where kind is
        'final' or 'partial'. Call this once per frame in your main loop."""
        while True:
            try:
                yield self._result_queue.get_nowait()
            except queue.Empty:
                return
 
    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()

# Serial communication with turret servos 
from servo_link import ServoLink 

link = ServoLink("/dev/tty.usbmodem101")

last_fire_time = 0 
FIRE_COOLDOWN = 3.0 

# Config
# 
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
DB_PATH = os.path.join(os.path.dirname(__file__), "known_faces.pkl")

# SFace's documented cosine-similarity threshold for "same person" (from
# OpenCV's own face recognition sample). Tune if you see false matches/misses.
COSINE_MATCH_THRESHOLD = 0.363

SAMPLES_TO_CAPTURE = 15
CAPTURE_INTERVAL_SEC = 0.3  # pause between enrollment captures for pose variety

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

#model = WhisperModel("tiny", device="cpu", compute_type="int8")
MODEL_PATH = "/Users/juha/Documents/PYTHON/Fallout/speech-model"
SAMPLE_RATE = 16000 
BLOCK_SIZE = 8000 

audio_queue = queue.Queue() 

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block; push raw bytes to the queue."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


# XIAO ESP32s3-sense used as webcam (picam and continuity camera offered as alternatives)
class Camera:
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, usb_index=0):
        self.use_picamera2 = False
        self.picam2 = None
        self.cap = None

        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.use_picamera2 = True
            print("[camera] Using Raspberry Pi Camera Module via picamera2")
        except Exception:
            backends = [("default", 0)]
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append(("DSHOW", cv2.CAP_DSHOW))
            if hasattr(cv2, "CAP_AVFOUNDATION"):
                backends.append(("AVFOUNDATION", cv2.CAP_AVFOUNDATION))

            self.cap = None
            for index in [usb_index, 0, 1, 2]:
                for backend_name, backend in backends:
                    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            self.cap = cap
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            print(f"[camera] Using webcam via cv2.VideoCapture "
                                  f"(index={index}, backend={backend_name})")
                            # Warm-up 
                            for _ in range(5):
                                self.cap.read()
                                time.sleep(0.05)
                            break
                    cap.release()
                if self.cap is not None:
                    break

            if self.cap is None:
                raise RuntimeError(
                    "No Pi Camera (picamera2) found and no webcam could be "
                    "opened. On Mac: check System Settings > Privacy & "
                    "Security > Camera and allow your terminal/IDE, then "
                    "restart it. On Windows: make sure no other app "
                    "(Zoom, Teams, browser tab) is using the webcam. Also "
                    "try unplugging/replugging any external camera."
                )

    def read(self):
        if self.use_picamera2:
            frame = self.picam2.capture_array()
            # RGB888 -> BGR 
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # Retry a few times 
            for _ in range(10):
                ok, frame = self.cap.read()
                if ok:
                    return frame
                time.sleep(0.05)
            raise RuntimeError("Failed to read frame from webcam.")

    def release(self):
        if self.use_picamera2:
            self.picam2.stop()
        elif self.cap is not None:
            self.cap.release()


# YuNet (detection) + SFace (recognition) 
class FaceEngine:
    def __init__(self, input_size=(320, 320), score_threshold=0.9):
        if not os.path.exists(DETECTOR_PATH) or not os.path.exists(RECOGNIZER_PATH):
            raise FileNotFoundError(
                "Model files not found. Run ./download_models.sh first to "
                f"fetch them into {MODELS_DIR}/"
            )

        self.detector = cv2.FaceDetectorYN.create(
            DETECTOR_PATH,
            "",
            input_size,
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")

    def set_input_size(self, width, height):
        # Call once per frame size 
        self.detector.setInputSize((width, height))

    def detect(self, frame):
        # Returns one row per detected face:
        # [x, y, w, h, <5 landmark x/y pairs>, confidence_score]
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15))

    def get_embedding(self, frame, face_row):
        # Align + crop the face and return its embedding vector 
        aligned = self.recognizer.alignCrop(frame, face_row)
        return self.recognizer.feature(aligned)

    def compare(self, embedding1, embedding2):
        # Similarity between two embeddings 
        return self.recognizer.match(
            embedding1, embedding2, cv2.FaceRecognizerSF_FR_COSINE
        )

# Store names and embeddings 
class FaceDatabase:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def add(self, name, embedding):
        self.data.setdefault(name, []).append(embedding)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def identify(self, engine: FaceEngine, embedding, threshold=COSINE_MATCH_THRESHOLD):
        # Compare an embedding against every stored sample for every known person. Returns (name, score) for the best match, or ("Unknown", best_score) if nothing clears the threshold.
        
        best_name, best_score = "Unknown", -1.0
        for name, embeddings in self.data.items():
            for known_embedding in embeddings:
                score = engine.compare(embedding, known_embedding)
                if score > best_score:
                    best_score = score
                    best_name = name if score >= threshold else "Unknown"
        return best_name, best_score


# Commands
def run_enroll(name):
    engine = FaceEngine()
    db = FaceDatabase()
    cam = Camera()

    print(f"[enroll] Capturing {SAMPLES_TO_CAPTURE} samples for '{name}'.")
    print("[enroll] Slowly turn/tilt your head between captures for variety.")

    embeddings = []
    last_capture = 0.0

    try:
        while len(embeddings) < SAMPLES_TO_CAPTURE:
            frame = cam.read()
            h, w = frame.shape[:2]
            engine.set_input_size(w, h)
            faces = engine.detect(frame)

            display = frame.copy()
            face_row = None
            if len(faces) > 0:
                # Use the largest detected face (closest to the camera).
                face_row = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = face_row[:4].astype(int)
                cv2.rectangle(display, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            cv2.putText(
                display,
                f"Captured: {len(embeddings)}/{SAMPLES_TO_CAPTURE}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Enroll", display)

            now = time.time()
            if face_row is not None and (now - last_capture) > CAPTURE_INTERVAL_SEC:
                embedding = engine.get_embedding(frame, face_row)
                embeddings.append(embedding)
                last_capture = now
                print(f"[enroll] Captured sample {len(embeddings)}/{SAMPLES_TO_CAPTURE}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[enroll] Cancelled, nothing saved.")
                cam.release()
                cv2.destroyAllWindows()
                sys.exit(1)
    finally:
        cam.release()
        cv2.destroyAllWindows()

    for embedding in embeddings:
        db.add(name, embedding)
    db.save()
    print(f"[enroll] Saved {len(embeddings)} embeddings for '{name}' to {db.path}")


def run_recognize():
    global last_fire_time 
    engine = FaceEngine()
    db = FaceDatabase()

    if not db.data:
        print("[recognize] No known faces enrolled yet. Run 'enroll' first.")
        return

    cam = Camera()

    button_click = {"choice": None}
    pending = {"name": None}   # which name is currently awaiting a click
    resolved = set()           # names already confirmed or rejected this session

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and pending["name"] is not None:
            if 10 <= x <= 110 and CAMERA_HEIGHT - 60 <= y <= CAMERA_HEIGHT - 20:
                print(f"[recognize] Confirmed: {pending['name']}")
                resolved.add(pending["name"])
                pending["name"] = None
            elif 130 <= x <= 230 and CAMERA_HEIGHT - 60 <= y <= CAMERA_HEIGHT - 20:
                print(f"[recognize] Rejected match: {pending['name']}")
                resolved.add(pending["name"])
                pending["name"] = None

    cv2.namedWindow("Recognize")
    cv2.setMouseCallback("Recognize", on_mouse)

    # Audio
    transcriber = TranscriptionProcess(model_path="/Users/juha/Documents/PYTHON/Fallout/speech-model-better")
    transcriber.start()

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16", channels=1, callback=audio_callback,):
        print("Listening...")
        try:
            while True:
                frame = cam.read()
                h, w = frame.shape[:2]
                engine.set_input_size(w, h)
                faces = engine.detect(frame)
                names_in_frame = set()

                for face_row in faces:
                    x, y, fw, fh = face_row[:4].astype(int)
                    embedding = engine.get_embedding(frame, face_row)
                    name, score = db.identify(engine, embedding)
                    names_in_frame.add(name)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                    label = f"{name} ({score:.2f})"
                    cv2.putText(
                        frame,
                        label,
                        (x, max(y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                    if name == "Unknown" and name not in resolved:
                        pending["name"] = name
                        frame_h = frame.shape[0]
                        cv2.putText(frame, "Unauthorized user. Obliterate?", (x, y + fh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (10, frame_h - 60), (110, frame_h - 20), (0, 200, 0), -1)
                        cv2.putText(frame, "YES", (30, frame_h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.rectangle(frame, (130, frame_h - 60), (230, frame_h - 20), (0, 0, 200), -1)
                        cv2.putText(frame, "NO", (155, frame_h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    elif name == "Unknown": 
                        print("?")
                    else: 
                        cv2.putText(frame, "Authorized user", (x, y + fh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if "Unknown" not in names_in_frame:
                    resolved.discard("Unknown")
                    if pending["name"] == "Unknown":
                        pending["name"] = None
                
                cv2.imshow("Recognize", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                for kind, text in transcriber.poll():
                    if "fire" in text.lower():
                        now = time.time()
                        if now - last_fire_time > FIRE_COOLDOWN:
                            print("Firing")
                            link.fire(True)
                            last_fire_time = now
                    #if kind == "final":
                    #    print(f"Final: {text}")
                    #else:
                    #    print(f"...{text}", end="\r")
                    #if "fire" in text.lower(): 
                    #    print("Firing") 
                    #    link.set_auto_mode(True) 
                    #    link.fire(True) 
                    #    text = "" 

        finally: 
            cam.release()
            transcriber.stop()
            cv2.destroyAllWindows()
    
    


# Entry point
def print_usage():
    print("Usage:")
    print('  python3 face_recognition_app.py enroll "Name"')
    print("  python3 face_recognition_app.py recognize")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "enroll":
        if len(sys.argv) != 3:
            print_usage()
            sys.exit(1)
        run_enroll(sys.argv[2])
    elif command == "recognize":
        run_recognize()
    else:
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()