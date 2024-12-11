from flask import Blueprint, render_template, request, jsonify, Response
from .camera import Camera, list_cameras
from ultralytics import YOLO
import cv2
import threading
import torch
from transformers import pipeline
import time
import datetime
from huggingface_hub import login
import os
from queue import Queue

# Import User Defined Functions
from app.llm_service import generate_llm_response
from app.object_processing import process_detected_objects
from app.control_robot_main import control_robot_main


main = Blueprint('main', __name__)

# ----------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# YOLO 모델 로드
yolo_model = YOLO("models/yolo11x.pt")
yolo_model.fuse()

# LLM 모델 로드
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
pad_token_id = pipe.model.config.eos_token_id
if isinstance(pad_token_id, list):
    pad_token_id = pad_token_id[0]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

yolo_model.to(device)
pipe.model.to(device)
print(f"YOLO/LLM 모델이 {device}에서 실행됩니다.")

# ----------------------------------------------

camera_instance = None
camera_search_results = []
buffer = Queue(maxsize=60)


def list_cameras():
    """사용 가능한 카메라 장치를 검색"""
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras


def camera_stream():
    global buffer
    while True:
        ret, frame = camera_instance.camera.read()
        if not ret:
            continue

        # YOLO detection
        results = yolo_model(frame, verbose=False)[0]
        detected_objects = []

        # save detected objects in a list
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            obj_name = results.names[int(cls.item())]

            detected_objects.append({
                "name": obj_name,
                "coords": [x1, y1, x2, y2]
            })

        # Add detected objects to the buffer (2 seconds)
        if buffer.full(): buffer.get() # remove the oldest frame
        buffer.put(detected_objects)

        # Draw detected objects on the frame
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame to JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')




def detect_and_draw_objects(frame):
    results = yolo_model(frame, verbose=False)[0]
    for box in results.boxes:
        # 바운딩 박스 좌표, 클래스 이름 및 신뢰도 가져오기
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)  # 클래스 ID
        conf = box.conf.item()  # 신뢰도
        label = f"{results.names[cls]} {conf:.2f}"  # 클래스 이름 + 신뢰도

        # 바운딩 박스와 클래스 이름 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def search_cameras():
    """카메라 검색 작업"""
    global camera_search_results
    camera_search_results = list_cameras()


@main.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@main.route('/start_camera_search', methods=['POST'])
def start_camera_search():
    """카메라 검색 시작"""
    thread = threading.Thread(target=search_cameras)
    thread.start()
    return jsonify({"status": "searching"})


@main.route('/get_camera_results', methods=['GET'])
def get_camera_results():
    """카메라 검색 결과 반환"""
    global camera_search_results
    if camera_search_results:
        return jsonify({"status": "completed", "cameras": camera_search_results})
    return jsonify({"status": "searching"})


@main.route('/connect_camera', methods=['POST'])
def connect_camera():
    """카메라 연결"""
    global camera_instance
    camera_index = int(request.json.get("camera_index"))

    try:
        if camera_instance:
            camera_instance.camera.release()  # 기존 카메라 해제
        camera_instance = Camera(camera_index)
        return jsonify({"status": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@main.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route("/generate_answer", methods=["POST"])
def generate_answer():
    global buffer
    
    # Receive data from the request --------------------------------
    # Clue & Exclude Objects
    data = request.json
    clue = data.get("clue", "").strip()
    exclude_objects = data.get("exclude", [])

    if not clue: return jsonify({"error": "Clue is missing"}), 400
    # ---------------------------------------------------------------

    # (1) YOLO + LLM ver.
    filtered_objects, unique_objects = process_detected_objects(exclude_objects, list(buffer.queue))
    answer, latency, matched_position = generate_llm_response(clue, unique_objects, filtered_objects)


    if not matched_position:
        return jsonify({
            "answer": answer or "No matching object found.",
            "latency": latency,
            "positions": None
        })

    x1, y1, x2, y2 = matched_position
    return jsonify({
        "answer": answer,
        "latency": latency,
        "positions": "yes",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })


@main.route("/control_robot", methods=["POST"])
def control_robot():
    control_robot_main()
    return jsonify({"status": "success"})