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

        # YOLO 객체 감지
        results = yolo_model(frame, verbose=False)[0]
        detected_objects = []

        # 감지된 객체의 이름과 좌표 저장
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            # 바운딩 박스 좌표 및 클래스 이름 추출
            x1, y1, x2, y2 = map(int, box.tolist())
            obj_name = results.names[int(cls.item())]

            # 객체 정보를 딕셔너리로 저장
            detected_objects.append({
                "name": obj_name,
                "coords": [x1, y1, x2, y2]
            })

        # 버퍼에 데이터 추가 (입력 전후 2초 동안 유지)
        if buffer.full():
            buffer.get()  # 오래된 데이터 제거
        buffer.put(detected_objects)

        # YOLO 감지 결과를 프레임에 표시
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임 반환
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
    """LLM으로 객체 선택"""
    global buffer
    data = request.json
    clue = data.get("clue", "").strip()

    if not clue:
        return jsonify({"error": "Clue is missing"}), 400

    # 버퍼에서 2초 전후의 객체 데이터 수집
    detected_objects = list(buffer.queue)  # 버퍼의 모든 데이터 가져오기
    flattened_objects = [obj for sublist in detected_objects for obj in sublist]  # 중첩 리스트 평탄화
    unique_objects = list(set(obj["name"] for obj in flattened_objects))  # 고유한 객체 이름

    # LLM 입력 생성
    user_prompt1 = f'''You are an assistant whose task is to identify the correct object from this list {unique_objects} that aligns best with the following clue: '{clue}'.
    Take a moment to think about each object's properties, considering how the clue relates to them, directly or indirectly.
    Then, provide only the name of the correct object as your answer. No additional explanations.
    The answer is: '''

    messages = [{"role": "user", "content": user_prompt1}]

    print("LLM 입력:", messages)

    # LLM 수행
    start = time.time()
    outputs = pipe(
        messages,
        max_new_tokens=50,
        pad_token_id=pad_token_id,
        do_sample=False,
    )
    end = time.time()

    latency = str(datetime.timedelta(seconds=(end - start)))

    # LLM 결과 추출 및 처리
    print("LLM Outputs:", outputs)
    if (
        isinstance(outputs, list)
        and "generated_text" in outputs[0]
    ):
        full_result = outputs[0]["generated_text"]
    else:
        return jsonify({"error": "Unexpected LLM output format.", "outputs": outputs}), 500

    print("LLM 전체 결과:", full_result)

    # 결과에서 감지된 객체와 매칭되는 이름 추출
    assistant_content = None
    for item in full_result:
        if item.get("role") == "assistant":
            assistant_content = item.get("content")
            break  # 첫 번째 assistant 역할을 찾으면 중단

    print("추출된 content:", assistant_content)

    # 매칭된 객체가 없는 경우 처리
    if not assistant_content:
        return jsonify({
            "answer": "No matching object found.",
            "latency": latency,
            "positions": None
        })

    # 결과와 일치하는 물체의 좌표 검색
    matched_position = None
    x1, y1, x2, y2 = 0, 0, 0, 0
    obj_data = None
    print("감지된 객체:", detected_objects)
    print("추출된 content:", assistant_content)
    for i, frame_objects in enumerate(detected_objects):
        for obj_data in frame_objects:
            print("obj_data:", obj_data)
            print("obj_data['name']:", obj_data["name"].lower())
            print("assistant_content:", assistant_content.lower())

            if obj_data["name"].lower() == assistant_content.lower():
                print ("Matched")
                obj_data = obj_data
                x1, y1, x2, y2 = obj_data["coords"]

                break
        if matched_position:  # 첫 번째 좌표를 찾으면 중단
            print(f"Matched Object: {assistant_content}, Coordinates: {x1}, {y1}, {x2}, {y2}")
            break


    return jsonify({
        "answer": assistant_content,
        "latency": latency,
        "positions": "yes",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })
