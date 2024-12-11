from flask import Blueprint, g, render_template, request, jsonify, Response
from .camera import Camera, list_cameras
from ultralytics import YOLO
import cv2
import threading
import torch
from transformers import pipeline
import time
import datetime
import whisper
from huggingface_hub import login
import os
from queue import Queue
from stt_and_tts import listen, understand, speak

main = Blueprint('main', __name__)

# ----------------------------------------------
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# YOLO 모델 로드
yolo_model = YOLO("models/yolo11n.pt")
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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

yolo_model.to(device)
pipe.model.to(device)
print(f"YOLO/LLM 모델이 {device}에서 실행됩니다.")

# ----------------------------------------------

# Load Whisper model for STT
whisper_model = whisper.load_model("tiny")
print("Whisper 모델", whisper_model)

camera_instance = None
camera_search_results = []
buffer = Queue(maxsize=60)

last_llm_answer = None
last_transcription = None

recording_thread = None

# Define a global variable for the current solution pipeline
current_solution = None

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
    global buffer, last_llm_answer, last_transcription, current_solution
    data = request.json
    clue = data.get("clue", "").strip()
    excluded_objects = data.get("excluded_objects", [])

    if not clue:
        if last_transcription and last_transcription.strip():
            clue = last_transcription.strip()
        else:
            return jsonify({"error": "No clue provided. Either type text or record audio first."}), 400

    detected_objects = list(buffer.queue)
    flattened_objects = [obj for sublist in detected_objects for obj in sublist]
    unique_objects = list(set(obj["name"] for obj in flattened_objects))

    # Remove excluded objects
    unique_objects = [obj for obj in unique_objects if obj not in excluded_objects]

    if current_solution == "YOLO and LLM":
        # user_prompt = f"""You are an assistant whose task is to identify the correct object from this list {unique_objects} that aligns best with the following clue: '{clue}'.
        # Some objects have been previously identified as incorrect: {excluded_objects}.
        # Exclude them from consideration.
        # Provide only the name of the correct object as your answer. No additional explanations.
        # The answer is: """

        user_prompt = f"""You are an assistant whose task is to identify the correct object from this list {unique_objects} that aligns best with the following clue: '{clue}'.
        Some objects have been previously identified as incorrect: {excluded_objects}.
        Exclude them from consideration.
        Provide the name of the correct object first as your answer and a very short reason why you chose that object.
        The answer is: """

        messages = [{"role": "user", "content": user_prompt}]
        print("LLM 입력:", messages)

        start = time.time()
        outputs = pipe(
            messages,
            max_new_tokens=50,
            pad_token_id=pad_token_id,
            do_sample=False,
        )
        end = time.time()
        latency = str(datetime.timedelta(seconds=(end - start)))

        print("LLM Outputs:", outputs)
        if isinstance(outputs, list) and "generated_text" in outputs[0]:
            full_result = outputs[0]["generated_text"]
        else:
            return jsonify({"error": "Unexpected LLM output format.", "outputs": outputs}), 500

        print("LLM 전체 결과:", full_result)

        assistant_content = None
        for item in full_result:
            if item.get("role") == "assistant":
                assistant_content = item.get("content")
                break

        if not assistant_content:
            if isinstance(full_result, list):
                assistant_content = "No matching object found."
            else:
                assistant_content = full_result.strip()

        print("추출된 content:", assistant_content)

        x1, y1, x2, y2 = 0, 0, 0, 0
        found_match = False
        for frame_objects in detected_objects:
            for obj_data_iter in frame_objects:
                if obj_data_iter["name"].lower() == assistant_content.lower():
                    x1, y1, x2, y2 = obj_data_iter["coords"]
                    found_match = True
                    break
            if found_match:
                break

        last_llm_answer = assistant_content

        if not found_match:
            return jsonify({
                "answer": assistant_content,
                "latency": latency,
                "positions": None
            })

        return jsonify({
            "answer": assistant_content,
            "latency": latency,
            "positions": "yes",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "obj_data": assistant_content
        })

    elif current_solution == "VLM and YOLO World":
        # Similar logic for VLM and YOLO World pipeline (placeholder)
        # For demonstration, simulate a result that respects excluded_objects
        filtered_unique_objects = [obj for obj in unique_objects if obj not in excluded_objects]
        assistant_content = filtered_unique_objects[0] if filtered_unique_objects else "No matching object found"

        latency = "0:00:01"
        x1, y1, x2, y2 = 0,0,0,0
        found_match = False
        for frame_objects in detected_objects:
            for obj_data_iter in frame_objects:
                if obj_data_iter["name"].lower() == assistant_content.lower():
                    x1, y1, x2, y2 = obj_data_iter["coords"]
                    found_match = True
                    break
            if found_match:
                break

        last_llm_answer = assistant_content

        if not found_match:
            return jsonify({
                "answer": assistant_content,
                "latency": latency,
                "positions": None
            })

        return jsonify({
            "answer": assistant_content,
            "latency": latency,
            "positions": "yes",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "obj_data": assistant_content
        })

    else:
        return jsonify({"error": "No valid solution pipeline selected."}), 400


    
@main.route('/select_solution', methods=['POST'])
def select_solution():
    global current_solution
    data = request.get_json()
    solution = data.get('solution')
    if not solution:
        return jsonify({"error": "No solution provided"}), 400
    
    # Set the global variable
    current_solution = solution
    return jsonify({"status": "solution set", "current_solution": current_solution})


@main.route('/start_listen', methods=['POST'])
def start_listen():
    global recording_thread
    if recording_thread and recording_thread.is_alive():
        return jsonify({"status": "already recording"}), 400
    # Start the listen function in a separate thread
    recording_thread = threading.Thread(target=listen, kwargs={"max_duration": 60, "output_path": "recorded_audio.wav"})
    recording_thread.start()
    return jsonify({"status": "recording started"})

@main.route('/stop_listen', methods=['POST'])
def stop_listen():
    global stop_recording_flag, recording_thread, whisper_model, device
    stop_recording_flag = True
    if recording_thread:
        recording_thread.join()
        recording_thread = None

    # After recording stops, we have recorded_audio.wav
    # Now run understand() to get transcription
    # Make sure you have whisper_model already loaded.
    transcription = understand(whisper_model, device, filename="recorded_audio.wav")

    return jsonify({"status": "recording stopped", "transcription": transcription})

@main.route('/speak_llm_output', methods=['POST'])
def speak_llm_output():
    global last_llm_answer
    if not last_llm_answer:
        return jsonify({"error": "No LLM answer to speak."}), 400

    # Convert LLM answer to speech and play it
    speak(last_llm_answer)

    return jsonify({"status": "Playing LLM speech"})