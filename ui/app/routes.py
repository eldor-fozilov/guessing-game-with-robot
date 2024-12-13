from flask import Blueprint, g, render_template, request, jsonify, Response
from .utils.camera import Camera, list_cameras
import cv2
import threading
import torch
import time
import whisper
from queue import Queue
from app.utils.proces_audio import understand, speak
import struct
import wave
from pvrecorder import PvRecorder
import numpy as np
from app.robot_control.guesser import GuessingBot
from app.utils.load_models import load_yolo_model, load_llm_model, load_vlm_model, load_yolo_world_model, load_yolo_model
from app.utils.inference_models import generate_llm_response, generate_vlm_response, yolo_world_detect
from app.utils.process_object import process_detected_objects
from TTS.api import TTS
import mujoco
from PIL import Image

main = Blueprint('main', __name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load Whisper model for STT
stt_model = whisper.load_model("base")
print("STT model loaded.")

# Load TTs model for TTS
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

if device == 'cpu':
    tts_model = TTS(tts_model_name, gpu=False)
else:
    tts_model = TTS(tts_model_name, gpu=True)

print("TTS model loaded.")

use_robot = False

if use_robot:

    end_effector = 'joint6'

    np.set_printoptions(precision=6, suppress=True)

    # Define model and data in mujoco
    urdf_path = './app/robot_control/low_cost_robot/scene.xml'
    device_name = '/dev/ttyACM0'

    robot = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(robot)

    # Make GueesingBot
    guesser = GuessingBot(
        model=robot,
        data=data,
        device_name=device_name,
        end_effector=end_effector
    )

    # Move real robot to home position
    guesser.move_to_home()


# variables for tracking
camera_instance = None
camera_search_results = []
buffer = Queue(maxsize=60)
last_llm_answer = None
last_llm_explanation = None
last_select_object_coords = None
last_transcription = None
stop_recording_flag = False
recording_thread = None
current_pipeline = None
camera_screenshot = None

# variables for models
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_tokenizer = None
yolo_model = None
yolo_world_model = None


@main.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


def search_cameras():
    """카메라 검색 작업"""
    global camera_search_results
    camera_search_results = list_cameras()


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


def camera_stream():
    global buffer, current_pipeline, camera_instance, yolo_model, last_llm_answer
    while True:
        ret, frame = camera_instance.camera.read()
        if not ret:
            continue

        detected_objects = []
        data = request.json
        wrong_answer = data.get("wrongAnswer", False)

        if current_pipeline == "YOLO and LLM":
            # YOLO 객체 감지
            results = yolo_model(frame, verbose=False)[0]

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

            if not wrong_answer and last_llm_answer:
                # only show the selected object
                detected_objects = [
                    obj for obj in detected_objects if obj["name"].lower() == last_llm_answer.lower()]

        elif current_pipeline == "VLM and YOLO World" and last_llm_answer and not wrong_answer:
            boxes, labels, label_texts, scores = yolo_world_detect(runner=yolo_world_model,
                                                                   object_description=last_llm_answer, input_image=frame)

            detected_objects = []
            for box, label, label_text, score in zip(boxes, labels, label_texts, scores):
                x1, y1, x2, y2 = map(int, box)
                detected_objects.append({
                    "name": label_text,
                    "coords": [x1, y1, x2, y2]
                })

        else:
            return jsonify({"error": "No valid solution pipeline selected."}), 400

        # 버퍼에 데이터 추가 (입력 전후 2초 동안 유지)
        if buffer.full():
            buffer.get()  # 오래된 데이터 제거
        buffer.put(detected_objects)

        # YOLO 감지 결과를 프레임에 표시
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj["name"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임 반환
        _, encoded_frame = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')


@main.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route("/generate_answer", methods=["POST"])
def generate_answer():
    global buffer, camera_instance, last_llm_answer, last_llm_explanation, last_transcription, current_pipeline, last_select_object_coords
    global yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer
    data = request.json
    clue = data.get("clue", "").strip()
    excluded_objects = data.get("rejectedObjects", [])

    if not clue:
        if last_transcription and last_transcription.strip():
            clue = last_transcription.strip()
        else:
            return jsonify({"error": "No clue provided. Either type text or record audio first."}), 400

    matched_position = None

    if current_pipeline == "YOLO and LLM":

        detected_objects = list(buffer.queue)

        filtered_objects, unique_objects = process_detected_objects(
            excluded_objects, detected_objects)

        selected_object, latency, explanation = generate_llm_response(
            llm_model, llm_tokenizer, clue, unique_objects)

        for obj_data in filtered_objects:
            if isinstance(obj_data, dict) and obj_data["name"].lower() == selected_object.lower():
                matched_position = obj_data["coords"]
                break

    elif current_pipeline == "VLM and YOLO World":

        ret, frame = camera_instance.camera.read()

        if not ret:
            return jsonify({"error": "Failed to capture frame."}), 500

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        selected_object, latency, explanation = generate_vlm_response(
            vlm_model, vlm_tokenizer, pil_image, clue, excluded_objects)

        boxes, labels, label_texts, scores = yolo_world_detect(
            runner=yolo_world_model, object_description=selected_object, input_image=pil_image)

        box = boxes[0]
        x1, y1, x2, y2 = map(int, box)
        matched_position = [x1, y1, x2, y2]

    else:
        return jsonify({"error": "No valid solution pipeline selected."}), 400

    last_llm_answer = selected_object
    last_llm_explanation = explanation

    # if not matched_position:
    #     return jsonify({
    #         "answer": "No matching object found.",
    #         "latency": latency,
    #         "positions": None
    #     })

    x1, y1, x2, y2 = matched_position
    last_select_object_coords = matched_position

    return jsonify({
        "answer": selected_object,
        "latency": latency,
        "positions": "yes",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })


@main.route('/select_solution', methods=['POST'])
def select_solution():
    global current_pipeline, yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer
    data = request.get_json()
    solution = data.get('solution')
    if not solution:
        return jsonify({"error": "No solution provided"}), 400

    # Set the global variable
    current_pipeline = solution

    if solution == "YOLO and LLM":
        # Load the YOLO model
        yolo_model = load_yolo_model(device="cpu")
        print("YOLO model loaded.")
        # Load the LLM model
        llm_model, llm_tokenizer = load_llm_model(device=device)
        print("LLM model loaded.")
    elif solution == "VLM and YOLO World":
        # Load the VLM model
        vlm_model, vlm_tokenizer = load_vlm_model(device=device)
        print("VLM model loaded.")
        # Load the YOLO World model
        yolo_world_model = load_yolo_world_model(device="cpu")
        print("YOLO World model loaded.")
    else:
        return jsonify({"error": "Invalid solution"}), 400

    return jsonify({"status": "solution set", "current_solution": current_pipeline})


def listen(max_duration=60, device_index=0, output_path='tmp.wav'):
    global stop_recording_flag
    stop_recording_flag = False

    recorder = PvRecorder(frame_length=1024, device_index=device_index)
    recorder.start()

    if output_path is not None:
        wavfile = wave.open(output_path, "wb")
        wavfile.setparams((1, 2, recorder.sample_rate, 0, "NONE", "NONE"))
    else:
        wavfile = None

    st = time.time()
    print("=======Start Listening")

    while True:
        if stop_recording_flag:
            print("=======Stopped by user")
            break
        frame = recorder.read()
        if wavfile is not None:
            wavfile.writeframes(struct.pack("h" * len(frame), *frame))
        if time.time()-st > max_duration:
            print("=======Stopping Listening due to max_duration")
            break

    recorder.delete()
    if wavfile is not None:
        wavfile.close()


@ main.route('/start_listen', methods=['POST'])
def start_listen():
    global recording_thread
    if recording_thread and recording_thread.is_alive():
        return jsonify({"status": "already recording"}), 400
    # Start the listen function in a separate thread
    recording_thread = threading.Thread(
        target=listen, kwargs={"max_duration": 60, "output_path": "recorded_audio.wav"})
    recording_thread.start()
    return jsonify({"status": "recording started"})


@ main.route('/stop_listen', methods=['POST'])
def stop_listen():
    global stop_recording_flag, recording_thread, whisper_model, device, stt_model
    stop_recording_flag = True
    if recording_thread:
        recording_thread.join()
        recording_thread = None

    # After recording stops, we have recorded_audio.wav
    # Now run understand() to get transcription
    # Make sure you have whisper_model already loaded.
    transcription = understand(
        stt_model, device, filename="recorded_audio.wav")

    return jsonify({"status": "recording stopped", "transcription": transcription})


@ main.route('/speak_llm_output', methods=['POST'])
def speak_llm_output():
    global last_llm_answer, last_llm_explanation, tts_model
    if not last_llm_answer:
        return jsonify({"error": "No LLM answer to speak."}), 400

    # Convert LLM answer to speech and play it
    full_response = f"The correct object is {last_llm_answer}. {last_llm_explanation}"
    speak(tts_model, full_response)

    return jsonify({"status": "Playing LLM speech"})


@ main.route("/control_robot", methods=["POST"])
def control_robot():
    global last_select_object_coords

    # calibrate the coordinates from 2D to 3D and from camera frame to robot frame
    x1, y1, x2, y2 = last_select_object_coords
    x1_3d, y1_3d, z1_3d = None, None, None  # needs to be specified

    target_point = [x1_3d, y1_3d, z1_3d]

    # move the robot to the target point and pick / place the object
    guesser.move_to_target(target_point)
    guesser.pick_and_place()
    return jsonify({"status": "success"})
