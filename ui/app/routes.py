from flask import Blueprint, render_template, request, jsonify, Response
from .utils.camera import Camera, list_cameras
import cv2
import threading
import torch
import time

import struct
import wave
from pvrecorder import PvRecorder
# import numpy as np
# from app.robot_control.guesser import GuessingBot
# from app.utils.inference_models import yolo_world_detect
# import mujoco
import requests
import base64
import os

main = Blueprint('main', __name__)

if torch.cuda.is_available():           device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else:                                   device = "cpu"

SERVER_URL = "http://10.20.12.178:5000"

# # =================================================================
# # ROBOT CONTROL (TEST)
# use_robot = False

# if use_robot:
#     end_effector = 'joint6'

#     np.set_printoptions(precision=6, suppress=True)

#     # Define model and data in mujoco
#     urdf_path = './app/robot_control/low_cost_robot/scene.xml'
#     device_name = '/dev/ttyACM0'

#     robot = mujoco.MjModel.from_xml_path(urdf_path)
#     data = mujoco.MjData(robot)

#     # Make GueesingBot
#     guesser = GuessingBot(
#         model=robot,
#         data=data,
#         device_name=device_name,
#         end_effector=end_effector
#     )

#     # Move real robot to home position
#     guesser.move_to_home()
# # =================================================================


# =================================================================
# variables for tracking
camera_instance = None
camera_search_results = []

last_llm_answer = None
last_all_identified_objects = None
last_llm_explanation = None
last_select_object_coords = None
last_transcription = None
stop_recording_flag = False
recording_thread = None
current_pipeline = None
camera_screenshot = None
wrong_answer = False

# variables for models
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_tokenizer = None
yolo_model = None
yolo_world_model = None
# =================================================================


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
    global camera_instance, wrong_answer
    camera_index = int(request.json.get("camera_index"))

    try:
        if camera_instance:
            camera_instance.camera.release()  # 기존 카메라 해제
        camera_instance = Camera(camera_index)
        return jsonify({"status": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def camera_stream():
    global current_pipeline, camera_instance, yolo_model, last_llm_answer, last_select_object_coords, wrong_answer  # buffer
    while True:
        ret, frame = camera_instance.camera.read()
        if not ret:
            continue

        detected_objects = []

        # if current_pipeline == "YOLO and LLM":
        #     # YOLO 객체 감지
        #     results = yolo_model(frame, verbose=False)[0]

        #     # 감지된 객체의 이름과 좌표 저장
        #     for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        #         # 바운딩 박스 좌표 및 클래스 이름 추출
        #         x1, y1, x2, y2 = map(int, box.tolist())
        #         obj_name = results.names[int(cls.item())]

        #         # 객체 정보를 딕셔너리로 저장
        #         detected_objects.append({
        #             "name": obj_name,
        #             "coords": [x1, y1, x2, y2]
        #         })

        #     if not wrong_answer and last_llm_answer:
        #         # only show the selected object
        #         detected_objects = [
        #             obj for obj in detected_objects if obj["name"].lower() == last_llm_answer.lower()]

        # elif current_pipeline == "VLM and YOLO World" and last_llm_answer and not wrong_answer:
        #     boxes, labels, label_texts, scores = yolo_world_detect(runner=yolo_world_model,
        #                                                            object_description=last_llm_answer, input_image=frame)

        #     detected_objects = []
        #     for box, label, label_text, score in zip(boxes, labels, label_texts, scores):
        #         x1, y1, x2, y2 = map(int, box)
        #         detected_objects.append({
        #             "name": label_text,
        #             "coords": [x1, y1, x2, y2]
        #         })

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

# =================================================================
# LLM & VLM
@main.route("/generate_answer", methods=["POST"])
def generate_answer():
    global camera_instance, current_pipeline, last_transcription
    global last_llm_answer, last_llm_explanation, last_select_object_coords, last_all_identified_objects

    data = request.json
    clue = data.get("clue", "").strip()
    excluded_objects = data.get("rejectedObjects", [])

    if not clue:
        if last_transcription and last_transcription.strip():
            clue = last_transcription.strip()
        else:
            return jsonify({"error": "No clue provided. Either type text or record audio first."}), 400
        
    ret, frame = camera_instance.camera.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame."}), 500
    
    _, buffer = cv2.imencode(".jpg", frame)
    frame_encoded = base64.b64encode(buffer).decode("utf-8")

    response = requests.post(f"{SERVER_URL}/generate_answer_server", json={
        "frame": frame_encoded,
        "clue": clue,
        "excluded_objects": excluded_objects
    })

    if response.status_code == 200:
        response_data = response.json()
        last_llm_answer = response_data.get("last_llm_answer")
        last_llm_explanation = response_data.get("last_llm_explanation")
        last_all_identified_objects = response_data.get("last_all_identified_objects")
        last_select_object_coords = response_data.get("last_select_object_coords")

        return jsonify({
            "answer": response_data.selected_object,
            "explanation": response_data.explanation,
            "all_identified_objects": response_data.all_identified_objects,
            "latency": response_data.latency,
            "positions": "yes",
            "x1": response_data.x1,
            "y1": response_data.y1,
            "x2": response_data.x2,
            "y2": response_data.y2,
        })
    else:
        raise RuntimeError(f"Server error: {response.status_code}, {response.text}")
    


@main.route('/select_solution', methods=['POST'])
def select_solution():
    global current_pipeline
    global yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer

    data = request.json
    solution = data.get('solution')
    if not solution:
        return jsonify({"error": "No solution provided"}), 400

    # Set the global variable
    current_pipeline = solution

    response = requests.post(f"{SERVER_URL}/select_solution_server", json={
        "current_pipeline": current_pipeline,
    })

    if response.status_code == 200:
        return jsonify({"status": "solution set", "current_solution": current_pipeline})
    else:
        raise RuntimeError(f"Server error: {response.status_code}, {response.text}")


@ main.route('/wrong_answer_status', methods=['POST'])
def wrong_answer_status():
    global wrong_answer
    wrong_answer = request.json.get("wrongAnswer", False)

    print("Wrong answer flag set to:", wrong_answer)

    return jsonify({"status": "wrong answer flag set"})
# =================================================================



# =================================================================
# TTS & STT
def listen(max_duration=60, device_index=-1, output_path='tmp.wav'):
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
    global stop_recording_flag, recording_thread, whisper_model, device
    stop_recording_flag = True

    if recording_thread:
        recording_thread.join()
        recording_thread = None

    with open("recorded_audio.wav", "rb") as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
    response = requests.post(f"{SERVER_URL}/stop_listen_server", json={
        "audio": audio_base64,
    })

    if response.status_code == 200:
        return jsonify({"status": "recording stopped", "transcription": response.json()["transcription"]})
    else:
        print("Error:", response.json()["error"])


@ main.route('/speak_llm_output', methods=['POST'])
def speak_llm_output():
    global last_llm_answer, last_llm_explanation, last_all_identified_objects
    if not last_llm_answer:
        return jsonify({"error": "No LLM answer to speak."}), 400

    full_response = f"Identified objects are {', '.join(last_all_identified_objects)}. The selected object is {last_llm_answer}. {last_llm_explanation}"

    response = requests.post(f"{SERVER_URL}/speak_llm_output_server", json={
            "full_response": full_response,
        }
    )
    
    if response.status_code == 200:
        audio_base64 = response.json().get("audio")
        if audio_base64:
            os.system(f"aplay tmp.wav")
        else:
            print("No audio data received.")
    else:
        print("Error:", response.json()["error"])
    

    return jsonify({"status": "Playing LLM speech"})
# =================================================================



# =================================================================
# ROBOT
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

    # some additional logic to handle the object

    # move the robot back to the home position
    guesser.move_to_home()

    return jsonify({"status": "success"})
# =================================================================