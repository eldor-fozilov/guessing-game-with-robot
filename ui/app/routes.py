# Standard libraries
import os
import sys
import time
import struct
import wave
import threading

# Third-party libraries
import cv2
import torch
import numpy as np
from flask import Blueprint, g, render_template, request, jsonify, Response
from pvrecorder import PvRecorder
import mujoco

# Project-specific utilities
from app.utils.camera import Camera, list_cameras
from app.utils.process_audio import understand, speak
from app.utils.load_models import load_yolo_model, load_llm_model, load_vlm_model, load_yolo_world_model
from app.utils.inference_models import generate_llm_response, generate_vlm_response, yolo_world_detect
from app.utils.process_object import process_detected_objects
from app.robot_control.guesser import GuessingBot
from app.robot_control.robot import Robot
from app.robot_control.img2world import CalibBoard
from app.robot_control.interface import SimulatedRobot

# Text-to-Speech API
from TTS.api import TTS
import whisper
# ==================================================================


# ==================================================================
# [1] Platform detection & Set device
def detect_os():
    if sys.platform.startswith("darwin"):
        return "macOS"
    elif sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
        return "Windows"
    elif sys.platform.startswith("linux"):
        return "Linux"
    else:
        return "Unknown"


current_os = detect_os()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if current_os == "macOS":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ==================================================================


# ==================================================================
# [2] Set Resolutions
TARGET_RES_WIDTH, TARGET_RES_HEIGHT = 640, 480
CAMERA_RES_WIDTH, CAMERA_RES_HEIGHT = 1920, 1080
# ==================================================================


# Flask Blueprint
main = Blueprint('main', __name__)


# ==================================================================
# [3] Load Models : STT, TTS
# STT
stt_model = whisper.load_model("base")
print("STT model loaded.")

# TTS
# <-> "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
if device == 'cuda':
    tts_model = TTS(tts_model_name, gpu=True)
else:
    tts_model = TTS(tts_model_name, gpu=False)
print("TTS model loaded.")
# ==================================================================


# ==================================================================
# [4] Set GuessingBot
use_robot = True

if use_robot:

    np.set_printoptions(precision=6, suppress=True)
    end_effector = 'joint6'

    # Define model and data in mujoco
    urdf_path = './ui/app/robot_control/low_cost_robot/scene.xml'
    usb_device_name = '/dev/tty.usbmodem58760435011'  # /dev/ttyACM0

    robot_model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(robot_model)

    # Setting Robot
    sim_robot = SimulatedRobot(robot_model, data)
    real_robot = Robot(device_name=usb_device_name)
    guesser = GuessingBot(sim_robot, real_robot, end_effector=end_effector)
    t = CalibBoard()

    # Move real robot to home position
    guesser.move_to_home()
    real_robot._gripper_off()


def run_viewer(robot_model, data):
    """Function to launch the MuJoCo viewer."""
    try:
        with mujoco.viewer.launch_passive(robot_model, data) as viewer:
            while viewer.is_running():
                time.sleep(0.01)  # Avoid excessive CPU usage
    except Exception as e:
        print(f"Viewer thread encountered an error: {e}")

# viewer_thread = threading.Thread(target=run_viewer, args=(robot_model, data), daemon=True)
# viewer_thread.start()

# ==================================================================


# ==================================================================
# [5] Variables
# for tracking
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
moved_to_target_point = False

# variables for models
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_tokenizer = None
yolo_model = None
yolo_world_model = None

# YOLO model paths and configurations
yolo_model_path = "models/yolo11l.pt"
# 'n' for nano, 's' for small, 'm' for medium, 'l' for large, 'x' for extra large
yolo_size = 'l'
# 's' for small, 'm' for medium, 'l' for large, 'x' for extra large
yolo_world_size = 'l'
yolo_world_model_path = "./YOLO-World/pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
yolo_world_model_config_path = "./YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"

# ==================================================================


# ==================================================================
# [6] Routes
@main.route('/')
def index():
    return render_template('index.html')


def search_cameras():
    global camera_search_results
    camera_search_results = list_cameras()


@main.route('/start_camera_search', methods=['POST'])
def start_camera_search():
    thread = threading.Thread(target=search_cameras)
    thread.start()
    return jsonify({"status": "searching"})


@main.route('/get_camera_results', methods=['GET'])
def get_camera_results():
    global camera_search_results
    if camera_search_results:
        return jsonify({"status": "completed", "cameras": camera_search_results})
    return jsonify({"status": "searching"})


def list_cameras():
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


        if not wrong_answer and last_llm_answer and last_select_object_coords:
            # only show the selected object
            # detected_objects = [
            #     obj for obj in detected_objects if obj["name"].lower() == last_llm_answer.lower()]

            label_text = last_llm_answer
            detected_objects = [{'name': label_text,
                                'coords': last_select_object_coords}]

        if current_pipeline == "YOLO and LLM":


            if yolo_size in ['n', 's', 'l']:
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

        elif current_pipeline == "VLM and YOLO World" and last_llm_answer and not wrong_answer:

            if yolo_world_size == 's':

                boxes, labels, label_texts, scores = yolo_world_detect(runner=yolo_world_model,
                                                                       object_description=last_llm_answer, input_image=frame)

                detected_objects = []
                for box, label, label_text, score in zip(boxes, labels, label_texts, scores):
                    x1, y1, x2, y2 = map(int, box)
                    detected_objects.append({
                        "name": label_text,
                        "coords": [x1, y1, x2, y2]
                    })

        # YOLO 감지 결과를 프레임에 표시
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj["name"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 프레임 반환
        _, encoded_frame = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')


@main.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route("/generate_answer", methods=["POST"])
def generate_answer():
    global camera_instance, last_llm_answer, last_llm_explanation, last_transcription, current_pipeline, last_select_object_coords, last_all_identified_objects  # buffer
    global yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer
    data = request.json
    clue = data.get("clue", "").strip()
    excluded_objects = data.get("rejectedObjects", [])

        
    print("Excluded Objects:", excluded_objects)

    if not clue:
        if last_transcription and last_transcription.strip():
            clue = last_transcription.strip()
        else:
            return jsonify({"error": "No clue provided. Either type text or record audio first."}), 400

    matched_position = None
    ret, frame = camera_instance.camera.read()

    # frame = cv2.resize(frame, (RES_WIDTH, RES_HEIGHT))

    if not ret:
        return jsonify({"error": "Failed to capture frame."}), 500

    if current_pipeline == "YOLO and LLM":
        detected_objects = []
        results = yolo_model(frame, verbose=False)[0]

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            obj_name = results.names[int(cls.item())]

            detected_objects.append({
                "name": obj_name,
                "coords": [x1, y1, x2, y2]
            })

        if len(detected_objects) == 0:
            return jsonify({"error": "No object detected."}), 400

        filtered_objects, unique_objects = process_detected_objects(
            excluded_objects, detected_objects)
        
        print("Unique Objects:", unique_objects)
        

        latency, selected_object, explanation = generate_llm_response(
            llm_model, llm_tokenizer, clue, unique_objects, device)

        all_identified_objects = unique_objects

        if selected_object != "unknown":

            for obj_data in filtered_objects:
                if isinstance(obj_data, dict) and obj_data["name"].lower() == selected_object.lower():
                    matched_position = obj_data["coords"]
                    break

    elif current_pipeline == "VLM and YOLO World":

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        latency, all_identified_objects, selected_object, explanation = generate_vlm_response(
            vlm_model, vlm_tokenizer, rgb_image, clue, excluded_objects, device)

        if selected_object != "unknown":

            boxes, labels, label_texts, scores = yolo_world_detect(
                runner=yolo_world_model, object_description=selected_object, input_image=rgb_image)

            if len(boxes) == 0:
                return jsonify({"error": "No object detected."}), 400

            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            matched_position = [x1, y1, x2, y2]

    else:
        return jsonify({"error": "No valid solution pipeline selected."}), 400

    last_llm_answer = selected_object
    last_llm_explanation = explanation
    last_all_identified_objects = all_identified_objects

    if not matched_position:
        return jsonify({
            "answer": "No matching object found.",
            "latency": latency,
            "positions": None
        })

    x1, y1, x2, y2 = matched_position
    last_select_object_coords = matched_position

    return jsonify({
        "answer": selected_object,
        "explanation": explanation,
        "all_identified_objects": all_identified_objects,
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
    data = request.json
    solution = data.get('solution')
    if not solution:
        return jsonify({"error": "No solution provided"}), 400

    # Set the global variable
    current_pipeline = solution

    if solution == "YOLO and LLM":
        # Load the YOLO model
        yolo_model = load_yolo_model(model_path=yolo_model_path, device="cpu")
        print("YOLO model loaded.")
        # Load the LLM model
        llm_model, llm_tokenizer = load_llm_model(device=device)
        print("LLM model loaded.")
    elif solution == "VLM and YOLO World":
        # Load the VLM model
        vlm_model, vlm_tokenizer = load_vlm_model(device=device)
        print("VLM model loaded.")
        # Load the YOLO World model
        yolo_world_model = load_yolo_world_model(
            model_path=yolo_world_model_path, config_path=yolo_world_model_config_path, device="cpu")
        print("YOLO World model loaded.")
    else:
        return jsonify({"error": "Invalid solution"}), 400

    return jsonify({"status": "solution set", "current_solution": current_pipeline})


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
    global stop_recording_flag, recording_thread, whisper_model, device, stt_model
    stop_recording_flag = True
    if recording_thread:
        recording_thread.join()
        recording_thread = None

    transcription = understand(
        stt_model, device, filename="recorded_audio.wav")

    return jsonify({"status": "recording stopped", "transcription": transcription})


@ main.route('/speak_llm_output', methods=['POST'])
def speak_llm_output():
    global last_llm_answer, last_llm_explanation, last_all_identified_objects, tts_model
    if not last_llm_answer:
        return jsonify({"error": "No LLM answer to speak."}), 400

    # Convert LLM answer to speech and play it
    full_response = f"Identified objects are {', '.join(last_all_identified_objects)}. The selected object is {last_llm_answer}. {last_llm_explanation}"
    speak(tts_model, full_response, current_os)

    return jsonify({"status": "Playing LLM speech"})


@ main.route('/wrong_answer_status', methods=['POST'])
def wrong_answer_status():
    global wrong_answer, moved_to_target_point, guesser, real_robot, use_robot, last_select_object_coords, last_llm_answer
    wrong_answer = request.json.get("wrongAnswer", False)

    print("Wrong answer flag set to:", wrong_answer)

    if wrong_answer and use_robot and moved_to_target_point:
        guesser.if_wrong()

        # return to the home position
        # guesser.move_to_home()

        moved_to_target_point = False

    if not wrong_answer and use_robot and moved_to_target_point:
        # move the robot to the home position
        real_robot._gripper_off()
        guesser.move_to_home()

        moved_to_target_point = False
    
    last_select_object_coords = None # reset the selected object coordinates
    last_llm_answer = None # reset the last LLM answer

    return jsonify({"status": "wrong answer flag set"})


@ main.route("/control_robot", methods=["POST"])
def control_robot():
    global last_select_object_coords, moved_to_target_point, guesser, use_robot

    if not last_select_object_coords:
        return jsonify({"error": "No object selected"}),

    if not use_robot:
        return jsonify({"error": "Robot control is disabled"}),

    # calibrate the coordinates from 2D to 3D and from camera frame to robot frame
    x1, y1, x2, y2 = last_select_object_coords

    # resize to the RES_WIDTH and RES_HEIGHT
    x1 = int(x1 * TARGET_RES_WIDTH / CAMERA_RES_WIDTH)
    y1 = int(y1 * TARGET_RES_HEIGHT / CAMERA_RES_HEIGHT)
    x2 = int(x2 * TARGET_RES_WIDTH / CAMERA_RES_WIDTH)
    y2 = int(y2 * TARGET_RES_HEIGHT / CAMERA_RES_HEIGHT)

    center_point_x = (x1 + x2) / 2
    center_point_y = (y1 + y2) / 2

    print("Center point:", center_point_x, center_point_y)

    target_point = t.cam2robot(center_point_x, center_point_y)

    # move the robot to the target point and pick / place the object
    # guesser.move_to_target()
    print("Target point:", target_point)
    guesser.pick_and_ask(target_point)

    # move the robot to the human contact point
    # guesser.move_to_target(HUMAN_CONTACT_POINT)

    moved_to_target_point = True

    return jsonify({"status": "success"})


@ main.route("/get_use_robot_status", methods=["GET"])
def get_use_robot_status():
    return jsonify({"use_robot": use_robot})
