from flask import Flask, request, jsonify
import torch
from TTS.api import TTS
from app.utils.inference_models import generate_llm_response, generate_vlm_response
from app.utils.process_object import process_detected_objects
from app.utils.inference_models import generate_llm_response, generate_vlm_response, yolo_world_detect
from app.utils.load_models import load_yolo_model, load_llm_model, load_vlm_model, load_yolo_world_model, load_yolo_model
from app.utils.proces_audio import speak
import cv2
import whisper
import base64
import numpy as np
from io import BytesIO

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================
# TTS, STT Model Load
# Whisper (STT)
stt_model = whisper.load_model("base")
print("STT model loaded.")

# TTs (TTS)
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = TTS(tts_model_name)
if device == 'cpu': tts_model.to("cpu")
else: tts_model.to("cuda")
print("TTS model loaded.")
# =================================================================


# =================================================================
# variables for tracking
current_pipeline = None

# variables for models
llm_model = None
llm_tokenizer = None
vlm_model = None
vlm_tokenizer = None
yolo_model = None
yolo_world_model = None
# =================================================================


# =================================================================
# LLM & VLM
@app.route('/generate_answer_server', methods=['POST'])
def generate_answer():
    global yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer
    global current_pipeline

    # --------------------------------------------------
    # Capture the request data
    data = request.json
    
    frame_data       = data.get("frame", None)
    clue             = data.get("clue", "").strip()
    excluded_objects = data.get("excluded_objects", [])


    frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_data), np.uint8), cv2.IMREAD_COLOR)
    # --------------------------------------------------

    matched_position = None

    # [1] Pipeline: YOLO and LLM
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

        filtered_objects, unique_objects = process_detected_objects(
            excluded_objects, detected_objects)

        # print("Detected objects:", detected_objects)
        # print("Excluded objects:", excluded_objects)
        # print("Filtered objects:", filtered_objects)
        # print("Unique objects:", unique_objects)

        latency, selected_object, explanation = generate_llm_response(
            llm_model, llm_tokenizer, clue, unique_objects)

        if selected_object != "unknown":
            all_identified_objects = unique_objects

            for obj_data in filtered_objects:
                if isinstance(obj_data, dict) and obj_data["name"].lower() == selected_object.lower():
                    matched_position = obj_data["coords"]
                    break


    # [2] Pipeline: YOLO and LLM
    elif current_pipeline == "VLM and YOLO World":
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        latency, all_identified_objects, selected_object, explanation = generate_vlm_response(vlm_model, vlm_tokenizer, rgb_image, clue, excluded_objects)

        if selected_object != "unknown":
            boxes, labels, label_texts, scores = yolo_world_detect(
                runner=yolo_world_model, object_description=selected_object, input_image=rgb_image)

            x1, y1, x2, y2 = map(int, boxes[0])
            matched_position = [x1, y1, x2, y2]

    else:
        return jsonify({"error": "No valid solution pipeline selected."}), 400
    

    if not matched_position:
        return jsonify({
            "answer": "No matching object found.",
            "latency": latency,
            "positions": None
        })

    x1, y1, x2, y2 = matched_position

    return jsonify({
        "answer": selected_object,
        "explanation": explanation,
        "all_identified_objects": all_identified_objects,
        "latency": latency,
        "positions": "yes",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "last_llm_answer": selected_object,
        "last_llm_explanation": explanation,
        "last_all_identified_objects": all_identified_objects,
        "last_select_object_coords": matched_position
    })


@app.route('/select_solution_server', methods=['POST'])
def select_solution():
    global current_pipeline
    global yolo_model, yolo_world_model, llm_model, llm_tokenizer, vlm_model, vlm_tokenizer

    data = request.json
    current_pipeline = data.get('current_pipeline')

    if current_pipeline == "YOLO and LLM":
        # Load the YOLO model
        yolo_model = load_yolo_model(device=device)
        print("YOLO model loaded.")

        # Load the LLM model
        llm_model, llm_tokenizer = load_llm_model(device=device)
        print("LLM model loaded.")

    elif current_pipeline == "VLM and YOLO World":
        # Load the VLM model
        vlm_model, vlm_tokenizer = load_vlm_model(device=device)
        print("VLM model loaded.")

        # Load the YOLO World model
        yolo_world_model = load_yolo_world_model(device=device)
        print("YOLO World model loaded.")

    else:
        return jsonify({"error": "Invalid solution"}), 400

    return jsonify({"status": "solution set", "current_solution": current_pipeline})


# =================================================================


# =================================================================
# TTS & STT
@app.route('/stop_listen_server', methods=['POST'])
def stop_listen():
    data = request.json
    audio_base64 = data.get("audio", None)

    if not audio_base64:
        return jsonify({"error": "No audio data provided"}), 400


    audio_data = base64.b64decode(audio_base64)
    audio_path = "received_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_data)

    try:
        transcription = stt_model.transcribe(audio_path)["text"]
        return jsonify({"status": "recording stopped", "transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/speak_llm_output_server', methods=['POST'])
def speak_llm_output():
    data = request.json
    full_response = data.get("full_response")

    try:
        buffer = BytesIO()
        speak(tts_model, full_response, buffer)
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({"status": "success", "audio": audio_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =================================================================




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
