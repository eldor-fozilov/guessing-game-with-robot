
import re
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import supervision as sv
import PIL
import numpy as np
import cv2
import torch

import time
import datetime
from transformers import pipeline
from flask import Blueprint, render_template, request, jsonify, Response
from app.utils.process_image import load_image
from torchvision.transforms.functional import to_tensor


def generate_llm_response(model, tokenizer, clue, detected_objects, device='cpu'):

    user_prompt = f"""You are an assistant whose task is to identify the correct object from this list {detected_objects} that aligns best with the following clue: '{clue}', which can be given in Korean or English.
    Take a moment to think about each object's properties, considering how the clue relates to them, directly or indirectly.
    Provide your response in the following format:

    Selected Object: <name of the correct object>
    Explanation: <brief explanation of how the object's properties align with the clue>

    The response should be structured exactly like this, with "Selected Object:" followed by the object's name, and "Explanation:" followed by a brief explanation.

    Response:"""

    pad_token_id = model.config.pad_token_id

    messages = [{"role": "user", "content": user_prompt}]
    print("LLM Input:", messages)

    start = time.time()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    outputs = pipe(
        messages,
        max_new_tokens=128,
        pad_token_id=pad_token_id,
        do_sample=False,
    )

    # Extract the output text
    output_text = outputs[0]["generated_text"][-1]['content']

    # output_text = "Selected Object: cup\nExplanation: The cup is the only object that can hold liquid."  # temporary

    print("LLM Response:", output_text)

    # Parse the output to get the selected object and explanation
    selected_object = None
    explanation = None

    for line in output_text.split("\n"):
        if line.startswith("Selected Object:"):
            selected_object = line.replace(
                "Selected Object:", "").strip().lower()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()

    if selected_object and selected_object in [obj.lower() for obj in detected_objects]:
        sel_obj_idx = [obj.lower()
                       for obj in detected_objects].index(selected_object)
    else:
        print("The exact object name is not returned. Applying heuristics to find the closest match.")
        # Find the closest match
        sel_obj_idx = 0
        max_score = 0
        for idx, obj in enumerate(detected_objects):
            score = sum(1 for word in selected_object.split()
                        if word in obj.lower())
            if score > max_score:
                max_score = score
                sel_obj_idx = idx

        if max_score == 0:
            print("Could not find a close match.")
            sel_obj_idx = -1

    if sel_obj_idx == -1:
        selected_object = 'unknown'
        explanation = "No explanation provided"
    else:
        selected_object = detected_objects[sel_obj_idx]

    end = time.time()
    latency = str(datetime.timedelta(seconds=(end - start)))

    latency = latency.split(".")[0]
    latency = latency + " seconds"

    if not explanation:
        explanation = "No explanation provided."

    print("Selected Object:", detected_objects[sel_obj_idx])
    print("Explanation:", explanation)

    return latency, selected_object, explanation


def generate_vlm_response(model, tokenizer, image, clue, excluded_objects, device='cpu'):
    user_prompt = f'''You are an assistant whose task is to analyze the given image and the following clue: '{clue}', which can be given in Korean or English.
    
    1. Identify and list all the objects present in the image.
    2. From the list of identified objects, select the one that best aligns with the clue, while ignoring these excluded objects if present: {excluded_objects}.
    3. Provide a brief explanation of why the selected object matches the clue.

    Format your response as follows:

    All Identified Objects: <comma-separated list of objects>
    Selected Object: <name of the correct object>
    Explanation: <brief explanation>

    Response:'''

    # Prepare the input question with the image token
    question = f'<image>\n{user_prompt}'
    print("VLM Input:", question)

    start = time.time()

    # Load and preprocess the image
    if device == 'mps':
        pixel_values = load_image(image, max_num=1).to(torch.float16).to(device)
    else:
        pixel_values = load_image(image, max_num=1).to(torch.bfloat16).to(device)

    generation_config = dict(max_new_tokens=128, do_sample=False)

    # Single inference call
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    # response = "All Identified Objects: cup, bottle, table\nSelected Object: cup\nExplanation: The cup is the only object that can hold liquid."  # temporary

    print("VLM Response:", response)

    # Parse the response to extract identified objects, selected object, and explanation
    all_identified_objects = []
    selected_object = None
    explanation = None

    for line in response.split("\n"):
        if line.startswith("All Identified Objects:"):
            all_identified_objects = [obj.strip().lower() for obj in line.replace(
                "All Identified Objects:", "").split(",")]
        elif line.startswith("Selected Object:"):
            selected_object = line.replace(
                "Selected Object:", "").strip().lower()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()

    # Post-process the selected object
    if selected_object:
        selected_object = selected_object.strip("[]{}()")

    # Fallback if selected object is not identified
    if not selected_object or selected_object in excluded_objects:
        print("The exact object name is not returned or is in the exclusion list. Applying heuristics to find the closest match.")
        selected_object = "unknown"

    if not explanation:
        explanation = "No explanation provided."

    end = time.time()
    latency = str(datetime.timedelta(seconds=(end - start)))

    latency = latency.split(".")[0]
    latency = latency + " seconds"

    print("All Identified Objects:", all_identified_objects)
    print("Selected Object:", selected_object)
    print("Explanation:", explanation)

    return latency, all_identified_objects, selected_object, explanation


def yolo_world_detect(
        runner,
        input_image,
        max_num_boxes=1,
        score_thr=0.05,
        nms_thr=0.5,
        object_description="",
):

    texts = [[t.strip()] for t in object_description.split(",")] + [[" "]]
    data_info = dict(img=input_image, img_id=0, texts=texts)
    data_info = runner.pipeline(data_info)

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    keep_idxs = nms(pred_instances.bboxes,
                    pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    # predictions
    pred_instances = pred_instances.cpu().numpy()

    boxes = pred_instances['bboxes']  # xyxy
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores
