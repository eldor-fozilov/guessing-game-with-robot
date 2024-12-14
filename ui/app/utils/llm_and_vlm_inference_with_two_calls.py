
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


def generate_vlm_response(model, tokenizer, image, clue, excluded_objects, device='cpu'):

    user_prompt = f'''You are an assistant whose task is to identify the correct object from the given image that aligns best with the following clue: '{clue}'
    First think about each object's properties in the image, considering how the clue relates to them, directly or indirectly. Then, provide only the name of the correct object as your answer.
    Avoid considering the objects in this list {excluded_objects}, if there are any.
    No additional explanations.
    The answer is: '''

    start = time.time()

    pixel_values = load_image(
        image, max_num=1).to(torch.bfloat16).to(device)
    generation_config = dict(max_new_tokens=64, do_sample=False)

    question = f'<image>\n{user_prompt}'

    print("VLM Input (for object selection):", question)

    selected_object = model.chat(
        tokenizer, pixel_values, question, generation_config)

    # selected_object = "Cup" # temporary

    print("VLM Response (for object selection):", selected_object)

    selected_object = selected_object.lower().strip()

    if selected_object[0] in ["[", "(", "{"]:
        selected_object = selected_object[1:]
    if selected_object[-1] in ["]", ")", "}"]:
        selected_object = selected_object[:-1]

    # run one more inference now to ask model to explain why it chose this object

    user_prompt = f'''You are an assistant whose task is to explain why the object '{selected_object}' is the best match for the clue '{clue}' among the objects in the image.
    First, consisely list out all the objects in the image, and then provide a very brief explanation (one or two sentences) of how the '{selected_object}' object's properties align with the clue, directly or indirectly.
    Response: '''

    question = f'<image>\n{user_prompt}'

    print("VLM Input (for explanation):", question)

    explanation = model.chat(tokenizer, pixel_values,
                             question, generation_config)

   #  explanation = "The object is a cup." # temporary

    print("VLM Response (for explanation):", explanation)

    end = time.time()

    latency = str(datetime.timedelta(seconds=(end - start)))

    return latency, selected_object, explanation


def generate_llm_response(model, tokenizer, clue, detected_objects, device='cpu'):

    user_prompt = f"""You are an assistant whose task is to identify the correct object from this list {detected_objects} that aligns best with the following clue: '{clue}', which can be given in Korean or English.
        Take a moment to think about each object's properties, considering how the clue relates to them, directly or indirectly.
        Provide only the name of the correct object as your answer. No additional explanations.
        The answer is: """

    pad_token_id = model.config.pad_token_id

    messages = [{"role": "user", "content": user_prompt}]
    print("LLM Input (for object selection):", messages)

    start = time.time()

    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer)

    outputs = pipe(
        messages,
        max_new_tokens=64,
        pad_token_id=pad_token_id,
        do_sample=False,
    )

    selected_object = outputs[0]["generated_text"][-1]['content']

#     selected_object = "Cup"

    print("LLM Response (for object selection):", selected_object)

    selected_object = selected_object.lower().strip()

    if selected_object in detected_objects:
        sel_obj_idx = detected_objects.index(selected_object)
    else:
        print("The exact object name is not returned. So we will apply heuristics to find the closest match.")
        # find the closest match
        sel_obj_idx = 0
        max_score = 0
        for idx, obj in enumerate(detected_objects):
            score = 0
            for word in selected_object.split():
                word = word.lower()
                if word in obj:
                    score += 1
            if score > max_score:
                max_score = score
                sel_obj_idx = idx

        print("Selected Object: ", detected_objects[sel_obj_idx])

    # run one more inference now to ask model to explain why it chose this object
    user_prompt = f"""You are an assistant whose task is to explain why the object '{detected_objects[sel_obj_idx]}' is the best match for the clue '{clue}' among the objects in the list {detected_objects}.
        Provide a very brief explanation (one or two sentences) of how the object's properties align with the clue, directly or indirectly.
        Response: """

    messages = [{"role": "user", "content": user_prompt}]
    print("LLM Input (for explanation):", messages)

    outputs = pipe(
        messages,
        max_new_tokens=64,
        pad_token_id=pad_token_id,
        do_sample=False,
    )

    end = time.time()

    latency = str(datetime.timedelta(seconds=(end - start)))

    explanation = outputs[0]["generated_text"][-1]['content']

#     explanation = "The object is a cup." # temporary

    print("LLM Response (for explanation):", explanation)

    return latency, detected_objects[sel_obj_idx], explanation
