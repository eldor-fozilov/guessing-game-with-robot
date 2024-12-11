import time
import datetime
from transformers import pipeline
from flask import Blueprint, render_template, request, jsonify, Response

# LLM 모델 초기화
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

def generate_llm_response(clue, unique_objects, filtered_objects):
    """LLM으로 응답 생성"""
    user_prompt = f"""You are an assistant whose task is to identify the correct object from this list {unique_objects} that aligns best with the following clue: '{clue}'.
    Take a moment to think about each object's properties, considering how the clue relates to them, directly or indirectly.
    Then, provide only the name of the correct object as your answer. No additional explanations.
    The answer is: """

    messages = [{"role": "user", "content": user_prompt}]
    print("LLM Input:", messages)

    start = time.time()
    outputs = pipe(
        messages,
        max_new_tokens=50,
        pad_token_id=pad_token_id,
        do_sample=False,
    )
    end = time.time()

    latency = str(datetime.timedelta(seconds=(end - start)))

    # LLM 응답 처리
    print("LLM Outputs:", outputs)
    if isinstance(outputs, list) and "generated_text" in outputs[0]:
        full_result = outputs[0]["generated_text"]
    else:
        return "Unexpected output format.", latency, None

    print("LLM Full Result:", full_result)


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
    for obj_data in filtered_objects:
        if isinstance(obj_data, dict) and obj_data["name"].lower() == assistant_content.lower():
            matched_position = obj_data["coords"]
            break

    if not matched_position:
        return assistant_content, latency, None

    print(f"Matched Object: {assistant_content}, Coordinates: {matched_position}")
    return assistant_content, latency, matched_position

