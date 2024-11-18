from ultralytics import YOLO
from transformers import pipeline
from huggingface_hub import login
import torch
import time
import datetime
start = time.time()
login(token='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')
# Load a model
model = YOLO("yolo11x.pt")
# Perform object detection on an image
result = model("123.png")[0]
classes = result.names
obj_pred = result.boxes.cls
result.save(filename="result.jpg")  # save to disk
obj_list = ['A ' + classes[obj.item()] for obj in obj_pred]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "user", "content": f"There is some objects in image: {obj_list}. Please select one that matches the following question: Can you give me the sharp thing? You must not provide any other response or explanation. The object is: "},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(messages)
print(outputs[0]["generated_text"][-1])
end = time.time()
sec = (end - start)
result_list = str(datetime.timedelta(seconds=sec)).split(".")
print(result_list[0]) 