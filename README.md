## Guessing Game with Robot Arm


#### Environment Installation Guideline

conda create -n <name> python=3.10

[OS Based]
- pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

[Common]
- pip install openmim
- mim install "mmcv==2.1.0"
- mim install "mmdet==3.3.0"
- mim install "mmcv-lite==2.0.1"
- mim install "mmengine==0.10.4"
- mim install "mmyolo==0.6.0"
- cd "YOLO-WORLD directory" --> pip install -e .
- cd "lerobot directory" --> pip install -e ".[dynamixel]"
- cd "TTS directory" --> pip install -e .
- pip install spacy timm accelerate sentencepiece
- pip install Flask-Assets
- pip install ultralytics openai-whisper pvrecorder
- pip install mujoco ikpy