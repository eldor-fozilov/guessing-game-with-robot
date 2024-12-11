import logging
import os
import platform
import struct
import wave
from pvrecorder import PvRecorder
from TTS.api import TTS
import whisper
import time

stop_recording_flag = False

def listen(max_duration=10, device_index=0, output_path='tmp.wav'):
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


def understand(model, device = "cpu", filename='tmp.wav'):
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio)
    if device == "cpu":
        result = whisper.decode(model, mel, whisper.DecodingOptions(fp16=False))
    else:
        result = whisper.decode(model, mel, whisper.DecodingOptions(fp16=True))

    return result.text

def speak(text, model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", output_path='tmp.wav'):
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.tts_to_file(text=text, file_path=output_path)
    os.system(f"aplay {output_path}")

def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
        if not blocking:
            cmd += " &"
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
        if blocking:
            cmd += "  --wait"
    elif platform.system() == "Windows":
        # TODO(rcadene): Make blocking option work for Windows
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    os.system(cmd)

def log_say(text, play_sounds, blocking=False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)