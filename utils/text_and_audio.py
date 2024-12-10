import logging
import os
import platform

import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import scipy.io.wavfile as wavfile

import struct
import wave
from pvrecorder import PvRecorder
import webrtcvad
import collections
import time

def listen(output_path='tmp.wav'):
    device_index = 0

    recorder = PvRecorder(frame_length=1024, device_index=device_index)
    recorder.start()

    wavfile = None

    if output_path is not None:
        wavfile = wave.open(output_path, "w")
        # noinspection PyTypeChecker
        wavfile.setparams((1, 2, recorder.sample_rate, recorder.frame_length, "NONE", "NONE"))

    st = time.time()
    print("=======Start Listening")
            
    while True:
        frame = recorder.read()
        if wavfile is not None:
            wavfile.writeframes(struct.pack("h" * len(frame), *frame))
        if time.time()-st > 10:
            print("=======Stopping Listening")
            break

    recorder.delete()
    if wavfile is not None:
        wavfile.close()


# def listen(output_path='output.wav', sample_rate=16000, frame_duration=30, silence_duration=1.5, device_index=1, max_duration=5):
#     """
#     Records audio from the specified device and stops after detecting a period of silence or reaching the max duration.

#     Parameters:
#     - output_path (str): The name of the output WAV file.
#     - sample_rate (int): The sample rate in Hz.
#     - frame_duration (int): The frame duration in milliseconds.
#     - silence_duration (float): The duration of silence to trigger stopping (in seconds).
#     - device_index (int): The index of the input audio device.
#     - max_duration (int): The maximum recording duration in seconds.
#     """
#     vad = webrtcvad.Vad()
#     vad.set_mode(1)  # 0: Most aggressive filtering, 3: Least aggressive

#     frame_length = int(sample_rate * frame_duration / 1000)  # Number of samples per frame
#     num_silent_frames = int(silence_duration * 1000 / frame_duration)

#     ring_buffer = collections.deque(maxlen=num_silent_frames)
#     triggered = False
#     frames = []

#     print("======= Start Listening")

#     start_time = time.time()

#     try:
#         with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', device=device_index) as stream:
#             while True:
#                 data, overflowed = stream.read(frame_length)
#                 frame = data[:, 0]  # Extract the mono channel

#                 # Convert the frame to bytes for VAD processing
#                 frame_bytes = struct.pack("h" * len(frame), *frame)
#                 is_speech = vad.is_speech(frame_bytes, sample_rate)

#                 if not triggered:
#                     ring_buffer.append((frame, is_speech))
#                     num_voiced = len([f for f, speech in ring_buffer if speech])
#                     if num_voiced > 0.9 * ring_buffer.maxlen:
#                         triggered = True
#                         frames.extend([f for f, _ in ring_buffer])
#                         ring_buffer.clear()
#                 else:
#                     frames.append(frame)
#                     ring_buffer.append((frame, is_speech))
#                     num_unvoiced = len([f for f, speech in ring_buffer if not speech])
#                     if num_unvoiced > ring_buffer.maxlen:
#                         print("======= Stopping Listening (silence detected)")
#                         break

#                 # Stop recording after max_duration
#                 if time.time() - start_time > max_duration:
#                     print("======= Stopping Listening (time limit reached)")
#                     break

#         # Save recorded frames to WAV file
#         if frames:
#             with wave.open(output_path, "w") as wavfile_out:
#                 wavfile_out.setparams((1, 2, sample_rate, 0, "NONE", "NONE"))
#                 for frame in frames:
#                     wavfile_out.writeframes(struct.pack("h" * len(frame), *frame))
#             print(f"Recording complete. Saved to {output_path}")
#         else:
#             print("No audio detected. No file was saved.")

#     except Exception as e:
#         print(f"An error occurred: {e}")

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