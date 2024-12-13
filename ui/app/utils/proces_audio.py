import logging
import os
import platform
import struct
import wave
from pvrecorder import PvRecorder
from TTS.api import TTS
import whisper
import time


def understand(model, device="cpu", filename='output.wav'):
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio)
    if device == "cpu":
        result = whisper.decode(
            model, mel, whisper.DecodingOptions(fp16=False))
    else:
        result = whisper.decode(model, mel, whisper.DecodingOptions(fp16=True))

    return result.text


def speak(model, text, speaker="Viktor Menelaos", output_path='tmp.wav'):

    # generate speech by cloning a voice using default settings
    model.tts_to_file(text=text,
                      file_path=output_path,
                      speaker=speaker,
                      language="en",
                      split_sentences=False
                      )

    os.system(f"aplay {output_path}")
