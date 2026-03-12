import whisper
import sounddevice as sd
from scipy.io.wavfile import write

model = whisper.load_model("small")

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Audio saved:", filename)

    return filename


def transcribe_audio(audio_path):
    result = model.transcribe(
        audio_path,
        language="en", 
        fp16=False
    )

    text = result["text"].strip()

    return text