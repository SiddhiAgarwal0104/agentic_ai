import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from speech_to_text import record_audio, transcribe_audio


def test_speech_to_text():
    print("🎤 Recording test audio...")

    audio_file = record_audio(duration=5)

    print("🔎 Transcribing audio...")

    text = transcribe_audio(audio_file)

    print("Transcribed text:", text)

    assert isinstance(text, str)
    assert len(text) > 0


if __name__ == "__main__":
    test_speech_to_text()