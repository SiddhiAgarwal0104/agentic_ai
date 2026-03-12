from gtts import gTTS
import os

def speak_text(text):
    tts = gTTS(text=text, lang="en")

    file = "response.mp3"
    tts.save(file)

    os.system(f"start {file}")

    # testing
speak_text("kitne din ho gye aapko scheme ke baare mein pata chalne mein?")