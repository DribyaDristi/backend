import pyttsx3

tts = pyttsx3.init(driverName='espeak')

output = "Testing fine synthesis hello"

def speak(text):
    tts.say(text)
    tts.runAndWait()

speak(output)

# import pyttsx3

# engine = pyttsx3.init(driverName='espeak')
# engine.say("Testing speech synthesis fine.")
# engine.runAndWait()
