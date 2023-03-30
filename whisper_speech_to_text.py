import openai
import sounddevice as sd
import numpy as np
import io
from pydub import AudioSegment
import pyaudio
from io import BytesIO
import keyboard

# Define the ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


def record_audio(sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        output=False,
        input_device_index=None)  # Use the default input device

    

    audio_data = b'' # Initialize empty byte string for audio data
    key_pressed = False # Start with key pressed state as False

    while True:
        # Check if the key is pressed
        if keyboard.is_pressed('ctrl'):
            key_pressed = True
            print("{0}Recording...{1}".format(RED, RESET))
        elif key_pressed:
            key_pressed = False
            print("{0}Stopped Recording{1}".format(YELLOW, RESET))
            # Yield the recorded audio data and reset the audio data buffer
            yield np.frombuffer(audio_data, dtype=np.int16)
            audio_data = b''

        # If the key is pressed, record audio
        if key_pressed:
            data = stream.read(sample_rate) # Read a single sample of audio data
            audio_data += data # Append the new data to the existing audio data



class NamedBytesIO(BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop("name", "audio_file.mp3")
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name


def convert_audio_to_file_like_object(audio_data, sample_rate):
    # Convert the NumPy array to a PyDub AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )

    # Create a NamedBytesIO object and export the audio data as an MP3
    file_like_object = NamedBytesIO()
    audio_segment.export(file_like_object, format="mp3")
    file_like_object.seek(0)
    return file_like_object



def transcribe_audio(api_key, audio_file):
    openai.api_key = api_key
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def transcribe_and_chat(api_key, recorded_audio, sample_rate):
    # Save recorded audio as a file-like object
    file_like_audio = convert_audio_to_file_like_object(recorded_audio, sample_rate)

    # Transcribe the audio using the Whisper API
    transcript = transcribe_audio(api_key, file_like_audio)
    print("{0}Transcript: {1}{2}".format(BLUE, transcript, RESET))

    # Send transcript over to ChatGPT
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{transcript}"}
        ]
    )
    # Add bots respond to the conversation
    response = completions["choices"][0]["message"]["content"]
    print("{0}Response: {1}{2}".format(GREEN, response, RESET))

    return response

def main():
    sample_rate = 16000
    api_key = "your_api_key"

    # Initialize the recording generator
    recording_generator = record_audio(sample_rate)

    while True:
        if keyboard.is_pressed('ctrl'):
            # If the key is pressed, record and transcribe audio
            recorded_audio = next(recording_generator)
            transcribe_and_chat(api_key, recorded_audio, sample_rate)

if __name__ == "__main__":
    main()