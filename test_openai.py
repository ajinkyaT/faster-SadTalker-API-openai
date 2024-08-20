import os
from pathlib import Path
from openai import OpenAI

# Load your OpenAI API key from an environment variable
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def text_to_speech(text, voice='alloy', model='tts-1'):
    # Define the path to save the speech file
    speech_file_path = Path(__file__).parent / "speech.wav"
    
    # Make a request to the OpenAI TTS API using the with method
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav"
    ) as response:
        response.stream_to_file(speech_file_path)
    print(f"Audio file saved as {speech_file_path}")

# Example usage
text_to_speech("Hello, this is a test of the OpenAI TTS API.")