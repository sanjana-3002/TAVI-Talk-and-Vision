import os
import uuid
import threading
import time
import json
import requests
import wave
import cv2  # OpenCV for video recording

import numpy as np
from kivy.clock import Clock
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.video import Video   # To display video in the UI
from kivy.core.audio import SoundLoader

import pyaudio
import pvporcupine

from config import BACKEND_URL, PORCUPINE_KEY

# KV string for a simple chat UI layout
KV = '''
<ChatScreen>:
    orientation: 'vertical'
    ScrollView:
        id: scroll_view
        do_scroll_x: True
        do_scroll_y: True
        BoxLayout:
            id: chat_box
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
    Label:
        id: mic_indicator
        text: ''
        size_hint_y: None
        height: '40dp'
'''

Builder.load_string(KV)

class ChatScreen(BoxLayout):
    pass

class MyKivyApp(App):
    def build(self):
        self.chat_screen = ChatScreen()
        threading.Thread(target=self.wake_word_listener, daemon=True).start()
        return self.chat_screen

    def add_message(self, message, sender="system"):
        lbl = Label(text=f"[{sender}] {message}", markup=True, size_hint_y=None, height='30dp')
        self.chat_screen.ids.chat_box.add_widget(lbl)
        Clock.schedule_once(lambda dt: setattr(self.chat_screen.ids.scroll_view, 'scroll_y', 0))

    def wake_word_listener(self):
        """
        Listens for the wake word "Jarvis" using Porcupine and starts recording audio when detected.   
        """
        try:
            porcupine = pvporcupine.create(access_key=PORCUPINE_KEY, keywords=["jarvis"])
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            
            Clock.schedule_once(lambda dt: self.add_message("Hi there! Just say 'Jarvis' to get started.", sender="app"))

            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                result = porcupine.process(pcm)
                if result >= 0:
                    Clock.schedule_once(lambda dt: self.add_message("Hello! This is Jarvis. How can I make your day easier", sender="Jarvis"))
                    self.record_audio()
                time.sleep(0.01)
        except Exception as e:
            Clock.schedule_once(lambda dt, err=e: self.add_message(f"Error in calling up 'Jarvis': {err}", sender="error"))

    def record_audio(self):
        """
        Records audio from the microphone (simulated for 5 seconds) and sends it to the backend.
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        Clock.schedule_once(lambda dt: self.add_message("You've got my attention. I'm listening for the next 5 seconds.", sender="Jarvis"))
        record_seconds = 5
        for i in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        filename = os.path.join(temp_dir, f"{uuid.uuid4()}_input.wav")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        #Clock.schedule_once(lambda dt: self.add_message("Recording complete. Sending audio to backend...", sender="app"))
        #Clock.schedule_once(lambda dt: self.add_message("Hang tight, I'm processing that for you!", sender="app"))
        self.send_audio_to_backend(filename)
    
    def send_audio_to_backend(self, audio_filepath):
        """
        Posts the recorded audio file to the /process_audio/ endpoint.
        """
        try:
            files = {'file': open(audio_filepath, 'rb')}
            url = f"{BACKEND_URL}/process_audio/"
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                self.process_audio_response(data)
            else:
                Clock.schedule_once(lambda dt, err=response.status_code: self.add_message(f"Backend error: {err}", sender="error"))
        except Exception as e:
            Clock.schedule_once(lambda dt, err=e: self.add_message(f"Error sending audio: {err}", sender="error"))
    
    def process_audio_response(self, data):
        """
        Processes the JSON response from the audio API.
        """
        data1 = data.get("data1", {})
        data2 = data.get("data2", "")
        data3 = data.get("data3", "")  # Relative URL (e.g., "/download_audio/filename.mp3")
        
        audio_url = f"{BACKEND_URL}{data3}"
        transcript = data.get("transcript", "")
        
        Clock.schedule_once(lambda dt: self.add_message(f": {transcript}", sender="user"))
        Clock.schedule_once(lambda dt: self.add_message("Hang tight, I'm processing that for you!", sender="Jarvis"))
        #Clock.schedule_once(lambda dt: self.add_message(f"Response: {data2}", sender="assistant"))
        
        if data1.get("Record"):
            #Clock.schedule_once(lambda dt: self.add_message("Record intent detected. Launching video capture...", sender="app"))
            Clock.schedule_once(lambda dt: self.add_message("Got it! You’d like to start recording—camera’s coming on. ", sender="Jarvis"))
            self.capture_video()  # Launch video capture
        else:
            #Clock.schedule_once(lambda dt: self.add_message("Playing response audio...", sender="app"))
            Clock.schedule_once(lambda dt: self.add_message("Umm... here's what I know!", sender="Jarvis"))
            self.play_audio(audio_url)
            Clock.schedule_once(lambda dt: self.add_message(f": {data2}", sender="Jarvis"))
            #self.play_audio(audio_url)
        
        #Clock.schedule_once(lambda dt: self.add_message("Re-listening for wake word...", sender="app"))
        Clock.schedule_once(lambda dt: self.add_message("Just say 'Jarvis' if you need my help again!", sender="app"))
    
    def play_audio(self, audio_url):
        """
        Loads and plays the audio using Kivy's SoundLoader.
        """
        sound = SoundLoader.load(audio_url)
        if sound:
            sound.play()
        else:
            Clock.schedule_once(lambda dt: self.add_message("Failed to load audio.", sender="error"))
    
    def capture_video(self):
        """
        Captures video for 5 seconds using the webcam (via OpenCV), sends it to the backend,
        and updates the chat UI with the video summary and plays the TTS audio.
        The recorded video is displayed in the UI using a Kivy Video widget.
        """
        # Start video capture using OpenCV
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                Clock.schedule_once(lambda dt: self.add_message("Error: Unable to access the camera.", sender="error"))
                return
            
            # Define video codec and output file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_dir = "temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            video_filename = os.path.join(temp_dir, f"{uuid.uuid4()}_video.mp4")
            fps = 20.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
            
            Clock.schedule_once(lambda dt: self.add_message("Recording video for 5 seconds...", sender="Jarvis"))
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break
            cap.release()
            out.release()
            
            #Clock.schedule_once(lambda dt: self.add_message("Video recorded. Sending video to backend...", sender="app"))
            Clock.schedule_once(lambda dt: self.add_message("Just a moment... I’m processing what’s around you.", sender="Jarvis"))
            
            # Send the video file to process_video API
            files = {'file': open(video_filename, 'rb')}
            url = f"{BACKEND_URL}/process_video/"
            response = requests.post(url, files=files)
            if response.status_code == 200:
                video_data = response.json()
                text_summary = video_data.get("text_summary", "")
                video_audio_relative = video_data.get("audio_file", "")
                full_audio_url = f"{BACKEND_URL}{video_audio_relative}"
                # Display video summary in UI
                #Clock.schedule_once(lambda dt: self.add_message(f"Video summary: {text_summary}", sender="assistant"))
                Clock.schedule_once(lambda dt: self.add_message(f"Based on what I see, here's my take on what's around you: {text_summary}", sender="Jarvis"))
                # Play the video TTS audio
                self.play_audio(full_audio_url)
                # Display the recorded video in the chat UI using a Video widget
                Clock.schedule_once(lambda dt: self.add_video(video_filename))
            else:
                Clock.schedule_once(lambda dt, err=response.status_code: self.add_message(f"Error from video API: {err}", sender="error"))
        except Exception as e:
            Clock.schedule_once(lambda dt, err=e: self.add_message(f"Error during video capture: {err}", sender="error"))
    
    def add_video(self, video_filepath):
        """
        Adds a Video widget to the chat UI to display the recorded video.
        The video will remain in the UI until the user closes the app.
        """
        from kivy.uix.video import Video
        video_widget = Video(source=video_filepath, state='play', options={'eos': 'loop'}, size_hint_y=None, height='200dp')
        self.chat_screen.ids.chat_box.add_widget(video_widget)
        Clock.schedule_once(lambda dt: self.add_message("Saving your video in our chat", sender="app"))
    
if __name__ == "__main__":
    MyKivyApp().run()
