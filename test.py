import pyaudio
import wave
import deepspeech
import numpy as np


def Speech_to_Text(AudioFile):
    model_path = '/home/omerhatim/thesis/speaker-recognition-exploration/pretrained_models/deepspeech-0.9.3-models.pbmm'
    ds = deepspeech.Model(model_path)
    audio_path = AudioFile
    audio = wave.open(audio_path, 'rb')
    audio_data = audio.readframes(audio.getnframes())
    sample_rate = audio.getframerate()
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    text = ds.stt(audio_array)

    return text

chunk = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
sample_rate = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk)

print("Start Recording ...")

frames = []
seconds = 5
for i in range (0, int(sample_rate / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)
print("Recording Stopped...")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav",'wb')
wf.setnchannels(CHANNELS)
wf.setframerate(sample_rate)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.writeframes(b''.join(frames))
wf.close

pat = "output.wav"

sent = Speech_to_Text(pat)

print(sent)