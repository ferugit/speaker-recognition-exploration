import pyaudio
import wave

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
for i in renge (0, int(sample_rate / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)
print("Recording Stopped...")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav",'wb')
wf.setnchannels(CHANNELS)
wf.setframerate(sample_rate)
wf.writeframes(b''.join(frames))
wf.close