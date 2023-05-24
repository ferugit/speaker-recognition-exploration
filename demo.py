# imports
from TTS.api import TTS
from IPython.display import Audio
import deepspeech
import numpy as np
import wave
import pyaudio
import pandas as pd
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# functions

def extract_Name(sentence):
    word = "name is"
    words = sentence.split()
    if word in words:
        index = words.index(word)
        extracted = ' '.join(words[index+1:])
        return extracted
    else:
        return ""

def check_speaker(tensor_value):
    if tensor_value > 0.3:
        sentence = "checked you are the same speaker as claimed"
    else:
        sentence = "Alarm there is an imposter"

    return sentence

def Record(sec,path):
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
    seconds = sec
    for i in range (0, int(sample_rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Recording Stopped...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(path,'wb')
    wf.setnchannels(CHANNELS)
    wf.setframerate(sample_rate)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.writeframes(b''.join(frames))
    wf.close



def Text_to_Speech(sentence, audio_file):
    model_name = "tts_models/en/ljspeech/glow-tts"
    tts = TTS(model_name)
    tts.tts_to_file(text= sentence,  file_path= audio_file)
    wn = Audio('output.wav', autoplay=True)

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

def perform_service(input):
    if "enroll" in new_service:
        enroll()

    if "verification" in  new_service:
        verify()

    if "delete" in  new_service:
        delete()

    else:
        sentence= "none of the service have been detected"
        path = 'nothing.wav'
        Text_to_Speech(sentence,path)
    

def enroll():

    df = pd.read_csv('Speakers.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()


    sent = "for enrollment please say your name"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    sec = 5
    pat = "name.wav"
    Record(sec,pat)
    name = Speech_to_Text(pat)
    name = extract_Name(name)

    if name in A:
        Text_to_Speech(sent = "the speaker is existing", pat = 'existing_speaker.wav')

    else :
        sec=5
        pathtosave = "".join([name,".wav"])
        Record(sec, pathtosave)

        df = df.append({'Speaker_Name': 10, 'Audio_File': 1}, ignore_index=True)
        df.to_csv('Speakers.tsv', sep='\t', index=False)

    

    

def verify():
    df = pd.read_csv('Speakers.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Filename'].to_list()
    speaker_files = list(zip(A, B))

    sent = "say your name for verification"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    sec = 5
    pat = "name.wav"
    Record(sec,pat)
    name = Speech_to_Text(pat)
    name = extract_Name(name)

    if name in A:
        for speaker, file in speaker_files:
            if speaker == name:
                audio_file = file

        score, prediction = verification.verify_files(pat, audio_file)
        sentence = check_speaker(prediction)
        verified_audio = 'verification'
        Text_to_Speech(sentence, verified_audio)

    else :
        Text_to_Speech(sent = "the speaker is not registered", pat = 'notRegistered_speaker.wav')




def delete():
    df = pd.read_csv('Speakers.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    

    sent = "say your name in order to be deleted"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    sec = 5
    pat = "name.wav"
    Record(sec,pat)
    name = Speech_to_Text(pat)
    name = extract_Name(name)

    if name in A:
        df = df[~df['Speaker_Name'].str.contains(name, case=False)]
    else :
        Text_to_Speech(sent = "the speaker is not registered", pat = 'notRegistered_speaker.wav')

# welcoming
sent = "welcome, how can I help you you can either enroll, verify your self, or delete your audio"
pat = "welcome.wav"
Text_to_Speech(sent,pat)

# service selection
sec = 5
pat = "service.wav"
Record(sec,pat)

new_service = Speech_to_Text(pat)

perform_service(new_service)


sent = "Thank You"
pat = "thanks.wav"
Text_to_Speech(sent,pat)
# 







# #sent = "this is my name and I love you"
# pat = "output.wav"

# #Text_to_Speech(sent,pat)

# sent = Speech_to_Text(pat)

# print(sent)
