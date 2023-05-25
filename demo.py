# imports
from TTS.api import TTS
from IPython.display import Audio
import azure.cognitiveservices.speech as speechsdk
import wave
import numpy as np
import pyaudio
import pandas as pd

from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
speech_key, service_region = "085eea5fc45d442781fec647bd0f0bba", "westeurope"


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
    wn.play()

def Speech_to_Text():
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")
    result = speech_recognizer.recognize_once()

    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(cancellation_details.error_details))
 

def perform_service(new_service):
    if "enroll" in new_service:
        enroll()

    if "verification" in new_service:
        verify()

    if "delete" in new_service:
        delete()

    else:
        sentence= "none of the service have been detected"
        path = 'nothing.wav'
        Text_to_Speech(sentence,path)
    

def enroll():

    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()


    sent = "for enrollment please say your name"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)

    if name in A:
        Text_to_Speech(sent = "the speaker is existing", pat = 'existing_speaker.wav')

    else :
        sec=5
        pathtosave = "".join([name,".wav"])
        Record(sec, pathtosave)

        df = df.append({'Speaker_Name': name, 'Audio_File': pathtosave}, ignore_index=True)
        df.to_csv('Speakers.tsv', sep='\t', index=False)

    

    

def verify():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Filename'].to_list()
    speaker_files = list(zip(A, B))

    sent = "please say your name"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)

    if name in A:
        sent = "please say any sentence"
        pat = 'enroll_name.wav'
        Text_to_Speech(sent,pat)

        sec=3
        pathtosave = 'verify_check.wav'
        Record(sec, pathtosave)
        for speaker, file in speaker_files:
            if speaker == name:
                audio_file = file

        score, prediction = verification.verify_files(pat, audio_file)
        sentence = check_speaker(prediction)
        verified_audio = 'verification.wav'
        Text_to_Speech(sentence, verified_audio)

    else :
        Text_to_Speech(sent = "the speaker is not registered", pat = 'notRegistered_speaker.wav')




def delete():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    

    sent = "say your name in order to be deleted"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    name = Speech_to_Text()
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
new_service = Speech_to_Text()

perform_service(new_service)

sent = "Thank You"
pat = "thanks.wav"
Text_to_Speech(sent,pat)