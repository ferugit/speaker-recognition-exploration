# imports
from TTS.api import TTS
from IPython.display import Audio
import azure.cognitiveservices.speech as speechsdk
import wave
import time
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import pyaudio
import pygame
import pandas as pd

from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
speech_key, service_region = "085eea5fc45d442781fec647bd0f0bba", "westeurope"




# functions

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(30)
        
    #time.sleep(0.5)
    

def extract_Name(sentence):
    word = "name is"
    index = sentence.find(word)
    text = ""
    for char in sentence[index+len(word)+1:]:
        if char in [" ", ",", "."]:
            break
        else:
            text += char
    return text


def check_speaker(tensor_value):
    if tensor_value > 0.2702677249908447:
        sentence = "checked you are the same speaker as claimed"
    else:
        sentence = "Alarm there is an imposter"

    return sentence
def verify_files(embeddings, embeddings_1):
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    output = cosine_similarity(embeddings, embeddings_1)
    return output[0]



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
    print("audio saved")



def Text_to_Speech(sentence, audio_file):
    model_name = "tts_models/en/ljspeech/glow-tts"
    tts = TTS(model_name)
    tts.tts_to_file(text= sentence,  file_path= audio_file)
    play_audio(audio_file)
    



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
    print(new_service)
    if new_service == "sign up." or new_service == "Sign up.":
        print(f"enrolled because I found message of{new_service}")
        enroll()
        return

    if  new_service == "verification." or new_service == "Verification.":
        verify()
        return

    if  new_service == "delete." or new_service == "Delete.":
        delete()
        return
        
    if  new_service == "who am I?" or new_service == "Who am I?":
        who_am_I()
        return

    else:
        sentence= "none of the service have been detected"
        path = 'nothing.wav'
        Text_to_Speech(sentence,path)
        return
    

def enroll():

    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()


    sent = "for enrollment please say your name"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)
    print(name)

    if name in A:
        sent = "the speaker is existing"
        pat = 'existing_speaker.wav'
        Text_to_Speech(sent, pat)

    else :
        # Record 
        sec=5
        pathtosave = "".join([name,".wav"])
        Record(sec, pathtosave)

        # get embeddings
        signal  =classifier.load_audio(pathtosave)
        embeddings = classifier.encode_batch(signal, normalize=True)
        df = df.append({'Speaker_Name': name, 'Embeddings': embeddings,'Filename': pathtosave}, ignore_index=True)
        df.to_csv('Speakers_file.tsv', sep='\t', index=False)

    

    

def verify():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Embeddings'].to_list()
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
        start_time= time.time()
        for speaker, embeddin in speaker_files:
            if speaker == name:
                embeddings_1 = embeddin
        # extract embeddings
        signal  =classifier.load_audio(pathtosave)
        embeddings = classifier.encode_batch(signal, normalize=True)

        embeddings_1 = embeddings_1[6:]
        tensor_list = eval(embeddings_1)
        embeddings_1 = torch.tensor(tensor_list)
        score = verify_files(embeddings,embeddings_1)

        # score, prediction = verification.verify_files('verify_check.wav', audio_file)
        sentence = check_speaker(score)
        end_time = time.time()
        inference_time = end_time - start_time
        verified_audio = 'verification.wav'
        print(f"the inference time is {inference_time}")
        Text_to_Speech(sentence, verified_audio)

    else :
        sent = "the speaker is not registered"
        pat = 'notRegistered_speaker.wav'
        Text_to_Speech(sent,pat)




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
        #df = df[~df['Speaker_Name'].str.contains(name, case=False)]
        df = df[df.Speaker_Name != name]
        df.to_csv('Speakers_file.tsv', sep='\t', index=False)
    else :
        sent = "the speaker is not registered"
        pat = 'notRegistered_speaker.wav'
        Text_to_Speech(sent,pat )
        
def who_am_I():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Embeddings'].to_list()
    speaker_files = list(zip(A, B))

    sent = "please say any sentence"
    pat = 'enroll_name.wav'
    Text_to_Speech(sent,pat)

    sec=3
    pathtosave = 'anonymous.wav'
    Record(sec, pathtosave)

    signal  =classifier.load_audio(pathtosave)
    embeddings = classifier.encode_batch(signal, normalize=True)

    for speaker1, embeddin in speaker_files:
        embeddings_1 = embeddin[6:]
        tensor_list = eval(embeddings_1)
        embeddings_1 = torch.tensor(tensor_list)
        score = verify_files(embeddings,embeddings_1)
        
        #score, prediction = verification.verify_files('anonymous.wav', file1)
        sentence = check_speaker(score)

        if sentence == "checked you are the same speaker as claimed":
            your_name = speaker1
            sent = " ".join(["you are ", your_name])
            pat = 'anonymous_checked.wav'
            Text_to_Speech(sent,pat)
            return
    
    sent = "you have not enrolled yet"
    pat = 'not_enrolled_yet.wav'
    Text_to_Speech(sent,pat)
    return


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