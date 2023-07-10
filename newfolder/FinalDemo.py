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
import threading
from leds import Pixels
import RPi.GPIO as GPIO
from speechbrain.pretrained import EncoderClassifier

# load the model and credentials for azure STT
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
speech_key, service_region = "085eea5fc45d442781fec647bd0f0bba", "westeurope"



# Global variable to store the current status
status = "off"
running = True

# button configuration
BUTTON_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
button_state = 0

# Threading event to synchronize the main and LED threads
event = threading.Event()

# Function for the LED thread to control the lights based on the status
def led_thread():
    global status, running
    while running:
        event.wait()  # Wait for the event to be set
        if status == "off":
            # Turn off the LED lights
            print("LED OFF")
            #pixels.status = "off"
            pixels.off()
        elif status == "wake up":
            # Set the LED lights for wake-up status
            print("LED Wake Up")
            #pixels.status = "wake up"
            pixels.wake_up()
        elif status == "speak":
            # Set the LED lights for speaking status
            print("LED Speak")
            #pixels.status = "speak"
            pixels.speak()
        elif status == "listen":
            # Set the LED lights for listening status
            print("LED Listen")
            #pixels.status = "listen"
            pixels.listen()
        elif status == "process":
            # Set the LED lights for listening status
            print("LED process")
            #pixels.status = "process"
            pixels.processing()
        event.clear()  # Reset the event for the next status change

# Create and start the LED thread
led_thread = threading.Thread(target=led_thread)
led_thread.start()

# Function to update the status and trigger the LED thread
def update_status(new_status):
    global status
    status = new_status
    pixels.update_status(status)
    event.set()  # Set the event to trigger the LED thread

# functions

# to play the audio file
def play_audio(file_path):
    time.sleep(1)
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    update_status("speak")
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(30)
    
    update_status("off")
    
#extract the name from the sentence
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

# chck the speaker from the threshold
def check_speaker(tensor_value):
    if tensor_value > 0.2702677249908447:
        sentence = "checked you are the same speaker as claimed"
    else:
        sentence = "Alarm there is an imposter"

    return sentence
    
# calculate the Cosine Similarity
def verify_files(embeddings, embeddings_1):
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    output = cosine_similarity(embeddings, embeddings_1)
    return output[0]


# start the recording
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
    update_status("listen")
    print("Start Recording ...")
    frames = []
    seconds = sec
    for i in range (0, int(sample_rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Recording Stopped...")
    update_status("process")
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


# text to speech function
def Text_to_Speech(sentence, audio_file):
    model_name = "tts_models/en/ljspeech/glow-tts"
    tts = TTS(model_name)
    tts.tts_to_file(text= sentence,  file_path= audio_file)

# speech to text
def Speech_to_Text():
    update_status("listen")
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
    
    update_status("off")
 
 
# determine the selected service
def perform_service(new_service):
    print(new_service)
    if new_service == "sign up." or new_service == "Sign up.":
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
    
        play_audio("nothing.wav")
        return
    
# enrollment
def enroll():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    play_audio("enroll_name.wav")
    update_status("process")

    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)
    print(name)

    if name in A:
        play_audio("existing_speaker.wav")
        update_status("process")

    else :
    
        play_audio("enrollment_recording.wav")
        # Record 
        sec=5
        pathtosave = "".join([name,".wav"])
        Record(sec, pathtosave)

        # get embeddings
        signal  =classifier.load_audio(pathtosave)
        embeddings = classifier.encode_batch(signal, normalize=True)
        df = df.append({'Speaker_Name': name, 'Embeddings': embeddings,'Filename': pathtosave}, ignore_index=True)
        df.to_csv('Speakers_file.tsv', sep='\t', index=False)

# verification
def verify():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Embeddings'].to_list()
    speaker_files = list(zip(A, B))

    play_audio("verify_name.wav")
    update_status("process")

    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)

    if name in A:
        play_audio("verify_sentence.wav")
        update_status("process")

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

        sentence = check_speaker(score)
        end_time = time.time()
        inference_time = end_time - start_time
        verified_audio = 'verification.wav'
        print(f"the inference time is {inference_time}")
        Text_to_Speech(sentence, verified_audio)
        play_audio('verification.wav')
        update_status("process")

    else :
        play_audio("notRegistered_speaker.wav")
        update_status("process")

# delete
def delete():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    
    play_audio("delete_name.wav")
    update_status("process")
    
    # entering name
    name = Speech_to_Text()
    name = extract_Name(name)

    if name in A:
        df = df[df.Speaker_Name != name]
        df.to_csv('Speakers_file.tsv', sep='\t', index=False)
    else :
        play_audio("notRegistered_speaker.wav")
        update_status("process")
        
def who_am_I():
    df = pd.read_csv('Speakers_file.tsv', header = 0, sep = '\t')
    A = df['Speaker_Name'].to_list()
    B = df['Embeddings'].to_list()
    speaker_files = list(zip(A, B))

    play_audio("WhoAmI_name.wav")
    update_status("process")

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
        sentence = check_speaker(score)

        if sentence == "checked you are the same speaker as claimed":
            your_name = speaker1
            sent = " ".join(["you are ", your_name])
            pat = "anonymous_checked.wav"
            Text_to_Speech(sent,pat)
            play_audio("anonymous_checked.wav")
            return
    
    play_audio("WhoAmI_notEnrolled.wav")
    update_status("process")
    return
    
# start the processing

# LEDs class
pixels = Pixels()

#Generate Audio Files
sent = "welcome, how can I help you you can either enroll, verify your self, or delete your audio"
pat = "welcome.wav"
Text_to_Speech(sent,pat)

sent = "Thank You"
pat = "thanks.wav"
Text_to_Speech(sent,pat)

sentence= "none of the service have been detected"
path = "nothing.wav"
Text_to_Speech(sentence,path)

sent = "for enrollment please say your name"
pat = "enroll_name.wav"
Text_to_Speech(sent,pat)

sent = "the speaker is exist"
pat = "existing_speaker.wav"
Text_to_Speech(sent, pat)

sent = "It is recording now, please say any sentence"
pat = "enrollment_recording.wav"
Text_to_Speech(sent, pat)

sent = "say your name in order to be deleted"
pat = "delete_name.wav"
Text_to_Speech(sent,pat)

sent = "the speaker is not registered"
pat = "notRegistered_speaker.wav"
Text_to_Speech(sent,pat )

sent = "To determine your identity, please say any sentence"
pat = "WhoAmI_name.wav"
Text_to_Speech(sent,pat)

sent = "you have not enrolled yet"
pat = "WhoAmI_notEnrolled.wav"
Text_to_Speech(sent,pat)

sent = "please say your name"
pat = "verify_name.wav"
Text_to_Speech(sent,pat)

sent = "please say any sentence"
pat = "verify_sentence.wav"
Text_to_Speech(sent,pat)


print ("you can press the button now")
try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            if button_state == 0:
                button_state = 1
                update_status("wake up")
                time.sleep(4)
                # welcoming
                play_audio("welcome.wav")


                # service selection
                new_service = Speech_to_Text()
                update_status("process")
                perform_service(new_service)

                
                play_audio("thanks.wav")
                update_status("off")
                pixels.clear_strips

        else:
            button_state = 0


except KeyboardInterrupt:
    # Clean up GPIO on keyboard interrupt
    GPIO.cleanup()


# Terminate the LED thread
running = False
event.set()  # Set the event to allow the LED thread to exit

led_thread.join()  # Wait for the LED thread to finish