# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import itertools
import time
import os
import itertools
import torchaudio
import soundfile as sf
import wave
#import torchinfo
#from torchsummary import summary

#import the model
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")



# functions

def write_to_file(file_path, gt, pred, score):
    data = {'true label': gt, 'predection': pred, 'score': score}
    df = pd.DataFrame(data)

    df.to_csv(file_path, sep='\t', index=False)

    print(f"Data has been successfully written to {file_path}.")

def save_audio_sequence(audio_sequence, sample_rate, file_path):
    # Open a WAV file in write mode
    with wave.open(file_path, 'w') as wav_file:
        # Set the audio file parameters
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(sample_rate)  # Sample rate

        # Write the audio data to the file
        wav_file.writeframes(audio_sequence.tobytes())

def crop_audio(audio_files, path1, path2):
    # path1 for where is the data now and path 2 for where the data will be restored
    
    # Set the desired duration in seconds
    desired_duration = 3 
    for i in audio_files:
        audio_name, start_point , end_point = i

        # Load the audio file
        audio_file = os.path.join(path1 , audio_name)
        audio, sample_rate = sf.read(audio_file)

        # Calculate the number of samples for the desired duration
        desired_samples = int(desired_duration * sample_rate)
        if (start_point != 'Unknown' and end_point != 'Unknown'):
            start_point = float(start_point)
            end_point = float(end_point)

            spoken_duration = end_point - start_point 
            mid_point = start_point + spoken_duration // 2

            # Calculate the start sample index for the middle portion
            start_index = int(max(0, (mid_point * sample_rate) - desired_samples // 2))
        else :
            start_index = max(0, len(audio) // 2 - desired_samples // 2)
        # Crop the audio to the desired duration
        cropped_audio = audio[start_index : start_index + desired_samples] # take exactly from the middle of the spoken part
        
        # Save the cropped audio to a new file
        output_file = os.path.join(path2 , audio_name)
        #save_audio_sequence(cropped_audio, sample_rate, output_file)
        sf.write(output_file, cropped_audio, sample_rate)



original_dataset = "/home/omerhatim/thesis/ok-aura-v1.0.0/clips/"
prefix = "/home/omerhatim/thesis/speaker-recognition-exploration/clips3/" # for the original dataset
savedfile = '/home/omerhatim/thesis/speaker-recognition-exploration/Verification1s.tsv'

df = pd.read_csv('/home/omerhatim/thesis/ok-aura-v1.0.0/dataset.tsv', header = 0, sep = '\t')
A = df['Speaker_ID'].value_counts()



unique_speaker_ids = A.index
string_to_number = {string: idx for idx, string in enumerate(unique_speaker_ids)}

# Map the 'Speaker_ID' column using the dictionary
df['Speaker_ID'] = df['Speaker_ID'].map(string_to_number)



B = df['Speaker_ID'].value_counts()



# Assuming you have a DataFrame called 'df' with a column 'path' containing the file paths
audio_files = df['Filename'].tolist()

# Add the prefix to each file path and keep only .wav files
#audio_files = [os.path.join(prefix, f) for f in audio_files if f.endswith(".wav")]

audio_with_length_files = []

start = df['Start_Time'].to_list()
end = df['End_Time'].to_list()
audio_with_length_files = list(zip(audio_files, start, end))

crop_audio(audio_with_length_files, original_dataset, prefix)




speaker_files = []

A = df['Speaker_ID'].to_list()
B = df['Filename'].to_list()
speaker_files = list(zip(A, B))



pairs = []
for speaker1, file1 in speaker_files:
    for speaker2, file2 in speaker_files:
        pairs.append([prefix+file1, prefix+file2, speaker1, speaker2])

true_labels = []
predictions = []
scores = []

for pair in pairs[:10]:
    file1, file2, speaker1, speaker2 = pair
    print(f"{speaker1} and {speaker2}")
    score, prediction = verification.verify_files(os.path.join(file1), os.path.join(file2))
    scores.append(score)

    # Label: 1 if same speaker, 0 if different speakers
    label = int(speaker1 == speaker2)
    true_labels.append(label)
    predictions.append(int(prediction))


write_to_file(savedfile, true_labels, predictions, scores)