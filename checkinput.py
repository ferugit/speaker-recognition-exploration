import torch
import torchaudio
import requests
import matplotlib.pyplot as plt



audio_file = "/home/omerhatim/thesis/ok-aura-v1.0.0/clips/aura_wuw_NQdL9262V0mpD89b_0007.wav"

data_waveform, rate_of_sample = torchaudio.load(audio_file)



print("This is the shape of the audio file: {}".format(data_waveform.size()))
print("This is the summary of the audio file: {}".format(torchaudio.info(audio_file)))