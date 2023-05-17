# import libraries and model
import torch
import time
import torchaudio
#import torchinfo
import numpy as np
#from torchsummary import summary
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")


# perform verification
score, prediction = verification.verify_files('/home/omerhatim/thesis/speechbrain dataset/speechbrain/tests/samples/ASR/spk1_snt1.wav','/home/omerhatim/thesis/speechbrain dataset/speechbrain/tests/samples/ASR/spk2_snt1.wav') # Different Speakers
print (score,prediction)


# inference time for verification
inferences = [] # 11 values starting from 0 to 10
for j in range(11):
    inference_time=[]
    if j == 0:
        samplingrate= 8000
    else:
        samplingrate = j * 16000 
    for i in range(1000):
        sample1 = torch.randn(1,samplingrate)
        sample2 = torch.randn(1,samplingrate)
        start_time = time.time()
        verification.verify_batch(sample1,sample2)
        end_time = time.time()
        inftime = end_time - start_time
        inference_time.append(inftime)
    inferencetime = (np.mean(inference_time))/2
    inferences.append(inferencetime)

print (', '.join(map(str, inferences)))
#print (f"Inference took {inferencetime:.2f} seconds.")