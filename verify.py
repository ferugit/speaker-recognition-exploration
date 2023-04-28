# import libraries and model
import torch
import time
import torchaudio
import torchinfo
from torchsummary import summary
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")


# perform verification
score, prediction = verification.verify_files('/home/omerhatim/thesis/speechbrain dataset/speechbrain/tests/samples/ASR/spk1_snt1.wav','/home/omerhatim/thesis/speechbrain dataset/speechbrain/tests/samples/ASR/spk2_snt1.wav') # Different Speakers
print (score,prediction)


# inference time for verification

sample1 = torch.randn(1,160000)
sample2 = torch.randn(1,160000)
start_time = time.time()
verification.verify_batch(sample1,sample2)
end_time = time.time()
inference_time = end_time - start_time
print (f"Inference took {inference_time:.2f} seconds.")