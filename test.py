# import libraries
import torch
import time
import torchaudio
import torchinfo

# import embeddings model and calculate embeddings for one sample
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('/home/omerhatim/thesis/speechbrain dataset/speechbrain/tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)
print (embeddings)


# calculate inference time for one sample 
sample = torch.randn(1,32000)
start_time = time.time()
classifier.encode_batch(sample)
end_time = time.time()
inference_time = end_time - start_time
print (f"Inference took {inference_time:.2f} seconds.")


#print summary of the model, parameters, size, etc
torchinfo.summary(classifier, input_size=(1,16000))