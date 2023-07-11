import torch
import torchinfo
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from speechbrain.pretrained import EncoderClassifier


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')


# Print the Spkrec-ecapa-voxceleb summary
torchinfo.summary(classifier, input_size=(1,48000))

# Print the Wavlm-base model summary 
torchinfo.summary(model,(1,48000))
