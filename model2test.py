import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import time
import pandas as pd
import numpy as np

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")



# writing to files
def write_to_file(file_path, one, two, three, four, five, six):
    length = [0,1,2,3,4,5,6,7,8,9,10]
    data = {'audio length':length,'ecapa embeddings inference': one , 'ecapa scoring inference': two, 'wavlm embeddings inference': three, 'wavlm scoring inference': four, 'ecapa embeddings size': five, 'wavlm embeddings size': six}
    df = pd.DataFrame(data)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"Data has been successfully written to {file_path}.")

ecapa_model_embeddings_inferences = []
ecapa_model_scoring_inferences = []
wavlm_model_embeddings_inferences = []
wavlm_model_scoring_inferences = []
ecapa_embeddings_size = []
wavlm_embeddings_size = []
for j in range(11):
    inference_time1=[] # for ecapa embeddings
    inference_time2=[] # for ecapa cosine distance
    inference_time3=[] # for wavlm embeddings
    inference_time4=[] # for wavlm cosine distance
    if j == 0:
        samplingrate= 8000
    else:
        samplingrate = j * 16000 
    sample1 = torch.randn(1,samplingrate)
    sample2 = torch.randn(1,samplingrate)

    for i in range(1000):
        start1_time = time.time()
        embeddings = classifier.encode_batch(sample1)
        end1_time = time.time()
        inferenceTime1 = end1_time - start1_time # for ecapa embeddings
        embeddings_1 = classifier.encode_batch(sample2)


        start2_time = time.time()
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output = cosine_similarity(embeddings, embeddings_1)
        end2_time = time.time()
        inferenceTime2 = end2_time - start2_time



        start3_time = time.time()
        waveform1 = sample1.squeeze()
        input1 = feature_extractor(waveform1, return_tensors="pt")
        embeddings2 = model(**input1).embeddings
        end3_time = time.time()
        inferenceTime3 = end3_time - start3_time

        waveform2 = sample2.squeeze()
        inputs2 = feature_extractor(waveform2, return_tensors="pt")
        embeddings3 = model(**inputs2).embeddings


        start4_time = time.time()
        normalized_embeddings1 = torch.nn.functional.normalize(embeddings2, dim=-1)
        normalized_embeddings2 = torch.nn.functional.normalize(embeddings3, dim=-1)
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(normalized_embeddings1, normalized_embeddings2).item()
        end4_time = time.time()
        inferenceTime4 = end4_time - start4_time
        inference_time1.append(inferenceTime1)
        inference_time2.append(inferenceTime2)
        inference_time3.append(inferenceTime3)
        inference_time4.append(inferenceTime4)
    infTime1= (np.mean(inference_time1))
    infTime2= (np.mean(inference_time2))
    infTime3= (np.mean(inference_time3))
    infTime4= (np.mean(inference_time4))

    ecapa_model_embeddings_inferences.append(infTime1)
    ecapa_model_scoring_inferences.append(infTime2)
    wavlm_model_embeddings_inferences.append(infTime3)
    wavlm_model_scoring_inferences.append(infTime4)
    ecapa_embeddings_size.append(embeddings_1.size())
    wavlm_embeddings_size.append(embeddings3.size())

file_path = 'inferences.tsv'
write_to_file(file_path, ecapa_model_embeddings_inferences, ecapa_model_scoring_inferences, wavlm_model_embeddings_inferences,wavlm_model_scoring_inferences,ecapa_embeddings_size,wavlm_embeddings_size)


# sample = torch.randn(1,16000)

# start_time = time.time()
# #waveform1, _ = torchaudio.load(sample)
# waveform1 = sample.squeeze()
# inputs = feature_extractor(waveform1, return_tensors="pt")
# embeddings1 = model(**inputs).embeddings
# end_time = time.time()
# inference_time = end_time - start_time
# print(f"the inference took {inference_time} seconds")

# sample2 = torch.randn(1,16000)
# waveform2 = sample2.squeeze()
# inputs2 = feature_extractor(waveform1, return_tensors="pt")
# embeddings2 = model(**inputs).embeddings
# start_time2 = time.time()

# normalized_embeddings1 = torch.nn.functional.normalize(embeddings1, dim=-1)
# normalized_embeddings2 = torch.nn.functional.normalize(embeddings2, dim=-1)
# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
# similarity = cosine_sim(normalized_embeddings1, normalized_embeddings2).item()

# end_time2 = time.time()
# inference2= end_time2 - start_time2
# print(f"cosine similarity took {inference2} seconds")