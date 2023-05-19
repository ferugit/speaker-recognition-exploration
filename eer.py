# import libraries
import os
import argparse
import itertools
import pandas as pd

#import the model
from speechbrain.pretrained import SpeakerRecognition


def write_to_file(file_path, gt, pred, score):
    data = {'true label': gt, 'predection': pred, 'score': score}
    df = pd.DataFrame(data)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"Data has been successfully written to {file_path}.")


parser = argparse.ArgumentParser(description='KWS models')

parser.add_argument('--prefix', default='', help='path to data')
parser.add_argument('--savedfile', default='', help='where to place the results')

args = parser.parse_args()

prefix = args.prefix
savedfile = args.prefix

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# read database excel file
df = pd.read_csv('Data/dataset.tsv', header = 0, sep = '\t')
A = df['Speaker_ID'].value_counts()

unique_speaker_ids = A.index
string_to_number = {string: idx for idx, string in enumerate(unique_speaker_ids)}

# Map the 'Speaker_ID' column using the dictionary
df['Speaker_ID'] = df['Speaker_ID'].map(string_to_number)

B = df['Speaker_ID'].value_counts()

# Assuming you have a DataFrame called 'df' with a column 'path' containing the file paths
audio_files = df['Filename'].tolist()

# Add the prefix to each file path and keep only .wav files
audio_files = [os.path.join(prefix, f) for f in audio_files if f.endswith(".wav")]

# Pairwise comparison of audio files and speakers
pairs = []
unique_speaker_ids = df['Speaker_ID'].unique()

for speaker_id in unique_speaker_ids:
    speaker_files = df.loc[df['Speaker_ID'] == speaker_id, 'Filename'].tolist()
    speaker_files = [os.path.join(prefix, f) for f in speaker_files if f.endswith(".wav")]
    speaker_pairs = list(itertools.combinations(speaker_files, 2))
    speaker_pairs_with_id = [((pair[0], pair[1]), (speaker_id, speaker_id)) for pair in speaker_pairs]
    pairs.extend(speaker_pairs_with_id)

scores = []
true_labels = []
speaker_files = []

A = df['Speaker_ID'].to_list()
B = df['Filename'].to_list()
speaker_files = list(zip(A, B))

pairs = []
for speaker1, file1 in speaker_files:
    for speaker2, file2 in speaker_files:
        pairs.append([prefix+file1, prefix+file2, speaker1, speaker2])

predictions = []
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