from sklearn.cluster import SpectralClustering 
import numpy as np
import numpy.linalg as npl
from scipy import io
import matplotlib.pyplot as plt
import librosa
import os



chunk_dir = "processed_chunks"   # folder of 79 matlab files
out_file = "audio_data_features.mat" #new file 
sample_rate = 8000  
os.makedirs("feature_chunks", exist_ok= True)



def load_all_chunks(chunk_dir):
    all_audio = []
    for fname in sorted(os.listdir(chunk_dir)):
        if fname.endswith(".mat"):
            data = io.loadmat(os.path.join(chunk_dir, fname))
            audio_instances = data["audio_instances"]
            all_audio.append(audio_instances)
            print(f"Loaded {fname} with shape {audio_instances.shape}")
    return np.vstack(all_audio)  # combine into one big array

step = 128
w_size = 256

import librosa
import numpy as np


def extract_features(audio, sr=8000):   #### important - i will use mfc in lieu of fourier transformation

    #
    # this medium article does a great job of explaining the usage of mfcc algorithm
    # https://jonathan-hui.medium.com/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9


    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    # spectral centroid measures brightness
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))

    #bandwidth is the measure of spread within the data
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))

    # measures rate at which frequency changes 
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))


    # measures changes in power spectrum over time
    flux = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))

    # store column_wise 
    features = np.hstack([mfccs_mean, mfccs_var, centroid, bandwidth, rolloff, flux])

    return features # 7997 x 30


def process_chunk(fname, sr=sample_rate):
    data = io.loadmat(fname)
    audio_instances = data["audio_instances"]

    features = []
    for i, audio in enumerate(audio_instances):
        try:
            f = extract_features(audio, sr=sr)
            features.append(f)
        except Exception as e:
            print(f"Skipping track {i} in {fname}: {e}")

    features = np.vstack(features) # store as 
    base = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(chunk_dir, f"{base}_features.mat")
    io.savemat(out_path, {"feature_vector": features})
    print(f"Saved {features.shape[0]} feature vectors to {out_path}")




for fname in sorted(os.listdir(chunk_dir)):
    if fname.endswith(".mat"):
        process_chunk(os.path.join(chunk_dir, fname))
print("Saved all features to audio_data_features.mat")



''' # fixed because this crashed my computer 3 times 

def run_feature_extraction(audio_instances, sr=sample_rate): #iterate over instances in files
    features = []
    for i, audio in enumerate(audio_instances): #
        try:
            f = extract_features(audio, sr=sr) # iterate and append
            features.append(f)
        except Exception as e: # skip if needed
            print(e)
    return np.vstack(features)



audio_instances = load_all_chunks(chunk_dir)
print("Full dataset shape:", audio_instances.shape)

features = run_feature_extraction(audio_instances, sr=sample_rate)
print("Feature matrix shape:", features.shape)

io.savemat("audio_data_features.mat", {
    "audio_instances": audio_instances,
    "feature_vector": features})

print("Saved features to audio_data_features.mat")

'''