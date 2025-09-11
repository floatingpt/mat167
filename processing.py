import numpy as np
import numpy.linalg as npl
from pydub import AudioSegment
import os
from matplotlib import pyplot as plt
import scipy.io as io # save output as matlab file
from scipy.io import savemat
'''
Note - This file takes a long time to run (30+ mins). this file loads and samples 7gb of data. 

If you want to use reduced model , change i counter in line 31 to the number of desired subdirectories

The reduced model may not perform as well, but it will still work (however, i will add features in extraction file so the model can be reduced based on number of chunks chosen)

'''


sample_rate = 80000          # desired sample rate

duration = 30               # seconds per clip

#original code is edited below because it uses nearly 20gb of memory ()
"""
This script processes MP3s from the FMA dataset into normalized arrays,
saves them in chunks to avoid memory overload, and outputs .mat files.
"""


sample_rate = 8000       # lower sample rate to reduce memory (try 20000 if you have RAM)
duration = 30            # seconds per clip
dim = sample_rate * duration
chunk_size = 100         # number of tracks per .mat file

root_directory = "/Users/jpxmaestas/Desktop/fma_small"  # edit this to where directory is located if you want to run full process
out_dir = "processed_chunks" 
os.makedirs(out_dir, exist_ok=True)



def file_scraping():
    """Collect all MP3 file paths from FMA small dataset"""
    file_vector = []
    for i in range(156):  # for each subdirectory
        num = str(i).zfill(3)
        subdirectory = os.path.join(root_directory, num)
        if os.path.isdir(subdirectory):
            for file in os.listdir(subdirectory):
                full_path = os.path.join(subdirectory, file)
                if (
                    os.path.isfile(full_path)
                    and file.endswith(".mp3")
                    and "checksums" not in file.lower()
                    and "readme" not in file.lower() # do not append non-audio files
                ):
                    file_vector.append(full_path)
    return file_vector


def preprocess_file(file, dim, target_sr=sample_rate):
    """Load and preprocess one MP3 file"""
    audio = AudioSegment.from_file(file, format="mp3")
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.iinfo(audio.array_type).max  # normalize [-1, 1]

    # pad or trim to return matrix with correct size
    if len(samples) > dim:
        samples = samples[:dim]
    elif len(samples) < dim:
        pad_width = dim - len(samples)
        samples = np.pad(samples, (0, pad_width))

    return samples


def process_in_chunks(file_vector, dim, chunk_size):
    for i in range(0, len(file_vector), chunk_size):
        chunk_files = file_vector[i:i + chunk_size]
        audio_instances = []

        for file in chunk_files:
            try:
                samples = preprocess_file(file, dim)
                audio_instances.append(samples)
            except Exception as e:
                print(f"Skipping {file}: {e}")

        audio_instances = np.array(audio_instances, dtype=np.float32)
        out_path = os.path.join(out_dir, f"audio_data_{i//chunk_size}.mat")
        savemat(out_path, {"audio_instances": audio_instances})

        print(f"Saved {audio_instances.shape[0]} clips to {out_path}")



if __name__ == "__main__":
    file_vector = file_scraping()
    print(f"Found {len(file_vector)} files.")

    process_in_chunks(file_vector, dim, chunk_size) # rewrite instance each time to reduce memory usage
    print("Processing complete. Chunks saved in:", out_dir)




'''
def load_mp3(file_vector, dim, target_sr=sample_rate):
    audio_instances = []
    #Load and preprocess all MP3 files (Note - this takes a while to run!!)
    for file in file_vector:
        file = file_vector[0]
        audio = AudioSegment.from_file(file, format="mp3")
        audio = audio.set_frame_rate(target_sr).set_channels(1) # 20000 per second

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(audio.array_type).max  # normalize [-1, 1]

        # pad or trim to match target length
        if len(samples) > dim:
            samples = samples[:dim]
        elif len(samples) < dim:
            pad_width = dim - len(samples)
            samples = np.pad(samples, (0, pad_width))
        
        audio_instances.append(samples)
    return audio_instances



def plot_audio_instance(audio_instance, sr): # visualize mp3 in time/amplitude domain
    num_frames = len(audio_instance)
    time_axis = np.arange(0, num_frames) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, audio_instance, linewidth=1)
    plt.grid(True)
    plt.title("audio_instance")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()


file_vector = file_scraping()
audio_instances = load_mp3(file_vector, dim)
io.savemat("audio_data.mat", {"audio_instances": audio_instances}) #save to .mat file for easier access
print("Saved 'audio_instances' to audio_data.mat")

'''