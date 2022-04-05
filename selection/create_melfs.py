import numpy as np
import sys
import os
import librosa
import torch
from tqdm import tqdm
from speechbrain.lobes.features import Fbank
import multiprocessing as mp
from multiprocessing import Pool
from utils import load_wav_to_torch, load_filepaths_and_text, from_str_to_encoded , initialize_speakers_triplets,  initialize_speakers , triplets_filepaths_and_text
import warnings
warnings.filterwarnings("ignore")

max_wav_value = 32768.0
fbank = Fbank()
 
def treat_file(filename): 
    audio, sampling_rate = load_wav_to_torch(filename)
    fname = filename.split("/")[-1].split(".")[0] + ".npy"
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = fbank(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    melspec= melspec.cpu().detach().numpy()
    np.save(os.path.join(pathout, fname) , melspec)


def create_melfs(pathin, pathout, sampling_rate) :
    """
    This function takes the path to a dir containing the wav files, the path to the dir where npy files will be written, and the sampling rate of the files
    The npy files are files containing the melfs corresponding to each file
    """
    if not os.path.exists(pathout) :
        os.makedirs(pathout)
    
    files = [os.path.join(pathin, x) for x in os.listdir(pathin) ]

    p=Pool(mp.cpu_count())
    print(f"computing mels on {len(files)} files ")
    sr = sampling_rate
    r = list(tqdm(p.imap(treat_file, files),
                  total=len(files)))


if __name__=="__main__" :
    pathin = sys.argv[1]
    pathout = sys.argv[2]
    sr=16000
    create_melfs(pathin, pathout, sr)
