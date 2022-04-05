import numpy as np
import os
import librosa
import sys
from scipy.io.wavfile import read
import torch
from tqdm import tqdm
import soundfile as sf

def from_str_to_encoded(sequence_string) :
    vector = sequence_string.split("_")
    vector = [int(x) for x in vector]
    return vector
def stereo_create_wholetempo(pathin,files, pathout) :
    """
    This function takes the path to a dir containing the wav files, the path to the dir where npy files will be written, and the sampling rate of the files
    The npy files are files containing the melfs corresponding to each file
    """
    try : 
        filename = files[pathin]
        sr = 44100
        y, sr = librosa.load(filename, sr=44100, mono=False)
        y=librosa.to_mono(y)
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        print(tempo)
        fname = filename.split("/")[-1].split(".")[0] + ".npy"
        np.save(os.path.join(pathout, fname) , tempo)
        return 1
    except : 
        print("problem")
        return 1


def create_wholetempo(pathin,files, pathout) :
    """
    This function takes the path to a dir containing the wav files, the path to the dir where npy files will be written, and the sampling rate of the files
    The npy files are files containing the melfs corresponding to each file
    """
    try : 
        filename = files[pathin]
        sr = 16000
        y, sr = librosa.load(filename, sr=16000)
        hop_length = 512
        S = (librosa.feature.melspectrogram(y=y, sr=sr))
        oenv = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
        tempo = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                                                                    hop_length=hop_length)

        fname = filename.split("/")[-1].split(".")[0] + ".npy"
        np.save(os.path.join(pathout, fname) , tempo)
        return 1
    except : 
        print("problem")
        return 1

def create_tempo(pathin,files, pathout) :
    """
    This function takes the path to a dir containing the wav files, the path to the dir where npy files will be written, and the sampling rate of the files
    The npy files are files containing the melfs corresponding to each file
    """
    try : 
        filename = files[pathin]
        sr = 16000
        y, sr = librosa.load(filename, sr=16000)
        hop_length = 512
        tempogram = librosa.feature.tempogram(y=y, sr=sr,
                                          hop_length=hop_length)

        fname = filename.split("/")[-1].split(".")[0] + ".npy"
        np.save(os.path.join(pathout, fname) , tempogram)
        return 1
    except : 
        print("problem")
        return 1
def create_chroma(pathin,files, pathout) :
    """
    This function takes the path to a dir containing the wav files, the path to the dir where npy files will be written, and the sampling rate of the files
    The npy files are files containing the melfs corresponding to each file
    """
    try :
        filename = files[pathin]
        sr = 16000
        y, sr = librosa.load(filename, sr=sr)
        chroma=librosa.feature.chroma_stft(y=y, sr=sr)
        fname = filename.split("/")[-1].split(".")[0] + ".npy"
        np.save(os.path.join(pathout, fname) , chroma)
        return 1
    except : 
        print("problem")
        return 
def load_one_hot_dict(files_onehot) :
    filein= open(files_onehot, "r")
    lines = filein.read().splitlines()
    one_hot_encoder = dict()
    for line in lines :
        wavname, transcription = line.split("|")
        sentence = transcription.split('_')
        one_hot_encoder[wavname] = sentence
    return one_hot_encoder

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate
def soundfile_load_wav_to_torch(full_path) : 
    data, sampling_rate = sf.read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        filepaths_and_text= [(x[1], x[2], x[3], x[0])for x in filepaths_and_text]
    return filepaths_and_text
def triplets_filepaths_and_text(filename, firstsplit="|" , secondsplit= "%" ) :

    with open(filename, encoding='utf-8') as f:
        triplets = [line.strip().split(secondsplit) for line in f ]
    final_triplets = []
    for triplet in tqdm(triplets) :
        split_points = [x.split(firstsplit) for x in triplet]
        triplet_cleaned = [(x[1], x[2],x[3], x[0]) for x in split_points]
        final_triplets.append(triplet_cleaned)
    return final_triplets

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def initialize_speakers_triplets(loaded_triplets) :
    speakers_dict = {}
    counter = 0
    for triplets in loaded_triplets  :
        potential_speakers = [x[0] for x in triplets]
        for spk in potential_speakers :
            if spk not in speakers_dict :
                speakers_dict[spk] = counter
                counter +=1
    return speakers_dict

def initialize_speakers(loaded_elements, mode) :
    speakers_dict = {}
    counter = 0
    for element in loaded_elements :
        spk = element[0]
        if spk not in speakers_dict :
       	    speakers_dict[spk] = counter
            counter +=1
    j= open("melf_new_dict_"+ mode+ ".txt", "w")
    for speak in speakers_dict :
        j.write(":".join([speak, str(speakers_dict[speak])]))
        j.write("\n")
    j.close()
    print("SPEAKERS DICT WRITTEN")
    return speakers_dict

def initialize_languages(loaded_elements) :
    languages_dict = {}
    counter = 0
    for element in loaded_elements :
        lng = element[3]
        if lng not in languages_dict :
       	    languages_dict[lng] = counter
            counter +=1
    return languages_dict

def initialize_languages_triplets(loaded_triplets) :
    languages_dict = {}
    counter = 0
    for triplets in loaded_triplets :
        potential_languages = [x[3] for x in triplets]
        for lng in potential_languages :

            if lng not in languages_dict :
       	        languages_dict[lng] = counter
                counter +=1
    return languages_dict


def load_speakers_dict(dictpath) :
    resulting_dict = {}
    filein = open(dictpath, "r")
    lines = filein.read().splitlines()
    for line in lines :
        speaker, code = line.split(":")
        resulting_dict[speaker] = int(code)
    print(resulting_dict)
    return resulting_dict




if __name__=="__main__" :
    filein = sys.argv[1]
    points = triplets_filepaths_and_text(filein)
    print(len(points))
    print(points[0])
