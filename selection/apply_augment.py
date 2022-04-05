import os
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import torchaudio
import json
from tqdm import tqdm
import numpy as np
import sys
import random
import augment
from augment import EffectChain
import torch
from dataclasses import dataclass
from create_params_dict import create_params_sample
print(mp.cpu_count())
class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)

class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max
    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)

@dataclass
class RandomReverb:
    reverberance_min: int = 50
    reverberance_max: int = 50
    damping_min: int = 50
    damping_max: int = 50
    room_scale_min: int = 0
    room_scale_max: int = 100

    def __call__(self):
        reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
        damping = np.random.randint(self.damping_min, self.damping_max + 1)
        room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)

        return [reverberance, damping, room_scale]
def freq2mel(f):
    return 2595. * np.log10(1 + f / 700)

def mel2freq(m):
    return ((10.**(m / 2595.) - 1) * 700)


class SpecAugmentBand:
    def __init__(self, sampling_rate, scaler):
        self.sampling_rate = sampling_rate
        self.scaler = scaler

    def __call__(self):
        F = 27.0 * self.scaler
        melfmax = freq2mel(self.sampling_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.)
        melf0 = np.random.uniform(0, melfmax - meldf)
        low = mel2freq(melf0)
        high = mel2freq(melf0 + meldf)
        return f'{high}-{low}'


def augmentation_factory(params, sampling_rate):
    chain = augment.EffectChain()
    effects_dict = params["effects_probs"]
    for effect in effects_dict:
        if effect == 'bandreject':
            if random.random()< effects_dict[effect] : 
                chain = chain.sinc('-a', '120',
                                   SpecAugmentBand(sampling_rate,params["band_scaler"]))
        elif effect == 'pitch':

            if random.random()< effects_dict[effect] : 
                pitch_randomizer = RandomPitchShift(params["pitch_shift_max"])
                if random.random()<params["pitch_quick_prob"]:
                    chain = chain.pitch('-q', pitch_randomizer).rate('-q', sampling_rate)
                else:
                    chain = chain.pitch(pitch_randomizer).rate(sampling_rate)
        elif effect == 'reverb':

            if random.random()< effects_dict[effect] : 
                randomized_params = RandomReverb(params["reverberance_min"],
                                                 params["reverberance_max"], 
                                    params["damping_min"],
                                                 params["damping_max"],
                                                 params["room_scale_min"],
                                                 params["room_scale_max"])
                chain = chain.reverb(randomized_params).channels()
        elif effect == 'time_drop':

            if random.random()< effects_dict[effect] : 
                chain = chain.time_dropout(max_seconds=params["t_ms"] / 1000.0)
        elif effect == 'clip':

            if random.random()< effects_dict[effect] : 
                chain = chain.clip(RandomClipFactor(params["clip_min"],
                                                    params["clip_max"]))
        elif effect == 'none':
            pass
        else:
            raise RuntimeError(f'Unknown augmentation type {effect}')
    return chain
def get_files (files, n) : 
    #sampling method here, start with n points per class
    classes = list(set([x.split("_")[0] for x in files]))
    print(f"number of classes : {len(classes)}")
    sampled_files= []
    for c in classes[0:20] : 
        candidates = [ x for x in files if x.split("_")[0]==c]
        random.shuffle(candidates)
        p = min(len(candidates), n)
        sampled_files+= candidates[0:p]
    print(f"number of sampled_files : {len(sampled_files)}")
    return sampled_files

def load_params(params_path):
    with open(params_path) as json_file:
        data = json.load(json_file)

    return data

def file_function (fil, aug_params, output_dir, views_number, sampling_rate): 
    x, sampling_rate = torchaudio.load(os.path.join(dataset,fil))

    for num in range(views_number): 

        augmentation_chain = augmentation_factory(aug_params, sampling_rate)


        y = augmentation_chain.apply(x, 
                src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
                target_info=dict(rate=sampling_rate, length=0)
        )


        fname =fil.split("/")[-1].split(".")[0]+"number"+str(num)+".wav"
        torchaudio.save(os.path.join(output_dir,fname), y, sampling_rate)



def augment_dataset(sampled_files, aug_parameters, output_dir, views_number,
                    sampling_rate) :
    augment_file = partial(file_function, aug_params=aug_parameters,
                           output_dir=output_dir, views_number=views_number,
                           sampling_rate=sampling_rate)
    p=Pool(mp.cpu_count())
    r = list(tqdm(p.imap(augment_file, sampled_files),
                  total=len(sampled_files)))
"""
    for filein in tqdm(sampled_files) :
        x, sampling_rate = torchaudio.load(os.path.join(dataset,filein))

        noise_generator = lambda: torch.zeros_like(x).uniform_()
        for num in range(views_number): 

            augmentation_chain = augmentation_factory(aug_parameters, sampling_rate)


            y = augmentation_chain.apply(x, 
                    src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
                    target_info=dict(rate=sampling_rate, length=0)
            )


            fname =filein.split("/")[-1].split(".")[0]+str(num)+".wav"
            torchaudio.save(os.path.join(output_dir,fname), y, sampling_rate)


"""
def check_args(params):

    if not (0 <= params["room_scale_min"] <= params["room_scale_max"] <= 100):
        raise RuntimeError('It should be that 0 <= room_scale_min <= room_scale_max <= 100')

    if not (0 <= params["reverberance_min"] <= params["reverberance_max"] <= 100):
        raise RuntimeError('It should be that 0 <= reverberance_min <= reverberance_max <= 100')

    if not (0 <= params["damping_min"] <= params["damping_max"] <= 100):
        raise RuntimeError('It should be that 0 <= damping_min <= damping_max <= 100')

    if not (0.0 <= params["clip_min"] <= params["clip_max"] <= 1.0):
        raise RuntimeError('It should be that 0 <= clip_min <= clip_max <= 1.0')
    
    print("params have been successfully checked")
    return 


def augment_randomly(dataset,views_number, output_dir, sampling_rate):
    nameout_params=create_params_sample(output_dir) 
    aug_dir= os.path.join(output_dir, "augmented_dataset")
    if not os.path.exists(os.path.join(output_dir, "augmented_dataset")):
        os.makedirs(os.path.join(output_dir, "augmented_dataset"))
    params = load_params(nameout_params)
    check_args(params)
    sampled_files = get_files(os.listdir(dataset),20)
    augment_dataset(sampled_files, params, aug_dir, views_number,
                    sampling_rate)



if __name__=="__main__" : 
    sampling_rate = 16000
    dataset= sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    views_number=100
    augment_randomly(dataset, views_number, output_dir, sampling_rate)
