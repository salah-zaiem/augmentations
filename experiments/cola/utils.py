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
import warnings


def load_params(params_path):
    with open(params_path) as json_file:
        data = json.load(json_file)

    return data

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


