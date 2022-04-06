#!/usr/bin/env python3
from tqdm import tqdm
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)

import torch.nn.functional as F
from functools import partial
import multiprocessing as mp
import time
import os
import numpy as np
import sys
from efficientnet_pytorch import EfficientNet
from torch.nn import Tanh, LayerNorm
from torch.nn.utils.rnn import pad_sequence
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.features import MFCC
from speechbrain.processing.features import spectral_magnitude, Deltas
from utils import  RandomPitchShift, RandomReverb, freq2mel, mel2freq, SpecAugmentBand 
from utils import RandomClipFactor, load_params
import random
import augment
from augment import EffectChain
from dataclasses import dataclass

from speechbrain.utils.distributed import run_on_main


"""
Authors :
    Salah Zaiem

"""
def crop_wav(wav, crop_size=16000):

    sizei = wav.size(0)
    random_start = torch.randint(0,sizei, (1,)).item()
    if  random_start+crop_size > sizei:
        padding = [crop_size +random_start - sizei, 0]
        new_img = F.pad(wav, padding, value=0, mode="constant")
        return new_img[random_start:random_start+crop_size]
    else :
        return wav[random_start:random_start+crop_size]

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

class BilinearProduct(torch.nn.Module):
  """Bilinear product."""

  def __init__(self, dim, epsilon=0.13):
    super().__init__()
    self.dim = dim

    v = torch.zeros(dim,dim) + torch.randn(dim,dim) *epsilon
    self.w = torch.nn.Parameter(v
    )
    
  def forward(self, positive, anchor):
    projection_positive = torch.mm(self.w, anchor.t())
    return torch.mm(positive, projection_positive)

class RandomCropAudio(torch.nn.Module):
        def __init__(self, size, pad_if_needed=True, fill=0, padding_mode="constant"):
            super().__init__()
            self.crop_size = size
            self.pad_if_needed = pad_if_needed
            self.fill = fill
            self.padding_mode = padding_mode
        def forward(self, img):
            width = img[-1].size()[0] 
            batch_size= len(img)
            # pad the width if needed
            random_starts = torch.randint(0,width,size=(batch_size,))
            new_batch=[]
            for i in tqdm(range(batch_size)) :
                sizei = img[i].size(0)
                if self.pad_if_needed and random_starts[i]+self.crop_size > sizei:
                    padding = [self.crop_size +random_starts[i] - sizei, 0]
                    new_img = F.pad(img[i], padding, value=self.fill, mode=self.padding_mode)
                    new_batch.append(new_img[random_starts[i]:random_starts[i]+self.crop_size])
                else :
                    new_batch.append(img[i][random_starts[i]:random_starts[i]+self.crop_size])
            return torch.vstack(new_batch)
randomaudiocrop= RandomCropAudio(16000)
class COLA(sb.core.Brain):
# Define training procedureclass SSL(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        wavs =  torch.vstack([x[0] for x in batch.sig])
        second_wavs =  torch.vstack([x[1] for x in batch.sig])
        first_elements = wavs.to(self.device)
        second_elements= second_wavs.to(self.device)
        considered_len=1.00
        first_elements_lens = torch.tensor([considered_len for x in range(self.hparams.batch_size)]) 
        second_elements_lens = torch.tensor([considered_len for x in range(self.hparams.batch_size)]) 
        feats_first = self.hparams.compute_features(first_elements)
        feats_second = self.hparams.compute_features(second_elements)
        feats_first = self.modules.normalize(feats_first, first_elements_lens )
        feats_second = self.modules.normalize(feats_second, second_elements_lens)
        encoded_first_reps=self.modules.enc(self.modules.cnn_start(feats_first.unsqueeze(1).detach()))
        projected_first_reps=self.modules.projector(encoded_first_reps.squeeze())
        projected_first_reps=torch.tanh(layernorm(projected_first_reps))
        encoded_second_reps =self.modules.enc(self.modules.cnn_start(feats_second.unsqueeze(1).detach()))
        projected_second_reps=self.modules.projector(encoded_second_reps.squeeze())
        projected_second_reps=torch.tanh(layernorm(projected_second_reps))
        projected_reps = [projected_first_reps, projected_second_reps]
        batch_chain = augmentation_factory(self.augparams, 16000)
        return projected_reps 
    def compute_objectives(self, representations):
        # Load prediction
        first, second = representations
        #similarities = self.modules.similarity(first, second)
        similarities = self.modules.bilinear(first,second)
        target_similarities= torch.arange(first.size(0), device=self.device)
        # Would need to add temperature here
        loss = self.hparams.train_loss(similarities, target_similarities)
        _, predicted = torch.max(similarities, 1)
        acc = (predicted == target_similarities).double().mean().item()
        self.acc_counter+=1
        if self.acc_counter %200==0:
            print(np.mean(self.accuracies))
            self.acc_counter=0
            self.accuracies=[]
        self.accuracies.append(acc)
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        representations = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(representations)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        reps= self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            losses = self.compute_objectives(reps
            )
        return losses.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage==sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only()

            torch.save(self.modules.enc.state_dict(), self.hparams.encoder_path)
            torch.save(self.modules.bilinear.state_dict(), self.hparams.bili_path)


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},

            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = valid_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    # 2. Define audio pipeline:

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        if random.random()>0:
            second_sig = batch_chain.apply(resampled, 
                src_info=dict(rate=16000, length=resampled.size(0), channels=1),
                target_info=dict(rate=16000, length=0)
                )
            second_sig= torch.squeeze(second_sig)
        else:
            second_sig=resampled

        resampled = crop_wav(resampled)
        second_sig = crop_wav(second_sig)
        return resampled, second_sig
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig"],
        )
    return train_data, valid_data, test_data


"""    
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("augmented")
    def augment_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        second_sig = resampled
        if hparams["apply_augmentations"]  and random.random()>-1: 
            second_sig = batch_chain.apply(resampled, 
                src_info=dict(rate=16000, length=resampled.size(0), channels=1),
                target_info=dict(rate=16000, length=0)
                )

        return  torch.squeeze(second_sig)

    sb.dataio.dataset.add_dynamic_item(datasets, augment_pipeline)
"""
    
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice) uncomment if using Commonvoice

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Due to DDP, we do the preparation ONLY on the main python process
    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_set = dataio_prepare(hparams)
    enc= EfficientNet.from_name(
                    "efficientnet-b0", include_top=False,
        drop_connect_rate=0.1, dropout_rate=0.2
                )
    hparams["modules"]["enc"]=enc.to("cuda:0")
    bilinear = BilinearProduct(hparams["projection_dim"])
    hparams["modules"]["bilinear"]=bilinear

    # Trainer initialization
    ssl_brain = COLA(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )
    if hparams["apply_augmentations"] : 
        ssl_brain.augparams=load_params(hparams["augmentation_file"])

    if os.path.exists(hparams["encoder_path"]):
        print("loading efficient net")
        print(hparams["encoder_path"])
        ssl_brain.modules.enc.load_state_dict(torch.load(hparams["encoder_path"]))
    if os.path.exists(hparams["bili_path"]):
        print("loading bilinear matrix")
        ssl_brain.modules.bilinear.load_state_dict(torch.load(hparams["bili_path"]))

    batch_chain = augmentation_factory(ssl_brain.augparams, 16000)



    layernorm = LayerNorm(normalized_shape=512).to(ssl_brain.device)
   # Adding objects to trainer.
    # Training
    ssl_brain.acc_counter=0
    ssl_brain.accuracies=[]
    ssl_brain.fit(
        ssl_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
