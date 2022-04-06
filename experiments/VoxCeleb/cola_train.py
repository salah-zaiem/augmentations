#!/usr/bin/env python3
import torch.nn.functional as F
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

from speechbrain.utils.distributed import run_on_main
"""Recipe for training a multi-task self supervised learning model.
Representations are learned in a self-supervised way through learning to
solve a set of pretext tasks, i.e learning to predict a set of pretext labels
like spectrograms or signal features.
More details here: 
https://arxiv.org/abs/2001.09239
https://arxiv.org/abs/2104.07388

To run this recipe, do the following:
> python train.py hparams/train.yaml
With the default hyperparameters, the system employs a CRDNN encoder.

Authors :
    Salah Zaiemfrom efficientnet_pytorch import EfficientNet

    Titouan Parcollet

"""
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
            batch_size,width= img.size()
            # pad the width if needed
            random_starts = torch.randint(0,width,size=(batch_size,))
            new_batch=[]
            for i in range(batch_size) :
                if self.pad_if_needed and random_starts[i]+self.crop_size > width:
                    padding = [self.crop_size +random_starts[i] - width, 0]
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
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        first_elements = randomaudiocrop(wavs)
        second_elements= randomaudiocrop(wavs)
        considered_len=1.00
        first_elements_lens = torch.tensor([considered_len for x in range(self.hparams.batch_size)]) 

        second_elements_lens = torch.tensor([considered_len for x in range(self.hparams.batch_size)]) 
        feats_first = self.hparams.compute_features(first_elements)
        feats_second = self.hparams.compute_features(second_elements)
        feats_first = self.modules.normalize(feats_first, first_elements_lens )
        feats_second = self.modules.normalize(feats_second, second_elements_lens)
        #print(f"feats shape beginning : {feats_first.size()}")

        encoded_first_reps=self.modules.enc(self.modules.cnn_start(feats_first.unsqueeze(1).detach()))

        #print(f"encoded shape : {encoded_first_reps.size()}")
        projected_first_reps=self.modules.projector(encoded_first_reps.squeeze())
        projected_first_reps=torch.tanh(layernorm(projected_first_reps))
        encoded_second_reps =self.modules.enc(self.modules.cnn_start(feats_second.unsqueeze(1).detach()))
        projected_second_reps=self.modules.projector(encoded_second_reps.squeeze())
        projected_second_reps=torch.tanh(layernorm(projected_second_reps))
        projected_reps = [projected_first_reps, projected_second_reps]
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
        if self.acc_counter %1000==0:
            print(np.mean(self.accuracies))
            self.acc_counter=0
            self.accuracies=[]
        #print(f"train_acc:  {acc}")
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
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig"],
        )
    return train_data, valid_data, test_data


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
    hparams["modules"]["enc"]=enc
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
    if os.path.exists(hparams["encoder_path"]):
        print("loading efficient net")
        print(hparams["encoder_path"])
        ssl_brain.modules.enc.load_state_dict(torch.load(hparams["encoder_path"]))
    if os.path.exists(hparams["bili_path"]):
        print("loading bilinear matrix")
        ssl_brain.modules.bilinear.load_state_dict(torch.load(hparams["bili_path"]))




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
