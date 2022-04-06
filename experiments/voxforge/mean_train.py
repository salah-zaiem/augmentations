#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality)
with an ECAPA-TDNN model.

To run this recipe, do the following:

> python train.py hparams/train.yaml --data_folder /path/to/IEMOCAP

Authors
 * Pierre-Yves Yanni 2021
"""
import torch.nn.functional as F

import os
import time
import sys
import csv
import speechbrain as sb
from efficientnet_pytorch import EfficientNet
from torch.nn import Tanh, LayerNorm
import torch
from torch.utils.data import DataLoader
from enum import Enum, auto
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from cola_train import COLA
class ProgressiveCuts(torch.nn.Module):
        def __init__(self, size,step, pad_if_needed=True, fill=0, padding_mode="constant"):
            super().__init__()
            self.crop_size = size
            self.pad_if_needed = pad_if_needed
            self.fill = fill
            self.padding_mode = padding_mode
            self.step=step
        def forward(self, img):
            batch_size,width= img.size()
            # pad the width if needed
            new_images= []
            if width < 19000 : 
                for i in range(batch_size):
                    padding = [self.crop_size +3000 - width, 0]
                    new_images.append(F.pad(img[i], padding, value=self.fill, mode=self.padding_mode))
                width = 19000
            else : 
                new_images = img
            possible_steps = (width - self.crop_size) //self.step

            new_batch=[]
            for i in range(batch_size) :
                elements = []
                for st in range(possible_steps) : 
                    element = new_images[i][st*self.step:(st*self.step)+ self.crop_size]
                    elements.append(element)
                stacked  = torch.vstack(elements)

                new_batch.append(stacked)
            return new_batch
progress = ProgressiveCuts(16000, step=200)
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
class DivideAudio(torch.nn.Module):
        def __init__(self, size, pad_if_needed=True, fill=0, padding_mode="constant"):
            super().__init__()
            self.crop_size = size
            self.pad_if_needed = pad_if_needed
            self.fill = fill
            self.padding_mode = padding_mode
        def forward(self, img):
            batch_size,width= img.size()
            padding_needed=width *( width // self.crop_size+1)
            # pad the width if needed
            elements_per_file = (width // self.crop_size) +1 
            new_batch=[]
            for i in range(batch_size) :
                if self.pad_if_needed :
                    elements= []
                    padding = [padding_needed - width, 0]
                    new_img = F.pad(img[i], padding, value=self.fill, mode=self.padding_mode)
                    for j in range(elements_per_file) : 
                        elements.append(new_img[j*self.crop_size : (j+1)* self.crop_size]) 
                    stacked  = torch.vstack(elements)
                    print(stacked.size())
                    new_batch.append(stacked)
            return new_batch
divide = DivideAudio(16000)
class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        considered_len=1.00
        all_first_elements = progress(wavs)
        all_feats = []
        for first_elements in all_first_elements : 
            first_elements_lens = torch.tensor([considered_len for x in range(first_elements.size(0))]) 
            feats_first = self.modules.compute_features(first_elements.to(self.device))

            feats_first = ssl_brain.modules.normalize(feats_first.to(self.device), first_elements_lens.to(self.device) )
            feats_first=ssl_brain.modules.enc(ssl_brain.modules.cnn_start(feats_first.unsqueeze(1)))
            feats = torch.squeeze(feats_first).detach()
            all_feats.append(torch.unsqueeze(feats, dim=0))
        all_feats = torch.vstack(all_feats)
        all_feats=torch.mean(all_feats, dim=1)
        #feats = torch.squeeze(feats_first)
        #feats = torch.stack([feats_first,feats_first, feats_first, feats_first], dim=1)
        embeddings = self.modules.embedding_model(all_feats)
        outputs = self.modules.classifier(embeddings)
        return outputs

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not finite
            self.check_gradients(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        _, lens = batch.sig
        considered_len=1.00
        emoid, _ = batch.language_encoded
        first_elements_lens = torch.tensor([considered_len for x in range(emoid.size(0))]) 
        lens= first_elements_lens.to(self.device)
        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN or stage==sb.Stage.VALID:

            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)
        
            loss = self.hparams.compute_cost(predictions, emoid, lens)



        if stage == sb.Stage.VALID : 
            self.error_metrics.append(batch.id, predictions, emoid, lens)
        if stage==sb.Stage.TEST :  
            #stage == sb.Stage.VALID:
            loss = self.hparams.compute_cost(predictions, emoid, lens)
            self.error_metrics.append(batch.id, predictions, emoid, lens)
        
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def output_predictions_test_set(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and create output file (id, predictions, true values).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )

            save_file = os.path.join(
                self.hparams.output_folder, "predictions.csv"
            )
            with open(save_file, "w", newline="") as csvfile:
                outwriter = csv.writer(csvfile, delimiter=",")
                outwriter.writerow(["id", "prediction", "true_value"])

        self.on_evaluate_start(max_key=max_key, min_key=min_key)  # done before
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1

                emo_ids = batch.id
                true_vals = batch.language_encoded.data.squeeze(dim=1).tolist()
                all_outputs = self.compute_forward(batch, stage=Stage.TEST)
                predictions = []
                for output in all_outputs : 
                    print(output)
                    prediction = (
                        torch.argmax(output, dim=1).squeeze().tolist()
                    )
                    predictions.append(torch.mode(torch.tensor(prediction)).values.item())
                with open(save_file, "a", newline="") as csvfile:
                    outwriter = csv.writer(csvfile, delimiter=",")
                    for emo_id, prediction, true_val in zip(
                        emo_ids, predictions, true_vals
                    ):
                        outwriter.writerow([emo_id, prediction, true_val])

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
        self.step = 0


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("language")
    @sb.utils.data_pipeline.provides("language", "language_encoded")
    def label_pipeline(language):
        yield language
        language_encoded = label_encoder.encode_label_torch(language)
        yield language_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "language_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    train_data = datasets["train"]
    test_data=datasets["test"]
    valid_data=datasets["valid"]
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="length",

            key_max_value={"length": hparams["avoid_if_longer_than"]},
        )
        valid_data=valid_data.filtered_sorted(

                key_max_value={"length": hparams["avoid_if_longer_than"]},
                sort_key="length")

        test_data=test_data.filtered_sorted(

                key_max_value={"length": hparams["avoid_if_longer_than"]},
                sort_key="length")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="length",

            key_max_value={"length": hparams["avoid_if_longer_than"]},
            reverse=True,
        )
        valid_data=valid_data.filtered_sorted(

                key_max_value={"length": hparams["avoid_if_longer_than"]},
                sort_key="length")


        test_data=test_data.filtered_sorted(
                sort_key="length",

                key_max_value={"length": hparams["avoid_if_longer_than"]},
                reverse=True)

        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    datasets["train"] = train_data
    datasets["test"]=test_data
    datasets["valid"]=valid_data
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="language",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[2:])

    colahparams_file, colarun_opts, colaoverrides = sb.parse_arguments([sys.argv[1]])
    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(colahparams_file) as cfin:
        colahparams = load_hyperpyyaml(cfin, overrides)


    # Dataset preparation (parsing CommonVoice) uncomment if using Commonvoice
    enc= EfficientNet.from_name(
                    "efficientnet-b0", include_top=False,
        drop_connect_rate=0.1, dropout_rate=0.2
                )
    colahparams["modules"]["enc"]=enc.to("cuda:0")
    # Trainer initialization
    ssl_brain = COLA(
        modules=colahparams["modules"],
        hparams=colahparams,
        run_opts=run_opts,
        opt_class=colahparams["opt_class"],
        checkpointer=colahparams["checkpointer"],
    )
    ssl_brain.checkpointer.recover_if_possible()
    if os.path.exists(colahparams["encoder_path"]):
        print("loading efficient net")
        ssl_brain.modules.enc.load_state_dict(torch.load(colahparams["encoder_path"]))



    layernorm = LayerNorm(normalized_shape=512).to(ssl_brain.device)
    ssl_brain.modules.eval()

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    ssl_brain.modules.normalize = ssl_brain.modules.normalize.to(emo_id_brain.device)
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
   )

    # Create output file with predictions
    """
    emo_id_brain.output_predictions_test_set(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
    """
