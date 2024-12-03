#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import itertools
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import yaml
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.dataset import GeneratorDataset, transforms, vision
from mindone.safetensors.mindspore import save_file
from mindone.diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from mindone.diffusers._peft import LoraConfig, set_peft_model_state_dict
from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer
from mindone.diffusers._peft.utils import get_peft_model_state_dict
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import (
    AttrJitWrapper,
    TrainStep,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    init_distributed_device,
    is_master,
    set_seed,
)
from mindone.diffusers.utils import convert_unet_state_dict_to_peft

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = logging.getLogger(__name__)


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    trackers,
    logging_dir,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # run inference
    generator = None if args.seed is None else np.random.Generator(np.random.PCG64(seed=args.seed))
    images = [pipeline(**pipeline_args, generator=generator)[0][0] for _ in range(args.num_validation_images)]

    phase_name = "test" if is_final_validation else "validation"
    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, phase_name, f"epoch{epoch}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx:04d}.jpg"))

    for tracker_name, tracker_writer in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker_writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker_name}")

    logger.info("Validation done.")

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from mindone.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from mindone.transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--token_abstraction",
        type=str,
        default="TOK",
        help="identifier specifying the instance(or instances) as used in instance_prompt, validation prompt, "
        "captions - e.g. TOK. To use multiple identifiers, please specify them in a comma separated string - e.g. "
        "'TOK,TOK2,TOK3' etc.",
    )

    parser.add_argument(
        "--num_new_tokens_per_abstraction",
        type=int,
        default=None,
        help="number of new tokens inserted to the tokenizers per token_abstraction identifier when "
        "--train_text_encoder_ti = True. By default, each --token_abstraction (e.g. TOK) is mapped to 2 new "
        "tokens - <si><si+1> ",
    )
    parser.add_argument(
        "--initializer_concept",
        type=str,
        default=None,
        help="the concept to use to initialize the new inserted tokens when training with "
        "--train_text_encoder_ti = True. By default, new tokens (<si><si+1>) are initialized with random value. "
        "Alternatively, you could specify a different word/words whos value will be used as the starting point for the new inserted tokens. "
        "--num_new_tokens_per_abstraction is ignored when initializer_concept is provided",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_text_encoder_ti",
        action="store_true",
        help=("Whether to use pivotal tuning / textual inversion"),
    )
    parser.add_argument(
        "--enable_t5_ti",
        action="store_true",
        help=(
            "Whether to use pivotal tuning / textual inversion for the T5 encoder as well (in addition to CLIP encoder)"
        ),
    )

    parser.add_argument(
        "--train_text_encoder_ti_frac",
        type=float,
        default=0.5,
        help=("The percentage of epochs to perform textual inversion"),
    )

    parser.add_argument(
        "--train_text_encoder_frac",
        type=float,
        default=1.0,
        help=("The percentage of epochs to perform text encoder tuning"),
    )
    parser.add_argument(
        "--train_transformer_frac",
        type=float,
        default=1.0,
        help=("The percentage of epochs to perform transformer tuning"),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"], "prodigy" not yet implemented'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for transformer params"
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            "The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. "
            'E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only. For more examples refer to https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md'
        ),
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    # check if needed
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    if args.train_text_encoder and args.train_text_encoder_ti:
        raise ValueError(
            "Specify only one of `--train_text_encoder` or `--train_text_encoder_ti. "
            "For full LoRA text encoder training check --train_text_encoder, for textual "
            "inversion training check `--train_text_encoder_ti`"
        )
    if args.train_transformer_frac < 1 and not args.train_text_encoder_ti:
        raise ValueError(
            "--train_transformer_frac must be == 1 if text_encoder training / textual inversion is not enabled."
        )
    if args.train_transformer_frac < 1 and args.train_text_encoder_ti_frac < 1:
        raise ValueError(
            "--train_transformer_frac and --train_text_encoder_ti_frac are identical and smaller than 1. "
            "This contradicts with --max_train_steps, please specify different values or set both to 1."
        )
    if args.enable_t5_ti and not args.train_text_encoder_ti:
        logger.warning("You need not use --enable_t5_ti without --train_text_encoder_ti.")

    if args.train_text_encoder_ti and args.initializer_concept and args.num_new_tokens_per_abstraction:
        logger.warning(
            "When specifying --initializer_concept, the number of tokens per abstraction is detrimned "
            "by the initializer token. --num_new_tokens_per_abstraction will be ignored"
        )

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    # Limitations for NOW.
    def error_template(feature, flag):
        return f"{feature} is not yet supported, please do not set --{flag}"

    assert args.allow_tf32 is False, error_template("TF32 Data Type", "allow_tf32")
    assert args.optimizer == "AdamW", error_template("Optimizer besides AdamW", "optimizer")
    assert args.use_8bit_adam is False, error_template("AdamW8bit", "use_8bit_adam")
    if args.push_to_hub is True:
        raise ValueError(
            "You cannot use --push_to_hub due to a security risk of uploading your data to huggingface-hub. "
            "If you know what you are doing, just delete this line and try again."
        )

    return args



# Modified from https://github.com/replicate/cog-sdxl/blob/main/dataset_and_utils.py
class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[ms.Tensor] = None
        self.train_ids_t5: Optional[ms.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_toks
            ), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Convert the token abstractions to ids
            if idx == 0:
                self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
            else:
                self.train_ids_t5 = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            std_token_embedding = embeds.embedding_table.data.std()

            logger.info(f"{idx} text encoder's std_token_embedding: {std_token_embedding}")

            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            # if initializer_concept are not provided, token embeddings are initialized randomly
            if args.initializer_concept is None:
                hidden_size = (
                    text_encoder.text_model.config.hidden_size if idx == 0 else text_encoder.encoder.config.hidden_size
                )
                embeds.weight.data[train_ids] = (
                    ops.randn((len(train_ids), hidden_size), dtype=self.dtype)
                    * std_token_embedding
                )
            else:
                # Convert the initializer_token, placeholder_token to ids
                initializer_token_ids = tokenizer.encode(args.initializer_concept, add_special_tokens=False)
                for token_idx, token_id in enumerate(train_ids):
                    embeds.embedding_table.data[token_id] = (embeds.embedding_table.data)[
                        initializer_token_ids[token_idx % len(initializer_token_ids)]
                    ].copy()

            self.embeddings_settings[f"original_embeddings_{idx}"] = embeds.embedding_table.data.copy()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            # makes sure we don't update any embedding weights besides the newly added token
            index_no_updates = ms.mint.ones((len(tokenizer),), dtype=ms.bool)
            index_no_updates[train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = index_no_updates

            logger.info(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_one, idx==0 - CLIP ViT-L/14, text_encoder_two, idx==1 - T5 xxl
        idx_to_text_encoder_name = {0: "clip_l", 1: "t5"}
        for idx, text_encoder in enumerate(self.text_encoders):
            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            assert embeds.embedding_table.data.shape[0] == len(self.tokenizers[idx]), "Tokenizers should be the same."
            new_token_embeddings = embeds.embedding_table.data[train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0),
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    # @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            embeds.embedding_table.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = embeds.embedding_table.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            embeds.embedding_table.data[index_updates] = new_embeddings


class DreamBoothDataset(object):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        args,
        instance_data_root,
        instance_prompt,
        tokenizer_one,
        tokenizer_two,
        class_prompt,
        train_text_encoder_ti,
        token_abstraction_dict=None,  # token mapping for textual inversion
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        add_special_tokens_clip=False,
        add_special_tokens_t5=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.add_special_tokens_clip = add_special_tokens_clip
        self.add_special_tokens_t5 = add_special_tokens_t5

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        self.max_sequence_length = args.max_sequence_length

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = vision.Resize(size, interpolation=vision.Inter.BILINEAR)
        train_crop = vision.CenterCrop(size) if center_crop else vision.RandomCrop(size)
        train_flip = vision.RandomHorizontalFlip(prob=1.0)
        train_transforms = transforms.Compose(
            [
                vision.ToTensor(),
                vision.Normalize([0.5], [0.5], is_hwc=False),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                h, w = image.height, image.width
                th, tw = args.resolution, args.resolution
                if h < th or w < tw:
                    raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
                y1 = np.random.randint(0, h - th + 1, size=(1,)).item()
                x1 = np.random.randint(0, w - tw + 1, size=(1,)).item()
                image = image.crop((x1, y1, x1 + tw, y1 + th))
            image = train_transforms(image)[0]
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                vision.Resize(size, interpolation=vision.Inter.BILINEAR),
                vision.CenterCrop(size) if center_crop else vision.RandomCrop(size),
                vision.ToTensor(),
                vision.Normalize([0.5], [0.5], is_hwc=False),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # replace instances of --token_abstraction in caption with the new tokens: "<si><si+1>" etc.
                    for token_abs, token_replacement in self.token_abstraction_dict.items():
                        caption = caption.replace(token_abs, "".join(token_replacement))
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # the given instance prompt is used for all images
            example["instance_prompt"] = self.instance_prompt

        example["instance_tokens_one"] = tokenize_prompt(
            self.tokenizer_one, example["instance_prompt"], max_sequence_length=77, add_special_tokens=self.add_special_tokens_clip
        )
        example["instance_tokens_two"] = tokenize_prompt(
            self.tokenizer_two, example["instance_prompt"], max_sequence_length=self.max_sequence_length, add_special_tokens=self.add_special_tokens_t5
        )

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)[0]
            example["class_prompt"] = self.class_prompt
            example["class_tokens_one"] = tokenize_prompt(
                self.tokenizer_one, example["class_prompt"], max_sequence_length=77
            )
            example["class_tokens_two"] = tokenize_prompt(
                self.tokenizer_two, example["class_prompt"], max_sequence_length=self.max_sequence_length
            )

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    tokens_one = [example["instance_tokens_one"] for example in examples]
    tokens_two = [example["instance_tokens_two"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        tokens_one += [example["class_tokens_one"] for example in examples]
        tokens_two += [example["class_tokens_two"] for example in examples]

    pixel_values = np.stack(pixel_values).astype(np.float32)
    tokens_one = np.concatenate(tokens_one, axis=0)
    tokens_two = np.concatenate(tokens_two, axis=0)

    return pixel_values, tokens_one, tokens_two


class PromptDataset(object):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, max_sequence_length, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        add_special_tokens=add_special_tokens,
        return_tensors="np",
    )
    text_input_ids = text_inputs.input_ids

    if text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    return text_input_ids


def _get_t5_prompt_embeds(
    text_encoder,
    num_images_per_prompt=1,
    text_input_ids=None,
):
    batch_size = text_input_ids.shape[0]
    # prompt = [prompt] if isinstance(prompt, str) else prompt
    # batch_size = len(prompt)

    # if tokenizer is not None:
    #     text_inputs = tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=max_sequence_length,
    #         truncation=True,
    #         return_length=False,
    #         return_overflowing_tokens=False,
    #         return_tensors="pt",
    #     )
    #     text_input_ids = text_inputs.input_ids
    # else:
    #     if text_input_ids is None:
    #         raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _get_clip_prompt_embeds(
    text_encoder,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    batch_size = text_input_ids.shape[0]

    # prompt = [prompt] if isinstance(prompt, str) else prompt
    # batch_size = len(prompt)

    # if tokenizer is not None:
    #     text_inputs = tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=77,
    #         truncation=True,
    #         return_overflowing_tokens=False,
    #         return_length=False,
    #         return_tensors="pt",
    #     )

    #     text_input_ids = text_inputs.input_ids
    # else:
    #     if text_input_ids is None:
    #         raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    # prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds[1]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoder_one,
    text_encoder_two,
    text_input_ids_one,
    text_input_ids_two,
    num_images_per_prompt: int = 1,
):
    dtype = text_encoder_one.dtype

    pooled_prompt_embeds = _get_clip_prompt_embeds(
        text_encoder=text_encoder_one,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_one,
    )

    prompt_embeds = _get_t5_prompt_embeds(
        text_encoder=text_encoder_two,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_two,
    )

    text_ids = ms.mint.zeros((prompt_embeds.shape[1], 3)).to(dtype=dtype)
    text_ids = text_ids.tile((num_images_per_prompt, 1, 1))

    return prompt_embeds, pooled_prompt_embeds, text_ids


# CustomFlowMatchEulerDiscreteScheduler was taken from ostris ai-toolkit trainer:
# https://github.com/ostris/ai-toolkit/blob/9ee1ef2a0a2a9a02b92d114a95f21312e5906e54/toolkit/samplers/custom_flowmatch_sampler.py#L95
class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create weights for timesteps
        num_timesteps = 1000

        # generate the multiplier based on cosmap loss weighing
        # this is only used on linear timesteps for now

        # cosine map weighing is higher in the middle and lower at the ends
        # bot = 1 - 2 * self.sigmas + 2 * self.sigmas ** 2
        # cosmap_weighing = 2 / (math.pi * bot)

        # sigma sqrt weighing is significantly higher at the end and lower at the beginning
        sigma_sqrt_weighing = (self.sigmas**-2.0).float()
        # clip at 1e4 (1e6 is too high)
        sigma_sqrt_weighing = ms.mint.clamp(sigma_sqrt_weighing, max=1e4)
        # bring to a mean of 1
        sigma_sqrt_weighing = sigma_sqrt_weighing / sigma_sqrt_weighing.mean()

        # Create linear timesteps from 1000 to 0
        timesteps = np.linspace(1000, 0, num_timesteps, dtype=np.float32)
        timesteps = ms.Tensor.from_numpy(timesteps).to(dtype=ms.float32)

        self.linear_timesteps = timesteps
        # self.linear_timesteps_weights = cosmap_weighing
        self.linear_timesteps_weights = sigma_sqrt_weighing

        # self.sigmas = self.get_sigmas(timesteps, n_dim=1, dtype=ms.float32)
        pass

    def get_weights_for_timesteps(self, timesteps: ms.Tensor) -> ms.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: ms.Tensor, n_dim, dtype) -> ms.Tensor:
        sigmas = self.sigmas.to(dtype=dtype)
        schedule_timesteps = self.timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
        self,
        original_samples: ms.Tensor,
        noise: ms.Tensor,
        timesteps: ms.Tensor,
    ) -> ms.Tensor:
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise

        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input

    def scale_model_input(self, sample: ms.Tensor, timestep: Union[float, ms.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, num_timesteps, device, linear=False):
        if linear:
            timesteps = np.linspace(1000, 0, num_timesteps)
            self.timesteps = timesteps
            return timesteps
        else:
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = ops.sigmoid(ops.randn((num_timesteps,)))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = (1 - t) * 1000

            # Sort the timesteps in descending order
            timesteps, _ = ops.sort(timesteps, descending=True)

            self.timesteps = timesteps

            return timesteps


def main(args):
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    init_distributed_device(args)

    logging_dir = Path(args.output_dir, args.logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            mindspore_dtype = ms.float32
            if args.prior_generation_precision == "fp32":
                mindspore_dtype = ms.float32
            elif args.prior_generation_precision == "fp16":
                mindspore_dtype = ms.float16
            elif args.prior_generation_precision == "bf16":
                mindspore_dtype = ms.bfloat16
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                mindspore_dtype=mindspore_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = GeneratorDataset(
                sample_dataset, sample_dataset, column_names=["example"], shard_id=args.rank, num_shards=args.world_size
            ).batch(batch_size=args.sample_batch_size)

            sample_dataloader_iter = sample_dataloader.create_tuple_iterator(output_numpy=True)

            for (example,) in tqdm(
                sample_dataloader_iter,
                desc="Generating class images",
                total=len(sample_dataloader),
                disable=not is_master(args),
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            gc.collect()
            ms.ms_memory_recycle()
            logger.warning(
                "After deleting the pipeline, the memory may not be freed correctly by mindspore. "
                "If you encounter an OOM error, please relaunch this script."
            )

    # Handle the repository creation
    if is_master(args):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder_ti:
        # we parse the provided token identifier (or identifiers) into a list. s.t. - "TOK" -> ["TOK"], "TOK,
        # TOK2" -> ["TOK", "TOK2"] etc.
        token_abstraction_list = [place_holder.strip() for place_holder in re.split(r",\s*", args.token_abstraction)]
        logger.info(f"list of token identifiers: {token_abstraction_list}")

        if args.initializer_concept is None:
            num_new_tokens_per_abstraction = (
                2 if args.num_new_tokens_per_abstraction is None else args.num_new_tokens_per_abstraction
            )
        # if args.initializer_concept is provided, we ignore args.num_new_tokens_per_abstraction
        else:
            token_ids = tokenizer_one.encode(args.initializer_concept, add_special_tokens=False)
            num_new_tokens_per_abstraction = len(token_ids)
            if args.enable_t5_ti:
                token_ids_t5 = tokenizer_two.encode(args.initializer_concept, add_special_tokens=False)
                num_new_tokens_per_abstraction = max(len(token_ids), len(token_ids_t5))
            logger.info(
                f"initializer_concept: {args.initializer_concept}, num_new_tokens_per_abstraction: {num_new_tokens_per_abstraction}"
            )

        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(num_new_tokens_per_abstraction)]
            token_idx += num_new_tokens_per_abstraction - 1

        # replace instances of --token_abstraction in --instance_prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in token_abstraction_dict.items():
            new_instance_prompt = args.instance_prompt.replace(token_abs, "".join(token_replacement))
            if args.instance_prompt == new_instance_prompt:
                logger.warning(
                    "Note! the instance prompt provided in --instance_prompt does not include the token abstraction specified "
                    "--token_abstraction. This may lead to incorrect optimization of text embeddings during pivotal tuning"
                )
            args.instance_prompt = new_instance_prompt
            if args.with_prior_preservation:
                args.class_prompt = args.class_prompt.replace(token_abs, "".join(token_replacement))
            if args.validation_prompt:
                args.validation_prompt = args.validation_prompt.replace(token_abs, "".join(token_replacement))

        # initialize the new tokens for textual inversion
        text_encoders = [text_encoder_one, text_encoder_two] if args.enable_t5_ti else [text_encoder_one]
        tokenizers = [tokenizer_one, tokenizer_two] if args.enable_t5_ti else [tokenizer_one]
        embedding_handler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)

    # We only train the additional adapter LoRA layers
    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.requires_grad = False

    freeze_params(transformer)
    freeze_params(vae)
    freeze_params(text_encoder_one)
    freeze_params(text_encoder_two)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    vae.to(dtype=weight_dtype)
    transformer.to(dtype=weight_dtype)
    text_encoder_one.to(dtype=weight_dtype)
    text_encoder_two.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        # if args.train_text_encoder:
        #     text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, output_dir):
        if is_master(args):
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(transformer)):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(text_encoder_one)):
                    if args.train_text_encoder:  # when --train_text_encoder_ti we don't save the layers
                        text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model, save_embedding_layers=False)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )
        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(f"{args.output_dir}/{Path(args.output_dir).name}_emb.safetensors")

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(transformer)):
                transformer_ = model
            elif isinstance(model, type(text_encoder_one)):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    models = [transformer]
    if args.train_text_encoder:
        models.extend([text_encoder_one])

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=ms.float32)

    # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    add_special_tokens_clip = True if args.train_text_encoder_ti else False
    add_special_tokens_t5 = True if (args.train_text_encoder_ti and args.enable_t5_ti) else False

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        args,
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        train_text_encoder_ti=args.train_text_encoder_ti,
        token_abstraction_dict=token_abstraction_dict if args.train_text_encoder_ti else None,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        add_special_tokens_clip=add_special_tokens_clip,
        add_special_tokens_t5=add_special_tokens_t5,
    )

    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names=["example"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        per_batch_map=lambda examples, batch_info: collate_fn(examples, args.with_prior_preservation),
        input_columns=["example"],
        output_columns=["c1", "c2", "c3"],  # pixel_values, tokens_one, tokens_two
        num_parallel_workers=args.dataloader_num_workers,
    )

    if not train_dataset.custom_instance_prompts:
        tokens_one = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_one, args.instance_prompt, 77, add_special_tokens=add_special_tokens_clip))
        tokens_two = ms.Tensor.from_numpy(
            tokenize_prompt(tokenizer_two, args.instance_prompt, args.max_sequence_length, add_special_tokens=add_special_tokens_t5)
        )

        if args.with_prior_preservation:
            class_tokens_one = ms.Tensor.from_numpy(tokenize_prompt(tokenizer_one, args.class_prompt, 77, add_special_tokens=add_special_tokens_clip))
            class_tokens_two = ms.Tensor.from_numpy(
                tokenize_prompt(tokenizer_two, args.class_prompt, args.max_sequence_length, add_special_tokens=add_special_tokens_t5)
            )

            tokens_one = ops.cat([tokens_one, class_tokens_one], axis=0)
            tokens_two = ops.cat([tokens_two, class_tokens_two], axis=0)

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
        # the redundant encoding.

        if not args.train_text_encoder:
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoder_one,
                text_encoder_two,
                tokens_one,
                tokens_two,
            )
        else:
            prompt_embeds, pooled_prompt_embeds, text_ids = None, None, None

    else:
        tokens_one, tokens_two = None, None
        prompt_embeds, pooled_prompt_embeds, text_ids = None, None, None

    # if freeze_text_encoder:
    #     tokenizers = [tokenizer_one, tokenizer_two]
    #     text_encoders = [text_encoder_one, text_encoder_two]

    #     def compute_text_embeddings(prompt, text_encoders, tokenizers):
    #         with torch.no_grad():
    #             prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
    #                 text_encoders, tokenizers, prompt, args.max_sequence_length
    #             )
    #             prompt_embeds = prompt_embeds.to(accelerator.device)
    #             pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
    #             text_ids = text_ids.to(accelerator.device)
    #         return prompt_embeds, pooled_prompt_embeds, text_ids

    # # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # # the redundant encoding.
    # if freeze_text_encoder and not train_dataset.custom_instance_prompts:
    #     instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
    #         args.instance_prompt, text_encoders, tokenizers
    #     )

    # # Handle class prompt for prior-preservation.
    # if args.with_prior_preservation:
    #     if freeze_text_encoder:
    #         class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
    #             args.class_prompt, text_encoders, tokenizers
    #         )

    # # Clear the memory here
    # if freeze_text_encoder and not train_dataset.custom_instance_prompts:
    #     del tokenizers, text_encoders, text_encoder_one, text_encoder_two
    #     free_memory()

    # # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    # add_special_tokens_clip = True if args.train_text_encoder_ti else False
    # add_special_tokens_t5 = True if (args.train_text_encoder_ti and args.enable_t5_ti) else False

    # # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # # pack the statically computed variables appropriately here. This is so that we don't
    # # have to pass them to the dataloader.

    # if not train_dataset.custom_instance_prompts:
    #     if freeze_text_encoder:
    #         prompt_embeds = instance_prompt_hidden_states
    #         pooled_prompt_embeds = instance_pooled_prompt_embeds
    #         text_ids = instance_text_ids
    #         if args.with_prior_preservation:
    #             prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
    #             pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
    #             text_ids = torch.cat([text_ids, class_text_ids], dim=0)
    #     # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
    #     # we need to tokenize and encode the batch prompts on all training steps
    #     else:
    #         tokens_one = tokenize_prompt(
    #             tokenizer_one, args.instance_prompt, max_sequence_length=77, add_special_tokens=add_special_tokens_clip
    #         )
    #         tokens_two = tokenize_prompt(
    #             tokenizer_two,
    #             args.instance_prompt,
    #             max_sequence_length=args.max_sequence_length,
    #             add_special_tokens=add_special_tokens_t5,
    #         )
    #         if args.with_prior_preservation:
    #             class_tokens_one = tokenize_prompt(
    #                 tokenizer_one,
    #                 args.class_prompt,
    #                 max_sequence_length=77,
    #                 add_special_tokens=add_special_tokens_clip,
    #             )
    #             class_tokens_two = tokenize_prompt(
    #                 tokenizer_two,
    #                 args.class_prompt,
    #                 max_sequence_length=args.max_sequence_length,
    #                 add_special_tokens=add_special_tokens_t5,
    #             )
    #             tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
    #             tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    # vae_config_shift_factor = vae.config.shift_factor
    # vae_config_scaling_factor = vae.config.scaling_factor
    # vae_config_block_out_channels = vae.config.block_out_channels

    # TODO cache_latents

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.get_parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.get_parameters()))
    # if we use textual inversion, we freeze all parameters except for the token embeddings
    # in text encoder
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []  # CLIP
        for name, param in text_encoder_one.parameters_and_names():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                param.set_dtype(ms.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        if args.enable_t5_ti:  # whether to do pivotal tuning/textual inversion for T5 as well
            text_lora_parameters_two = []
            for name, param in text_encoder_two.parameters_and_names():
                if "token_embedding" in name:
                    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                    param.set_dtype(ms.float32)
                    param.requires_grad = True
                    text_lora_parameters_two.append(param)
                else:
                    param.requires_grad = False

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # if --train_text_encoder_ti and train_transformer_frac == 0 where essentially performing textual inversion
    # and not training transformer LoRA layers
    pure_textual_inversion = args.train_text_encoder_ti and args.train_transformer_frac == 0

    # TODO the `te_idx`s marked here are for control te training frac, 
    # to set the lr in certain params groups to be 0 in torch.
    # the frac setting to be will support later.

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": lr_scheduler}
    if not freeze_text_encoder:
        text_encoder_lr_scheduler = [i * args.text_encoder_lr / args.learning_rate for i in lr_scheduler] if args.text_encoder_lr else lr_scheduler
        
        # different learning rate for text encoder and transformer
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": text_encoder_lr_scheduler,
        }
        if not args.enable_t5_ti:
            # pure textual inversion - only clip
            if pure_textual_inversion:
                params_to_optimize = [
                    text_parameters_one_with_lr,
                ]
                te_idx = 0
            else:  # regular te training or regular pivotal for clip
                params_to_optimize = [
                    transformer_parameters_with_lr,
                    text_parameters_one_with_lr,
                ]
                te_idx = 1
        elif args.enable_t5_ti:
            # pivotal tuning of clip & t5
            text_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
                "weight_decay": args.adam_weight_decay_text_encoder
                if args.adam_weight_decay_text_encoder
                else args.adam_weight_decay,
                "lr": text_encoder_lr_scheduler,
            }
            # pure textual inversion - only clip & t5
            if pure_textual_inversion:
                params_to_optimize = [text_parameters_one_with_lr, text_parameters_two_with_lr]
                te_idx = 0
            else:  # regular pivotal tuning of clip & t5
                params_to_optimize = [
                    transformer_parameters_with_lr,
                    text_parameters_one_with_lr,
                    text_parameters_two_with_lr,
                ]
                te_idx = 1
    else:
        params_to_optimize = transformer_lora_parameters

    # Optimizer creation
    optimizer = nn.AdamWeightDecay(
        params_to_optimize,
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    for peft_model in models:
        for _, module in peft_model.cells_and_names():
            if isinstance(module, BaseTunerLayer):
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in module.active_adapters and isinstance(layer, nn.Cell):
                            layer.to_float(weight_dtype)
    if args.train_text_encoder_ti:
        text_encoder_one.get_input_embeddings().to_float(weight_dtype)
    if args.enable_t5_ti:
        text_encoder_two.get_input_embeddings().to_float(weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_master(args):
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from checkpoint {path}")
            # TODO: load optimizer & grad scaler etc. like accelerator.load_state
            load_model_hook(models, os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # create train_step for training
    train_step = TrainStepForFluxDevAdvanced(
        vae=vae,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        transformer=transformer,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler_copy,
        weight_dtype=weight_dtype,
        length_of_dataloader=len(train_dataloader),
        args=args,
        tokens_one=tokens_one,
        tokens_two=tokens_two,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        custom_instance_prompts=train_dataset.custom_instance_prompts,
        freeze_text_encoder = freeze_text_encoder,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
    else:
        sink_process = None

    # create pipeline for validation
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        mindspore_dtype=weight_dtype,
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    # def get_sigmas(timesteps, n_dim=4, dtype=ms.float32):
    #     sigmas = noise_scheduler_copy.sigmas.to(dtype=dtype)
    #     schedule_timesteps = noise_scheduler_copy.timesteps
    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    # TODO control the frac
    # if args.train_text_encoder:
    #     num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
    #     num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)
    # elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
    #     num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)
    #     num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)

    # # flag used for textual inversion
    # pivoted_te = False
    # pivoted_tr = False
    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - first_epoch)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.set_train(True)
        if args.train_text_encoder:
            text_encoder_one.set_train()
        elif args.train_text_encoder_ti:  # textual inversion / pivotal tuning
            text_encoder_one.set_train()
        if args.enable_t5_ti:
            text_encoder_two.set_train()

        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader_iter)
        ):
            if args.enable_mindspore_data_sink:
                loss, model_pred = sink_process()
            else:
                loss, model_pred = train_step(*batch)


            # every step, we reset the embeddings to the original embeddings.
            if args.train_text_encoder_ti:
                embedding_handler.retract_embeddings()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if is_master(args):
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # TODO: save optimizer & grad scaler etc. like accelerator.save_state
                        os.makedirs(save_path, exist_ok=True)
                        save_model_hook(models, save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalars("train", logs, global_step)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                trackers=trackers,
                logging_dir=logging_dir,
                pipeline_args=pipeline_args,
                epoch=epoch + 1,
            )

    # Save the lora layers
    if is_master(args):
        transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        if args.train_text_encoder:
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(ms.float32))
        else:
            text_encoder_lora_layers = None

        if not pure_textual_inversion:
            FluxPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
            )

        if args.train_text_encoder_ti:
            embeddings_path = f"{args.output_dir}/{os.path.basename(args.output_dir)}_emb.safetensors"
            embedding_handler.save_embeddings(embeddings_path)

    # Final inference
    if args.validation_prompt and args.num_validation_images > 0:
        pipeline_args = {"prompt": args.validation_prompt, "num_inference_steps": 25}
        log_validation(
            pipeline,
            args,
            trackers,
            logging_dir,
            pipeline_args,
            args.num_train_epochs,
            is_final_validation=True,
        )

    # End of training
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


def compute_weighting_mse_loss(weighting, pred, target):
    """
    When argument with_prior_preservation is True in DreamBooth training, weighting has different batch_size
    with pred/target which causes errors, therefore we broadcast them to proper shape before mul
    """
    repeats = weighting.shape[0] // pred.shape[0]
    target_ndim = target.ndim
    square_loss = ((pred.float() - target.float()) ** 2).tile((repeats,) + (1,) * (target_ndim - 1))

    weighting_mse_loss = ops.mean(
        (weighting * square_loss).reshape(target.shape[0], -1),
        1,
    )
    weighting_mse_loss = weighting_mse_loss.mean()

    return weighting_mse_loss


class TrainStepForFluxDevAdvanced(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoder_one: nn.Cell,
        text_encoder_two: nn.Cell,
        transformer: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,
        tokens_one,
        tokens_two,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        custom_instance_prompts,
        freeze_text_encoder,
    ):
        super().__init__(
            transformer,
            optimizer,
            StaticLossScaler(4096),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.transformer = transformer
        self.transformer_config_guidance_embeds = transformer.config.guidance_embeds

        self.vae = vae
        self.vae_dtype = vae.dtype
        self.vae_config_scaling_factor = vae.config.scaling_factor
        self.vae_config_shift_factor = vae.config.shift_factor
        self.vae_config_block_out_channels = vae.config.block_out_channels
        self.vae_scale_factor = 2 ** (len(self.vae_config_block_out_channels))

        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.text_encoder_dtype = text_encoder_one.dtype
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))
        self.custom_instance_prompts = custom_instance_prompts

        self.freeze_text_encoder = freeze_text_encoder

        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps, and the following would be None here.
        self.tokens_one = tokens_one
        self.tokens_two = tokens_two
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids

    def get_sigmas(self, indices, n_dim=4, dtype=ms.float32):
        """
        origin `get_sigmas` which uses timesteps to get sigmas might be not supported
        in mindspore Graph mode, thus we rewrite `get_sigmas` to get sigma directly
        from indices which calls less ops and could run in mindspore Graph mode.
        """
        sigma = self.noise_scheduler.sigmas[indices].to(dtype=dtype)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, pixel_values, tokens_one, tokens_two):
        # encode batch prompts when custom prompts are provided for each image
        if self.custom_instance_prompts:
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoder_one=self.text_encoder_one,
                text_encoder_two=self.text_encoder_two,
                text_input_ids_one=tokens_one,  # from batch input
                text_input_ids_two=tokens_two,  # from batch input
            )
        else:
            if not self.freeze_text_encoder:
                # use pre-computed tokens.
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoder_one=self.text_encoder_one,
                    text_encoder_two=self.text_encoder_two,
                    text_input_ids_one=self.tokens_one,  # from pre-compute tokens
                    text_input_ids_two=self.tokens_two,  # from pre-compute tokens
                )
            else:
                # use pre-computed embeddings.
                prompt_embeds, pooled_prompt_embeds, text_ids = (
                    self.prompt_embeds,
                    self.pooled_prompt_embeds,
                    self.text_ids,
                )

        # Convert images to latent space
        pixel_values = pixel_values.to(dtype=self.vae_dtype)

        model_input = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values)[0])
        model_input = (model_input - self.vae_config_shift_factor) * self.vae_config_scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            self.weight_dtype,
        )
        # Sample noise that we'll add to the latents
        noise = ops.randn_like(model_input, dtype=model_input.dtype)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.args.logit_mean,
            logit_std=self.args.logit_std,
            mode_scale=self.args.mode_scale,
        )
        indices = (u * self.noise_scheduler_num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(indices, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

        # handle guidance
        if self.transformer_config_guidance_embeds:
            guidance = ms.tensor([self.args.guidance_scale])
            guidance = guidance.broadcast_to(model_input.shape[0])
        else:
            guidance = None

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * self.vae_scale_factor,
            width=model_input.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        if self.args.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = ops.chunk(model_pred, 2, axis=0)
            target, target_prior = ops.chunk(target, 2, axis=0)

            # Compute prior loss
            prior_loss = compute_weighting_mse_loss(weighting, model_pred_prior, target_prior)

        # Compute regular loss.
        loss = compute_weighting_mse_loss(weighting, model_pred, target)

        if args.with_prior_preservation:
            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss

        loss = self.scale_loss(loss)
        return loss, model_pred


if __name__ == "__main__":
    args = parse_args()
    main(args)
