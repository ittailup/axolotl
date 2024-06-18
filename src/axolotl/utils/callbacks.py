"""Callbacks for Trainer class"""

import logging
import os

from optimum.bettertransformer import BetterTransformer
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, IntervalStrategy

from axolotl.utils.bench import log_gpu_memory_usage

LOG = logging.getLogger("axolotl.callbacks")


class SavePeftModelCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """Callback to save the PEFT adapter"""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        return control


class SaveBetterTransformerModelCallback(
    TrainerCallback
):  # pylint: disable=too-few-public-methods
    """Callback to save the BetterTransformer wrapped model"""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        if control.should_save:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
            )

            model = BetterTransformer.reverse(kwargs["model"])
            model.save_pretrained(checkpoint_folder)
            # FIXME - need to cleanup old checkpoints

            # since we're saving here, we don't need the trainer loop to attempt to save too b/c
            # the trainer will raise an exception since it can't save a BetterTransformer wrapped model
            control.should_save = False
        return control


class GPUStatsCallback(
    TrainerCallback
):  # pylint: disable=too-few-public-methods disable=unused-argument
    """Callback to track GPU utilization"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logged = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.logged and state.global_step > 1:
            log_gpu_memory_usage(LOG, "while training", self.cfg.device)
            self.logged = True
        return control

class UploadToS3Callback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.s3_client = boto3.client('s3')

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.should_save_checkpoint(args, state):
            self.upload_checkpoint(args.output_dir, state.global_step)
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.should_save_checkpoint(args, state):
            self.upload_checkpoint(args.output_dir, state.global_step)
        return control

    def upload_checkpoint(self, output_dir, global_step):
        checkpoint_folder = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        s3_url = self.cfg.remote_output_dir
        bucket_name, prefix = self.parse_s3_url(s3_url)

        for root, _, files in os.walk(checkpoint_folder):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = os.path.join(prefix, os.path.relpath(local_path, output_dir))
                self.s3_client.upload_file(local_path, bucket_name, s3_key)

    @staticmethod
    def parse_s3_url(s3_url):
        assert s3_url.startswith("s3://")
        s3_url = s3_url[len("s3://"):]
        bucket_name, prefix = s3_url.split("/", 1)
        return bucket_name, prefix

    @staticmethod
    def should_save_checkpoint(args: TrainingArguments, state: TrainerState) -> bool:
        return (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        )

