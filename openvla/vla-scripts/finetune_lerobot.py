"""
finetune_lerobot.py

Variant of `finetune.py` that consumes a local LeRobot v3.0 dataset instead of an RLDS/TFDS
dataset. Mirrors the original training loop (LoRA + DDP + checkpoint saving + W&B) but swaps
the dataset for `LeRobotDatasetForOpenVLA` (map-style) and uses a standard PyTorch
DistributedSampler so that `num_workers > 0` and shuffling work.

Run (single GPU):
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_lerobot.py \\
        --vla_path "openvla/openvla-7b" \\
        --lerobot_root /home/cvlab/project/realsangbeom/robot/lerobot \\
        --dataset_name scannet_panda \\
        --run_root_dir runs \\
        --adapter_tmp_dir adapter-tmp \\
        --batch_size 16 --learning_rate 5e-4 --max_steps 50000

Notes:
    - LeRobot v3.0 video decoding via PyAV is CPU-bound; set `--num_workers` >= 4 to keep the
      GPU fed. Increase further if action stats sampling is a one-shot bottleneck.
    - Set `--stats_max_frames` smaller for quick smoke tests; set to `None` to compute
      quantiles over the full dataset.
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import LeRobotConfig, LeRobotDatasetForOpenVLA
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # Dataset paths
    lerobot_root: Path = Path("/home/cvlab/project/realsangbeom/robot/lerobot")
    dataset_name: str = "lerobot_local"
    image_video_key: str = "observation.images.image"   # primary camera; for jisang dataset use "observation.images.external_image" / "observation.images.wrist_image"
    action_column: str = "action"
    task_column: str = "task_index"
    stats_max_frames: Optional[int] = 50_000            # set to None to use every frame
    # If the source video is large (e.g. 1280x720 in the jisang dataset), resize at decode
    # time to keep the dataloader CPU-bound work small. (None disables.)
    image_resize_h: Optional[int] = None
    image_resize_w: Optional[int] = None
    # If the dataset has a single task and no per-frame language annotation worth using,
    # override the prompt explicitly (e.g. "pick and place the object").
    fixed_prompt: Optional[str] = None

    # Output paths
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # Fine-tuning hyperparameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5_000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    num_workers: int = 4
    save_latest_checkpoint_only: bool = True

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    # Logging
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None
    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA `{cfg.vla_path}` on LeRobot dataset at `{cfg.lerobot_root}`")

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # --- experiment id + run dirs -----------------------------------------------------------
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    exp_id += "--lerobot"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # --- model load -------------------------------------------------------------------------
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # --- dataset / dataloader ---------------------------------------------------------------
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    image_resize_hw = None
    if cfg.image_resize_h is not None and cfg.image_resize_w is not None:
        image_resize_hw = (cfg.image_resize_h, cfg.image_resize_w)

    vla_dataset = LeRobotDatasetForOpenVLA(
        config=LeRobotConfig(
            data_root=cfg.lerobot_root,
            dataset_name=cfg.dataset_name,
            image_video_key=cfg.image_video_key,
            action_column=cfg.action_column,
            task_column=cfg.task_column,
            stats_max_frames=cfg.stats_max_frames,
            image_resize_hw=image_resize_hw,
            fixed_prompt=cfg.fixed_prompt,
        ),
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    sampler = DistributedSampler(
        vla_dataset,
        num_replicas=distributed_state.num_processes,
        rank=distributed_state.process_index,
        shuffle=True,
        drop_last=True,
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # --- logging ----------------------------------------------------------------------------
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # --- training loop ----------------------------------------------------------------------
    # Map-style dataset needs explicit epoch iteration; we loop until `max_steps` gradient steps.
    progress = tqdm.tqdm(total=cfg.max_steps, leave=False)
    vla.train()
    optimizer.zero_grad()
    batch_idx = 0
    gradient_step_idx = 0
    epoch = 0
    stop_training = False

    while not stop_training:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Accuracy + L1 on the predicted action tokens
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        "epoch": epoch,
                    },
                    step=gradient_step_idx,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0 and (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            ):
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                dist.barrier()

                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                dist.barrier()

            if gradient_step_idx >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                stop_training = True
                break

            batch_idx += 1

        epoch += 1


if __name__ == "__main__":
    finetune()
