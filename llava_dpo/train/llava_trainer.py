import os
import sys
import itertools
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers import Trainer, get_scheduler
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from transformers.deepspeed import HfDeepSpeedConfig, deepspeed_load_checkpoint

from tqdm import tqdm
from typing import (
    List, Any, Dict,
    Optional, Union,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from llava_dpo.logger import Logger
from llava_dpo.constants import ADAM_BETAS, IMAGE_TOKEN_INDEX, IGNORE_INDEX, ASSISTANT_TOKEN_IDS
from llava_dpo.model.utils import gather_log_probabilities
from llava_dpo.model import LlavaLlamaForCausalLM
from llava_dpo.utils import is_main_process, to_device, get_all_reduce_mean, get_indexes, calculate_log_probs, get_log_probs, sample_random_image


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class DummyDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {}


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


class DPOLLaVATrainer(LLaVATrainer, Trainer):
    
    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine
    
    def __init__(
        self, args,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module],
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
        data_collator: Optional[DataCollator] = None,
        ptx_data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ptx_train_dataset: Optional[Dataset] = None,
        ptx_eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        args.num_train_epochs = int(args.num_train_epochs)
        self.args = args
        self.scale_coeff = self.args.scale_coeff
        self.args.need_eval = eval_dataset is not None
        self.logger = Logger(log_project=self.args.log_project, log_dir=self.args.output_dir)
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=args.per_device_train_batch_size,
        )
        
        self.use_ptx = ptx_train_dataset is not None
        if self.use_ptx:
            self.ptx_train_dataloader = DataLoader(
                ptx_train_dataset,
                collate_fn=ptx_data_collator,
                sampler=DistributedSampler(train_dataset, shuffle=True),
                batch_size=args.per_device_ptx_train_batch_size,
            )
        else:
            self.ptx_train_dataloader = DataLoader(DummyDataset(len(self.train_dataloader)))
        
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.num_train_epochs * self.args.num_update_steps_per_epoch
        if self.use_ptx:
            self.args.gradient_accumulation_steps *= 2
            ds_train_config['train_batch_size'] *= 2
            ds_train_config['gradient_accumulation_steps'] *= 2
        
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)
        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)
        
        self.model = model
        self.create_optimizer_and_scheduler(num_training_steps=self.args.total_training_steps)            
        
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.args,
            config=self.ds_train_config,
            lr_scheduler=self.lr_scheduler,
            dist_init_required=True,
        )
        if self.args.resume_from_ckpt:
            deepspeed_load_checkpoint(self.model, self.args.resume_from_ckpt)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.tokenizer = tokenizer
        
        self.reference_model, *_ = deepspeed.initialize(
            model=ref_model,
            config=ds_eval_config,
        )
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in self.model.named_parameters() if "mm_projector" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        
        if (
            self.ds_train_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            self.optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=ADAM_BETAS,
                # **optimizer_kwargs
            )
        else:
            self.optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=ADAM_BETAS,
            )
        
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        num_warmup_steps = int(self.args.warmup_ratio * self.args.total_training_steps)
        # self.ds_train_config['scheduler']['params']['warmup_num_steps'] = num_warmup_steps
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler
    
    @staticmethod
    def compute_log_probs(
        model: LlavaLlamaForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        images: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask, images=images).logits
        return gather_log_probabilities(logits[:, :-1], labels[:, 1:])
    
    def train_step(
        self,
        better_input_ids: torch.LongTensor,
        better_labels: torch.LongTensor,
        better_attention_mask: torch.BoolTensor,
        worse_input_ids: torch.LongTensor,
        worse_labels: torch.LongTensor,
        worse_attention_mask: torch.BoolTensor,
        images: torch.Tensor,
        better_txt_input_ids: torch.LongTensor = None,
        better_txt_labels: torch.LongTensor = None,
        better_out_input_ids: torch.LongTensor = None,
        better_out_labels: torch.LongTensor = None,
        worse_txt_input_ids: torch.LongTensor = None,
        worse_txt_labels: torch.LongTensor = None,
        worse_out_input_ids: torch.LongTensor = None,
        worse_out_labels: torch.LongTensor = None,
    ) -> dict[str, Any]:
        """Loss function for the DPO algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.
            images (torch.Tensor)

        Returns:
            dict[str, torch.Tensor]: loss, better sample rewards, worse sample rewards
        """
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        
        input_ids = torch.cat([better_input_ids, worse_input_ids], dim=0)
        attention_mask = torch.cat([better_attention_mask, worse_attention_mask], dim=0)
        labels = torch.cat([better_labels, worse_labels], dim=0)
        if images is not None:
            better_images, worse_images = images[:,0,:,:,:], images[:,1,:,:,:]
            images = torch.cat([better_images, worse_images], dim=0)
        
        label_mask = torch.logical_and(labels.ne(IMAGE_TOKEN_INDEX), labels.ne(IGNORE_INDEX))
        labels = (labels * label_mask).long()
        better_label_mask, worse_label_mask = label_mask[:, 1:].chunk(chunks=2, dim=0)
        
        sequence_log_probs = self.compute_log_probs(
            self.model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
        )  # size = (2 * B, L - 1)
        better_log_probs, worse_log_probs = sequence_log_probs.chunk(chunks=2, dim=0)   # size = (B, L - 1)
        
        self.reference_model.eval()
        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(
                self.reference_model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )  # size = (2 * B, L - 1)
            ref_better_log_probs, ref_worse_log_probs = ref_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
        
        randimg_better_log_probs_list, randimg_worse_log_probs_list = [], []
        ref_randimg_better_log_probs_list, ref_randimg_worse_log_probs_list = [], []
        for _ in range(self.args.n_random_images):
            fake_images = torch.stack([
                sample_random_image(images.shape[1:]) for _ in range(images.size(0))
            ], dim=0).to(images.device)
            randimg_sequence_log_probs = self.compute_log_probs(
                self.model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=fake_images,
            )  # size = (2 * B, L - 1)
            randimg_better_log_probs, randimg_worse_log_probs = randimg_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
            randimg_better_log_probs_list.append(randimg_better_log_probs)
            randimg_worse_log_probs_list.append(randimg_worse_log_probs)
            with torch.no_grad():
                ref_randimg_sequence_log_probs = self.compute_log_probs(
                    self.reference_model.module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=fake_images,
                )
                ref_randimg_better_log_probs, ref_randimg_worse_log_probs = ref_randimg_sequence_log_probs.chunk(chunks=2, dim=0)  # size = (B, L - 1)
                ref_randimg_better_log_probs_list.append(ref_randimg_better_log_probs)
                ref_randimg_worse_log_probs_list.append(ref_randimg_worse_log_probs)
        randimg_better_log_probs = torch.stack(randimg_better_log_probs_list, dim=-1).mean(dim=-1)
        randimg_worse_log_probs = torch.stack(randimg_worse_log_probs_list, dim=-1).mean(dim=-1)
        ref_randimg_better_log_probs = torch.stack(ref_randimg_better_log_probs_list, dim=-1).mean(dim=-1)
        ref_randimg_worse_log_probs = torch.stack(ref_randimg_worse_log_probs_list, dim=-1).mean(dim=-1)
        
        losses = []
        better_sample_rewards, worse_sample_rewards = [], []
        better_img_contributions_rand, worse_img_contributions_rand = [], []
        for i in range(batch_size):
            equal_text = torch.all(torch.eq(input_ids[i], worse_input_ids[i])).item()
            equal_img = torch.all(torch.eq(better_images[i], worse_images[i])).item()
            try:
                assert not equal_text or not equal_img, 'The better and worse samples are the same!'
            except:
                import ipdb; ipdb.set_trace()
                
            # better-worse logprobs / ref-logprobs: p(y|x,i)
            ith_better_log_probs = get_log_probs(better_input_ids[i], better_labels[i], better_log_probs[i], is_answer=True)
            ith_worse_log_probs = get_log_probs(worse_input_ids[i], worse_labels[i], worse_log_probs[i], is_answer=True)
            ith_ref_better_log_probs = get_log_probs(better_input_ids[i], better_labels[i], ref_better_log_probs[i], is_answer=True)
            ith_ref_worse_log_probs = get_log_probs(worse_input_ids[i], worse_labels[i], ref_worse_log_probs[i], is_answer=True)
            # better-worse logprobs with random noise: p(y|x,e) -- p(y|x)
            ith_rimg_better_log_probs = get_log_probs(better_input_ids[i], better_labels[i], randimg_better_log_probs[i], is_answer=True)
            ith_rimg_worse_log_probs = get_log_probs(worse_input_ids[i], worse_labels[i], randimg_worse_log_probs[i], is_answer=True)
            ith_ref_rimg_better_log_probs = get_log_probs(better_input_ids[i], better_labels[i], ref_randimg_better_log_probs[i], is_answer=True)
            ith_ref_rimg_worse_log_probs = get_log_probs(worse_input_ids[i], worse_labels[i], ref_randimg_worse_log_probs[i], is_answer=True)
            better_img_contributions_rand.append(ith_better_log_probs - ith_rimg_better_log_probs)
            worse_img_contributions_rand.append(ith_worse_log_probs - ith_rimg_worse_log_probs)
            # better <-> random
            better_log_ratio = ith_better_log_probs - ith_ref_better_log_probs
            rand_better_log_ratio = ith_rimg_better_log_probs - ith_ref_rimg_better_log_probs
            better_rand_logits = better_log_ratio - rand_better_log_ratio
            # random <-> worse
            rand_worse_log_ratio = ith_rimg_worse_log_probs - ith_ref_rimg_worse_log_probs
            worse_log_ratio = ith_worse_log_probs - ith_ref_worse_log_probs
            worse_rand_logits = rand_worse_log_ratio - worse_log_ratio
            
            # better <-> worse
            diverge_index, better_end_index, worse_end_index = get_indexes(
                better_input_ids[i], worse_input_ids[i],
                better_attention_mask[i], worse_attention_mask[i],
            )
            better_seq_slice = slice(diverge_index - 1, better_end_index)
            worse_seq_slice = slice(diverge_index - 1, worse_end_index)
            ith_better_log_probs = calculate_log_probs(better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice])
            ith_worse_log_probs = calculate_log_probs(worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice])
            ith_ref_better_log_probs = calculate_log_probs(ref_better_log_probs[i, better_seq_slice], better_label_mask[i, better_seq_slice])
            ith_ref_worse_log_probs = calculate_log_probs(ref_worse_log_probs[i, worse_seq_slice], worse_label_mask[i, worse_seq_slice])
            better_log_ratio = ith_better_log_probs - ith_ref_better_log_probs
            worse_log_ratio = ith_worse_log_probs - ith_ref_worse_log_probs
            logits = better_log_ratio - worse_log_ratio
            
            if self.args.ipo:
                losses.append((logits - 1 / (2 * self.scale_coeff)) ** 2)
                losses.append((better_rand_logits - 1 / (2 * self.scale_coeff)) ** 2)
                losses.append((worse_rand_logits - 1 / (2 * self.scale_coeff)) ** 2)
            else:
                losses.append(-F.logsigmoid(self.scale_coeff * logits))
                losses.append(-F.logsigmoid(self.scale_coeff * better_rand_logits))
                losses.append(-F.logsigmoid(self.scale_coeff * worse_rand_logits))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
        
        loss = torch.stack(losses).mean()
        better_sample_rewards = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_rewards = torch.stack(worse_sample_rewards)  # size = (B,)
        rewards_accuracy = (
            (better_sample_rewards > worse_sample_rewards).float().mean()
        )
        better_sample_rewards = better_sample_rewards.mean()  # size = ()
        worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        rewards_margin = better_sample_rewards - worse_sample_rewards
        
        better_img_contributions_rand = torch.stack(better_img_contributions_rand).mean()
        worse_img_contributions_rand = torch.stack(worse_img_contributions_rand).mean()
        
        self.model.backward(loss)
        self.model.step()
        
        loss = get_all_reduce_mean(loss)
        better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        rewards_margin = get_all_reduce_mean(rewards_margin)
        better_img_contributions_rand = get_all_reduce_mean(better_img_contributions_rand)
        worse_img_contributions_rand = get_all_reduce_mean(worse_img_contributions_rand)
        
        return {
            'train/loss': loss.item(),
            'train/better_sample_rewards': better_sample_rewards.item(),
            'train/worse_sample_rewards': worse_sample_rewards.item(),
            'train/rewards_accuracy': rewards_accuracy.item(),
            'train/rewards_margin': rewards_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
            'train/better_img_contributions_rand': better_img_contributions_rand.item(),
            'train/worse_img_contributions_rand': worse_img_contributions_rand.item(),
        }
    
    def ptx_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        images: torch.Tensor,
    ) -> dict[str, Any]:
        outputs = self.model.module(input_ids, attention_mask=attention_mask, labels=labels, images=images)
        ptx_loss = outputs.loss * self.args.ptx_coef
        
        if ptx_loss.isnan():
            import ipdb; ipdb.set_trace()
        
        self.model.backward(ptx_loss)
        self.model.step()
        
        ptx_loss = get_all_reduce_mean(ptx_loss)
        
        return {
            'train/ptx_loss': ptx_loss.item(),
        }
    
    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')
        progress_bar = tqdm(
            total=self.args.num_train_epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.num_train_epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        self.global_step = 0
        epochs_trained, steps_trained_in_current_epoch = 0, 0
        if self.args.resume_from_ckpt is not None:
            if self.use_ptx:
                steps_trained_in_current_epoch = self.model.global_steps * self.args.gradient_accumulation_steps // 2
            else:
                steps_trained_in_current_epoch = self.model.global_steps * self.args.gradient_accumulation_steps
            self.global_step = steps_trained_in_current_epoch
            epochs_trained = steps_trained_in_current_epoch // len(self.train_dataloader)
            steps_trained_in_current_epoch %= len(self.train_dataloader)
            print('\n')
            print(steps_trained_in_current_epoch, epochs_trained, len(self.train_dataloader))
            print(self.model.global_steps, self.args.gradient_accumulation_steps)
            print('\n')
            if not steps_trained_in_current_epoch:
                _step = int(self.args.resume_from_ckpt.split('/')[-1].replace('steps', '').split('-')[-1])
                steps_trained_in_current_epoch = _step
                progress_bar.update(steps_trained_in_current_epoch)
        
        num_prompt_only_batches = len(self.train_dataloader)
        num_ptx_batches = len(self.ptx_train_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches
        
        for epoch in range(self.args.num_train_epochs):
            if epoch < epochs_trained: continue
            self.model.train()
            for batch, ptx_batch in zip(
                self.train_dataloader,
                itertools.chain.from_iterable([self.ptx_train_dataloader] * num_ptx_replicas),
            ):
                progress_bar.update(1)
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                info = self.train_step(**to_device(batch, self.args.device))
                # torch.cuda.empty_cache()
                if self.use_ptx:
                    ptx_info = self.ptx_step(**to_device(ptx_batch, self.args.device))
                    # torch.cuda.empty_cache()
                
                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.num_train_epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)
                if self.use_ptx:
                    self.logger.log(ptx_info, step=self.global_step)
                
                if self.global_step % self.args.save_steps == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(global_steps=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            self.save(global_steps=self.global_step)
            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()
    
    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        ds_config: dict[str, Any] | None = None,
        global_steps: int = -1,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.model  # pylint: disable=no-member
        if ds_config is None:
            ds_config = self.ds_train_config  # pylint: disable=no-member
        
        output_dir = self.args.output_dir
        if global_steps > 0:
            output_dir = os.path.join(output_dir, f'checkpoint-{global_steps}')
            os.makedirs(output_dir, exist_ok=True)

        self.logger.print(f'Saving model to "{output_dir}" ...')

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)
        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(output_dir)

        # Save model checkpoint
        if ds_config['zero_optimization']['stage'] >= 2:
            self.logger.print('Saving DeepSpeed Checkpoints...')
            model.save_checkpoint(output_dir)
            self.logger.print('Converting DeepSpeed Checkpoints to Hugging Face format...')
            if is_main_process():
                subprocess.check_call(
                    [sys.executable, 'zero_to_fp32.py', '.', WEIGHTS_NAME],  # noqa: S603
                    cwd=output_dir,
                )
            dist.barrier()
        else:
            self.logger.print('Saving Hugging Face Checkpoints...')
            if is_main_process():
                model_to_save.save_pretrained(output_dir, is_main_process=True)

        self.logger.print('Model saved!')
    