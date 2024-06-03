import torch
import torch.nn.functional as F
from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def gather_log_probabilities(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.long().unsqueeze(dim=-1))
    return log_probs_labels.squeeze(dim=-1)


def gather_log_probabilities_for_cfg(logits: torch.Tensor, llm_logits: torch.Tensor, labels: torch.LongTensor, gamma=1):
    modified_logits = gamma * logits - (gamma - 1) * llm_logits
    log_probs = F.log_softmax(modified_logits.float(), dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.long().unsqueeze(dim=-1))
    return log_probs_labels.squeeze(dim=-1)
