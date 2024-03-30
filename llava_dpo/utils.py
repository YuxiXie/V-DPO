import datetime
import logging
import logging.handlers
import os
import sys

import requests
import threading
import dataclasses

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.modeling_outputs import ModelOutput
from transformers.tokenization_utils import BatchEncoding

import optree
from optree.typing import PyTreeTypeVar
from collections import OrderedDict
from typing_extensions import TypeAlias
from typing import Any, Callable, Generator, TypeVar, cast

from llava_dpo.constants import LOGDIR, ASSISTANT_TOKEN_IDS, IMAGE_TOKEN_INDEX, IGNORE_INDEX

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

TensorTree: TypeAlias = PyTreeTypeVar('TensorTree', torch.Tensor)
Func = TypeVar('Func', bound=Callable[..., Any])


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def get_optimizer_grouped_parameters(
    module: nn.Module,
    weight_decay: float,
    no_decay_name_set: set[str] | None = None,
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Get parameter groups with customized weight decay value."""
    if no_decay_name_set is None:
        no_decay_name_set = {'bias', 'LayerNorm.weight'}
    no_decay_name_set = set(map(str.lower, no_decay_name_set))

    named_parameters = [
        (name.lower(), param) for name, param in module.named_parameters() if param.requires_grad
    ]

    return [
        {
            'params': [
                param
                for name, param in named_parameters
                if not any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in named_parameters
                if any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': 0.0,
        },
    ]


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_subclasses(cls: type, memo: set[type] | None = None) -> Generator[type, None, None]:
    """Get all subclasses of a class recursively."""
    if memo is None:
        memo = set()

    for subclass in cls.__subclasses__():
        if subclass in memo:
            continue

        memo.add(subclass)
        yield subclass
        yield from get_subclasses(subclass, memo=memo)


__PYTREE_INITIALIZED = False
__PYTREE_REGISTRY_LOCK = threading.Lock()


def __initialize_pytree_registry_once() -> None:
    # pylint: disable-next=import-outside-toplevel,unused-import

    global __PYTREE_INITIALIZED  # pylint: disable=global-statement
    if __PYTREE_INITIALIZED:
        return

    with __PYTREE_REGISTRY_LOCK:
        if __PYTREE_INITIALIZED:
            return

        optree.register_pytree_node(
            BatchEncoding,
            lambda batch_encoding: (
                [batch_encoding.data],
                {'encoding': batch_encoding.encodings, 'n_sequences': batch_encoding.n_sequences},
            ),
            lambda metadata, children: BatchEncoding(children[0], **metadata),
            namespace='safe_rlhf',
        )
        optree.register_pytree_node(
            ModelOutput,
            lambda model_output: (model_output.values(), model_output.keys(), model_output.keys()),
            lambda keys, values: ModelOutput(OrderedDict(zip(keys, values))),
            namespace='safe_rlhf',
        )

        for model_output_class in filter(dataclasses.is_dataclass, get_subclasses(ModelOutput)):
            optree.register_pytree_node(
                model_output_class,
                lambda model_output: ([dataclasses.asdict(model_output)], type(model_output)),
                lambda metadata, children: metadata(**children[0]),
                namespace='safe_rlhf',
            )

        __PYTREE_INITIALIZED = True


def to_device(batch: TensorTree, device: torch.device | str | int | None) -> TensorTree:
    """Move a batch of tensors to a device."""
    if not __PYTREE_INITIALIZED:
        __initialize_pytree_registry_once()
    if device is None:
        return batch
    return optree.tree_map(lambda x: x.to(device), batch, namespace='llava')


def get_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the mean."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def rank_zero_only(func: Func) -> Func:
    """Decorator to make a function only run on the main process."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for the decorator."""
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return cast(Func, wrapper)

def is_multi_turn(input_ids):
    assistant_indexes = input_ids.eq(ASSISTANT_TOKEN_IDS[-1]).nonzero()
    cnt = 0
    for idx in assistant_indexes:
        if input_ids[idx - len(ASSISTANT_TOKEN_IDS) + 1: idx + 1].tolist() == ASSISTANT_TOKEN_IDS:
            cnt += 1
    return cnt

def get_answer_index(input_ids, final_answer=False):
    assistant_indexes = input_ids.eq(ASSISTANT_TOKEN_IDS[-1]).nonzero()
    assistant_indexes = [_ for _ in assistant_indexes]
    answer_index = assistant_indexes[-1] + 1
    assistant_indexes = assistant_indexes[::-1] if final_answer else assistant_indexes
    for idx in assistant_indexes:
        if input_ids[idx - len(ASSISTANT_TOKEN_IDS) + 1: idx + 1].tolist() == ASSISTANT_TOKEN_IDS:
            answer_index = idx + 1
            break
    return answer_index

def get_indexes(better_input_ids, worse_input_ids, 
                better_attention_mask, worse_attention_mask,
                use_answer_index=False):
    better_end_index = better_attention_mask.nonzero()[-1]
    worse_end_index = worse_attention_mask.nonzero()[-1]
    
    answer_index = get_answer_index(better_input_ids, final_answer=True)
    if not use_answer_index:
        try:
            diverge_index = (better_input_ids != worse_input_ids).nonzero()[0]
        except:
            # diverge_index = better_input_ids[i].eq(IMAGE_TOKEN_INDEX).nonzero()[0] + 1
            diverge_index = answer_index
    try:
        assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
        assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'
    except:
        better_end_index = max(better_end_index, worse_end_index)
        worse_end_index = max(better_end_index, worse_end_index)
    return diverge_index, better_end_index, worse_end_index
    
def calculate_log_probs(log_probs, label_mask):
    return (log_probs * label_mask).sum(dim=-1)

def get_log_probs(input_ids: torch.LongTensor, labels: torch.LongTensor, 
                  log_probs: torch.Tensor, is_answer=False, 
                  pad_token_id=0, final_answer=True):
    end_index = input_ids.ne(pad_token_id).nonzero()[-1]
    start_index = 1
    if is_answer:
        start_index = get_answer_index(input_ids, final_answer=final_answer)
    label_mask = torch.logical_and(labels.ne(IMAGE_TOKEN_INDEX), labels.ne(IGNORE_INDEX))[1:]
    return calculate_log_probs(log_probs[slice(start_index - 1, end_index)],
                               label_mask[slice(start_index - 1, end_index)])

def sample_random_image(shape, image_mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                        image_std=torch.tensor([0.26862954, 0.26130258, 0.27577711])):
    random_img = torch.empty(*shape)
    random_img = torch.randn_like(random_img)    
    for i in range(3):
        random_img[i] = (random_img[i] * image_std[i]) + image_mean[i]
    return random_img
