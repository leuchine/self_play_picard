from typing import Optional
from dataclasses import dataclass, field
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.picard_model_wrapper import PicardArguments
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from pathlib import Path
import json
import sys
import os

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class SQL2TextArguments:
    pretrain_file: str = field(
        default='',
        metadata={"help": "self play filename."})
    weight_decay_value: float = field(
        default=0.0,
        metadata={"help": "Value for weight decay."})
    pretrain_epochs: int = field(
        default=0,
        metadata={"help": "Pretraining epoches with self-play data."})
    max_source_seq_length: int = field(
        default=0,
        metadata={"help": "Max source sequence length."})
    max_target_seq_length: int = field(
        default=0,
        metadata={"help": "Max target sequence length."})
    beam_size: int = field(
        default=0,
        metadata={"help": "Beam search beam size."})
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether use sampling during text generation."})
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for text generation."})


def parse_args():
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (PicardArguments, ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments, SQL2TextArguments)
    )
    picard_args: PicardArguments
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    sql2text_args: SQL2TextArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, model_args, data_args, data_training_args, training_args, sql2text_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        picard_args, model_args, data_args, data_training_args, training_args, sql2text_args = parser.parse_dict(args=data)
    else:
        picard_args, model_args, data_args, data_training_args, training_args, sql2text_args = parser.parse_args_into_dataclasses()
    return picard_args, model_args, data_args, data_training_args, training_args, sql2text_args