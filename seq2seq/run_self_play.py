# todo: most self play data have single turns...

import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

import os
from contextlib import nullcontext
from transformers import T5Tokenizer
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from seq2seq.utils.pipeline import ConversationalText2SQLInput, ConversationalText2SQLGenerationPipeline
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.args import ModelArguments, SQL2TextArguments
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
import torch
from seq2seq.utils.pipeline import get_schema
from seq2seq.sqlprocess.process_sql import replace_table_alias
from seq2seq.utils.dataset import serialize_schema
from copy import deepcopy
import json
from seq2seq.sqlprocess.parse_sql import parse_one_sql
import re
import datasets.load
from third_party.spider.preprocess.get_tables import dump_db_json_schema


def convert_utterances(utterances):
    new_utterances = []
    for u in utterances:
        new_utterances.extend(re.split("/|\|", u))
    return new_utterances

def read_goals(path, worker_id, num_worker):
    goals = json.load(open(path))
    assigned_goals = []
    for idx, goal in enumerate(goals):
        if idx % num_worker == worker_id:
            assigned_goals.append(goal)
    return assigned_goals

def save_self_play_data(file_writer,
                        db_id,
                        previous_utterance,
                        previous_sql):
    assert len(previous_utterance) == len(previous_sql)
    for idx in range(len(previous_utterance)):
        turn_data = {
            "goal": previous_sql[-1],
            "utterances": convert_utterances(previous_utterance[:idx + 1]),
            "question": previous_utterance[idx].replace("|", "/"),
            "query": previous_sql[idx],
            "db_id": db_id,
            "turn_idx": idx,
        }
        file_writer.write(json.dumps(turn_data) + '\n')
        file_writer.flush()

def filter(metrics, goal, infer_sql, self_play_args, threshold=0.5):
    def create_reference(goal, self_play_args):
        schema = dump_db_json_schema(
            self_play_args.db_path + "/" + goal["db_id"] + "/" + goal["db_id"] + ".sqlite", goal["db_id"]
        )

        reference = {
            "db_id": goal["db_id"],
            "query": goal["sql"],
            "db_path": self_play_args.db_path,
            "db_table_names": schema["table_names_original"],
            "db_column_names":
                {"table_id": [table_id for table_id, _ in schema["column_names_original"]],
                 "column_name": [column_name for _, column_name in schema["column_names_original"]]},
            "db_foreign_keys":
                {"column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                 "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]},
        }
        return reference

    eval_result = metrics._compute([infer_sql], [create_reference(goal, self_play_args)])
    combined_acc, combined_acc_count = 0, 0
    combined_rec, combined_rec_count = 0, 0
    #print("printing value in partial: ")
    for _, value in eval_result['partial'].items():
        #print("value", value)
        combined_acc += value['acc']
        combined_rec += value['rec']
        combined_acc_count += value['acc_count']
        combined_rec_count += value['rec_count']
    combined_acc = float(combined_acc) / combined_acc_count
    combined_rec = float(combined_rec) / combined_rec_count
    return combined_acc > threshold and combined_rec > threshold # and (eval_result['exact_match'] == 1.0 or eval_result['exec'] == 1.0)


@dataclass
class SelfPlayArguments:
    """
    Arguments pertaining to self-play.
    """
    db_path: str = field(
        metadata={"help": "Where to to find the sqlite files"},
    )
    text2sql_model_path: str = field(
        metadata={"help": "Path for the text2sql model"},
    )
    sql2text_base_model: str = field(
        metadata={"help": "Base model for sql2text"}
    )
    sql2text_path: str = field(
        metadata={"help": "Path for the SQL2text model"}
    )
    goal_path: str = field(
        metadata={
            "help": "Path of goals"
        },
    )
    table_path: str = field(
        metadata={
            "help": "Path of tables"
        },
    )
    device: int = field(
        default=0,
        metadata={
            "help": "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU. A non-negative value will run the model on the corresponding CUDA device id."
        },
    )


class PretrainedSQL2Text:

    def __init__(self, model_args, self_play_args, sql2text_args, data_args, data_training_args):
        self.model_args = model_args
        self.self_play_args = self_play_args
        self.sql2text_args = sql2text_args
        self.data_args = data_args
        self.data_training_args = data_training_args
        self.model = torch.load(self_play_args.sql2text_path)
        self.orig_gen = deepcopy(self.model.generate)  # Avoid generate function overridden by Picard
        self.model.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(self_play_args.sql2text_base_model,
                                                     cache_dir=model_args.cache_dir,
                                                     local_files_only=False)
        self.tokenizer.add_tokens(['<', '<s>'])
        self.schema_cache = {}

    def generate(self, goal, previous_utterance, previous_sql_with_alias, db_id):
        previous_sql = []
        for sql_with_alias in previous_sql_with_alias:
            try:
                previous_sql.append(replace_table_alias(sql_with_alias))
            except:
                previous_sql.append(sql_with_alias)
        previous_utterance = convert_utterances(previous_utterance)
        # get serialized schema
        if db_id not in self.schema_cache:
            self.schema_cache[db_id] = get_schema(db_path=self.self_play_args.db_path, db_id=db_id)
        schema = self.schema_cache[db_id]
        serialized_schema = serialize_schema(
            question='',
            db_path=self.self_play_args.db_path,
            db_id=db_id,
            db_column_names=schema["db_column_names"],
            db_table_names=schema["db_table_names"],
            schema_serialization_type=self.data_training_args.schema_serialization_type,
            schema_serialization_randomized=self.data_training_args.schema_serialization_randomized,
            schema_serialization_with_db_id=self.data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.data_training_args.schema_serialization_with_db_content,
            normalize_query=self.data_training_args.normalize_query,
        ).lstrip(" | ")
        # construct context for generation
        previous_utterance = ' / '.join(reversed(previous_utterance))
        previous_sql = ' / '.join(reversed(previous_sql))
        context = ' | '.join([goal['sql'].strip(),
                              previous_utterance.strip(),
                              previous_sql.strip(),
                              serialized_schema.strip()]).lower()
        utterance = self.infer_turn(context)
        if utterance.endswith('<s>'):
            utterance = utterance.rstrip('<s>').strip()
            return utterance, True
        else:
            return utterance, False

    def infer_turn(self, turn):
        self.model.generate = self.orig_gen  # Avoid generate function overridden by Picard.
        input_ids = self.tokenizer(turn,
                                   max_length=self.sql2text_args.max_source_seq_length,
                                   truncation=True,
                                   return_tensors='pt').input_ids.cuda()
        outputs = self.model.generate(input_ids,
                                      num_beams=self.sql2text_args.beam_size,
                                      do_sample=self.sql2text_args.do_sample,
                                      length_penalty=self.sql2text_args.length_penalty,
                                      max_length = self.sql2text_args.max_target_seq_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class PretrainedText2SQL:

    def __init__(self, picard_args, self_play_args, model_args, data_training_args, data_args):
        self.picard_args = picard_args
        self.self_play_args = self_play_args
        self.model_args = model_args
        self.data_training_args = data_training_args
        self.data_args = data_args
        # Initialize config
        self.config = AutoConfig.from_pretrained(
            self_play_args.text2sql_model_path,
            cache_dir=model_args.cache_dir,
            max_length=data_training_args.max_target_length,
            num_beams=data_training_args.num_beams,
            num_beam_groups=data_training_args.num_beam_groups,
            diversity_penalty=data_training_args.diversity_penalty,
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self_play_args.text2sql_model_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
        )

        # Get Picard model class wrapper
        if self.picard_args.use_picard:
            self.model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=self.picard_args, tokenizer=self.tokenizer
            )
        else:
            self.model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        self.model = self.model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            self.self_play_args.text2sql_model_path,
            config=self.config,
            cache_dir=self.model_args.cache_dir,
        )
        # Initalize generation pipeline
        self.pipe = ConversationalText2SQLGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            db_path=self.self_play_args.db_path,
            prefix=self.data_training_args.source_prefix,
            normalize_query=self.data_training_args.normalize_query,
            schema_serialization_type=self.data_training_args.schema_serialization_type,
            schema_serialization_with_db_id=self.data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.data_training_args.schema_serialization_with_db_content,
            device=self.self_play_args.device
        )

    def generate(self, utterances, db_id):
        utterances = convert_utterances(utterances)
        outputs = self.pipe(inputs=ConversationalText2SQLInput(utterances=utterances,
                                            db_id=db_id))
        output = outputs[0]
        query = output["generated_text"]
        return query

def run_self_play(data_args, self_play_args, text2sql_model, sql2text_model, worker_id, num_worker):
    goals = read_goals(self_play_args.goal_path, worker_id, num_worker)
    #print("goals: ", goals)
    if 'cosql' in data_args.dataset:
        metrics =  datasets.load.load_metric(path=data_args.metric_paths["cosql"],
                                             config_name=data_args.metric_config,
                                             test_suite_db_dir=data_args.test_suite_db_dir)

    if 'sparc' in data_args.dataset:
        metrics =  datasets.load.load_metric(path=data_args.metric_paths["sparc"],
                                             config_name=data_args.metric_config,
                                             test_suite_db_dir=data_args.test_suite_db_dir)

    keep_count = 0
    total_count = 0
    os.makedirs(data_args.save_self_play_path, exist_ok=True)
    with open(os.path.join(data_args.save_self_play_path, "self_play_{}.jsonl".format(worker_id)), 'w') as file_writer:
        for goal in goals:
            #print("Try goal: ", goal)
            goal['sql'] = replace_table_alias(goal['sql'])
            try:  # ill-formatted SQL from GAZP.
                parse_one_sql(goal['sql'], goal['db_id'], self_play_args.table_path)
            except:
                continue
            #print("parsed res: ", parse_one_sql(goal['sql'], goal['db_id'], self_play_args.table_path))
            previous_utterance, previous_sql = [], []
            eos = False
            count = 0
            try:
                while count < 5 and not eos:
                    # sql2text
                    new_utterance, eos = sql2text_model.generate(goal, previous_utterance, previous_sql, goal['db_id'])
                    previous_utterance.append(new_utterance)
                    # text2sql
                    new_sql = text2sql_model.generate(previous_utterance, goal['db_id'])
                    previous_sql.append(new_sql)
                    count += 1
            except:
                continue  # skip this dialogue for now.
            total_count += 1
            if filter(metrics, goal, previous_sql[-1], self_play_args):
                keep_count += 1
                print([goal['db_id'], goal['sql'], previous_utterance, previous_sql])
                save_self_play_data(file_writer,
                                    goal['db_id'],
                                    previous_utterance,
                                    previous_sql)

            print("keep rate: %.2f" % (float(keep_count) / total_count))



def main():
    parser = HfArgumentParser((PicardArguments,
                               SelfPlayArguments,
                               ModelArguments,
                               SQL2TextArguments,
                               DataArguments,
                               DataTrainingArguments))
    picard_args: PicardArguments
    self_play_args: SelfPlayArguments
    model_args: ModelArguments
    sql2text_args: SQL2TextArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    picard_args, self_play_args, model_args, sql2text_args, data_args, data_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    worker_id = int(sys.argv[2])
    num_worker = int(sys.argv[3])
    # Create sql2text model before text2sql model. Otherwise, generate function will be overridden.
    sql2text_model = PretrainedSQL2Text(model_args, self_play_args, sql2text_args, data_args, data_training_args)
    with PicardLauncher() if picard_args.launch_picard else nullcontext(None):
        text2sql_model = PretrainedText2SQL(picard_args, self_play_args, model_args, data_training_args, data_args)
        run_self_play(data_args, self_play_args, text2sql_model, sql2text_model, worker_id, num_worker)

if __name__ == "__main__":
    main()