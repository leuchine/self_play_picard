from seq2seq.utils.args import parse_args
import datasets
from seq2seq.utils.cosql import cosql_add_serialized_schema
from seq2seq.utils.spider import spider_add_serialized_schema
import os
import json
import tqdm
from seq2seq.sqlprocess.process_sql import replace_table_alias


def add_turns_to_context(contexts, turns):
    prev_question = []
    prev_sql = []
    goal = replace_table_alias(turns[-1]['goal'])
    for idx, turn in enumerate(turns):
        if idx + 1 == len(turns):
            target = turn['question'].strip() + ' <s>'
        else:
            target = turn['question'].strip()
        contexts.append({
            'goal': goal,
            'question': ' / '.join(prev_question),
            'query': ' / '.join(prev_sql),
            'target': target,
            'schema': turn['schema'].lstrip(" | ")
        })
        prev_question = [target] + prev_question
        prev_sql = [replace_table_alias(turn['query'])] + prev_sql


class SQL2TextPreproc:

    def __init__(self, path, section):
        self.path = path
        self.section = section
        self.save_path = os.path.join(path, section + '.json')
        self.contexts = []

    def add_items(self, turns):
        add_turns_to_context(self.contexts, turns)

    def save(self):
        with open(self.save_path, 'w') as outfile:
            json.dump(self.contexts, outfile, indent=4)

class Preprocessor:

    def __init__(self, sections, splits, model_args, data_args):
        self.sections = sections
        self.splits = splits
        self.model_args = model_args
        self.data_args = data_args

    def preprocess(self):
        # save path
        path = os.path.join(self.model_args.cache_dir, self.data_args.dataset + "_sql2text")
        if not os.path.exists(path):
            os.mkdir(path)

        for section, split in zip(self.sections, self.splits):
            sql2text_preproc = SQL2TextPreproc(path, section)
            for dataset in split:
                turns = []
                last_turn_idx = -1
                for item in tqdm.tqdm(dataset):
                    if 'turn_idx' not in item:
                        item['turn_idx'] = 0
                    if 'goal' not in item:
                        item['goal'] = item['query']
                    if item['turn_idx'] == -1:
                        continue
                    goal, question, query, schema = item['goal'], item['question'], item['query'], item['serialized_schema']
                    if item['turn_idx'] <= last_turn_idx:
                        if len(turns) > 0:
                            sql2text_preproc.add_items(turns)
                        turns = []
                    turns.append({'goal': goal,
                                  'question': question,
                                  'query': query,
                                  'schema': schema})
                    last_turn_idx = item['turn_idx']
                # the last dialogue
                if len(turns) > 0:
                    sql2text_preproc.add_items(turns)
            sql2text_preproc.save()

def preprocess_dataset():
    picard_args, model_args, data_args, data_training_args, training_args, _, _ = parse_args()
    path = os.path.join(model_args.cache_dir, data_args.dataset + "_sql2text")
    if not os.path.exists(os.path.join(path, 'train.json')) or not os.path.exists(
            os.path.join(path, 'validation.json')
    ):
        # spider dataset
        spider_dataset_dict = datasets.load.load_dataset(
            path=data_args.dataset_paths["spider"], cache_dir=model_args.cache_dir
        )
        _spider_add_serialized_schema = lambda ex: spider_add_serialized_schema(
            ex=ex,
            data_training_args=data_training_args,
        )
        spider_train_dataset = spider_dataset_dict['train'].map(
            _spider_add_serialized_schema,
            batched=False,
            num_proc=data_training_args.preprocessing_num_workers,
            load_from_cache_file=not data_training_args.overwrite_cache,
        )

        if data_args.dataset == "cosql+spider":
            cosql_dataset_dict = datasets.load.load_dataset(
                path=data_args.dataset_paths["cosql"], cache_dir=model_args.cache_dir
            )
            _cosql_add_serialized_schema = lambda ex: cosql_add_serialized_schema(
                ex=ex,
                data_training_args=data_training_args,
            )
            cosql_train_dataset = cosql_dataset_dict['train'].map(
                _cosql_add_serialized_schema,
                batched=False,
                num_proc=data_training_args.preprocessing_num_workers,
                load_from_cache_file=not data_training_args.overwrite_cache,
            )
            cosql_dev_dataset = cosql_dataset_dict['validation'].map(
                _cosql_add_serialized_schema,
                batched=False,
                num_proc=data_training_args.preprocessing_num_workers,
                load_from_cache_file=not data_training_args.overwrite_cache,
            )
            train_dataset = [spider_train_dataset, cosql_train_dataset]
            dev_dataset = [cosql_dev_dataset]

        preprocessor = Preprocessor(['train', 'validation'],
                                    [train_dataset, dev_dataset],
                                    model_args,
                                    data_args)
        preprocessor.preprocess()

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()