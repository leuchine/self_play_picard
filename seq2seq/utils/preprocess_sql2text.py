import argparse
import json
import os

import _jsonnet
import tqdm

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql.utils import registry
# noinspection PyUnresolvedReferences
from ratsql.utils import vocab
from ratsql.utils.evaluation import *

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./pretrained_models', local_files_only=True)
REMOVE_VALUES = True

def pad_single_sentence(toks, cls=True):
    if cls:
        return [tokenizer.cls_token] + toks + [', ']
    else:
        return toks + [', ']

def convert_schema(schema, enc_preproc):
    processed_schema = enc_preproc._preprocess_schema(schema)
    cols = [pad_single_sentence(c[:-1], cls=False) for c in processed_schema.column_names]
    tabs = [pad_single_sentence(t, cls=False) for t in processed_schema.table_names]

    token_list = [c  for col in cols for c in col] + [
        t if t !='[SEP]' else ', ' for tab in tabs for t in tab]
    schema = ''.join([t.replace('Ġ', ' ') for t in token_list])
    return schema.rstrip(', ')

def add_turns_to_context(contexts, turns):
    prev_question = ''
    prev_sql = []
    goal = replace_table_alias(turns[-1]['query'])
    for idx, turn in enumerate(turns):
        if idx + 1 == len(turns):
            target = turn['question'].split('<s>')[0].strip() + ' <s>'
        else:
            target = turn['question'].split('<s>')[0].strip()
        contexts.append({
            'goal': goal,
            'question': prev_question,
            'query': ' <s> '.join(prev_sql),
            'target': target,
            'schema': turn['schema']
        })
        prev_question = turn['question']
        prev_sql = [remove_values(replace_table_alias(turn['query'])) if REMOVE_VALUES else replace_table_alias(
            turn['query'])] + prev_sql

class SQL2TextPreproc:

    def __init__(self, path, section):
        self.path = path
        self.section = section
        self.save_path = path + '/' + section + '_sql2text.json'
        self.contexts = []

    def add_items(self, turns):
        add_turns_to_context(self.contexts, turns)

    def save(self):
        with open(self.save_path, 'w') as outfile:
            json.dump(self.contexts, outfile, indent=4)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])

    def preprocess(self):
        for section in self.config['data']:
            data = registry.construct('dataset', self.config['data'][section])
            sql2text_preproc = SQL2TextPreproc(os.path.dirname(self.config['data'][section]['paths'][0]),
                                               section)
            turns = []
            for item in tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True):
                question = item.orig['question']
                query = item.orig['query']
                schema = convert_schema(item.schema, self.model_preproc.enc_preproc)
                if '<s>' not in question:
                    if len(turns) > 0:
                        sql2text_preproc.add_items(turns)
                    turns = []
                turns.append({'question': question,
                              'query': query,
                              'schema': schema})

            # save
            sql2text_preproc.save()

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))
    preprocessor = Preprocessor(config)
    preprocessor.preprocess()


if __name__ == '__main__':
    args = add_parser()
    main(args)