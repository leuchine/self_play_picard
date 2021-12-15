import os
import sys
mydir = os.path.dirname(__file__)
sys.path.append(mydir)
import json
import preprocess_nl2sql_sparc as preprocess_nl2sql
import importlib
import converter
from model.model import Module
from collections import defaultdict, Counter
import nltk
import math
nltk.download('punkt')

def get_parser():
    parser = Module.get_default_parser(lr=5e-5, batch=20, epoch=50, model='sql2nl', seed=3)
    parser.add_argument('--dataset', default='spider', choices=('spider', 'sparc', 'cosql'), help='dataset to use')
    parser.add_argument('--dcache', default=os.path.join(mydir, 'cache', 'bert'), help='cache directory')
    parser.add_argument('--num_gen', type=int, default=20000, help='how many examples to generate')
    parser.add_argument('--beam_size', type=int, default=0, help='beam size')
    parser.add_argument('--fparser', default='exp/nl2sql/default/best.pt', help='parser model to use for nl generation')
    parser.add_argument('--ftrain', default='data/spider/train.json', help='train json files')
    parser.add_argument('--db_split', default='eval', help='split to do augmentation on', choices=('train', 'eval'))
    parser.add_argument('--tables', default='data/tables.json', help='tables json')
    parser.add_argument('--db', default='data/database', help='SQLite database folder')
    parser.add_argument('--aug', default='ans2sql')
    parser.add_argument('--fout', default='gen/gen.json', help='generated output  file')
    return parser


def main(args):
    AugModel = importlib.import_module('model.{}'.format(args.aug)).Module
    conv = converter.Converter(tables=getattr(args, 'tables', 'data/spider/tables'),
                               db=getattr(args, 'db', 'data/database'))
    # sample some sql queries
    db_ids = sorted([k for k in conv.database_schemas.keys() if
                     os.path.isfile(os.path.join(args.db, k, '{}.sqlite'.format(k)))])
    print(len(db_ids), 'total db ids found')
    assert db_ids
    proc_cols = AugModel.process_cols(db_ids, conv.database_schemas)
    template_to_score = defaultdict(list)
    template_to_example_sql = defaultdict(str)
    paths = ["./val.eval", "./train.eval"]
    for path in paths:
        with open(path) as f:
            eval_data = json.load(f)
            for i in range(len(eval_data['per_item'])):
                db_id = eval_data['per_item'][i]['db_id']
                predicted = eval_data['per_item'][i]['predicted']
                query = eval_data['per_item'][i]['gold']
                exact_score = eval_data['per_item'][i]['exact']
                query_toks, query_toks_no_value = preprocess_nl2sql.SQLDataset.tokenize_query(query)
                query_norm = conv.convert_tokens(query_toks, query_toks_no_value, db_id)
                col_map = {c['key']: c for c in proc_cols[db_id]}
                if query_norm is None:
                    continue
                temp = AugModel.templatize(col_map, query_norm.split())
                template_to_score[temp].append(exact_score)
                template_to_example_sql[temp] = query
                print(temp)
    template_score_list = []
    threshold = 2
    for template, scores in template_to_score.items():
        if len(scores) > threshold:
            print(len(scores), template, sum(scores) / float(len(scores)))
            template_score_list.append((len(scores), template, sum(scores) / float(len(scores))))
    prob_threshold = 0.98
    filtered_template_score_list = []
    for i in sorted(template_score_list):
        i = list(i)
        if i[2] < prob_threshold:
            i.append(template_to_example_sql[i[1]])
            filtered_template_score_list.append(i)
            print(i)
    template_to_weight = {}
    for (count, template, exact_score, _) in filtered_template_score_list:
        print(template, count * (1 - exact_score))
        template_to_weight[template] = count * (1 - exact_score)
    temperature = 20
    denominator = sum([math.exp(v / temperature) for v in template_to_weight.values()])
    sql_prob = Counter({k: (math.exp(v / temperature) / denominator) for k, v in template_to_weight.items()})
    print(sql_prob)
    json.dump(sql_prob, open('sql_prob.json', 'w'), indent=4)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
