import os
import sys
mydir = os.path.dirname(__file__)
sys.path.append(mydir)

import utils
import importlib
from model.model import Module
import nltk
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
    parser.add_argument('--use_sql_prob_error_analysis', dest='use_sql_prob_error_analysis', action='store_true',
                        help="Whether use sql prob obtained from the error analysis")
    return parser


def main(args):
    args.gpu = False
    utils.manual_seed(args.seed)

    assert args.resume

    AugModel = importlib.import_module('model.{}'.format(args.aug)).Module
    #print("augmodel: ", AugModel) # ans2sql.Module
    args.ftrain = os.path.abspath(args.ftrain)
    args.tables = os.path.abspath(args.tables)
    args.db = os.path.abspath(args.db)
    #print(args) # resume is sql2nl
    gen_m = Module.load_inst(args.resume, overwrite=dict(tables=args.tables, db=args.db, dcache=args.dcache, beam_size=args.beam_size, batch=args.batch, fparser=args.fparser))
    #print("gen_m: ", gen_m) # nl2sql
    fout = args.fout

    if args.beam_size:
        fout = fout.replace('.json', '.beam.json')
    gen_m.generate_sql(args.num_gen, AugModel, args.db_split, args=args) # calls from sql2nl_sparc.py


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
