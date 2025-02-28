import pandas
import sys
import argparse
import logging

import math

import random
import numpy as np

"""
From the comparative results: 

target_id,comparison_id,position,target,score,choices,scores

to:

target_id,target,score,entropy,method

"""

def main():

    logging.basicConfig(level=logging.INFO, format='')
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file', nargs='?', default=sys.stdin)
    argparser.add_argument('--onlyscore', action='store_true', help='whether to output only the scores; otherwise full csv.')   # TODO: re-implement
    argparser.add_argument('--n_positions', required=False, type=int, default=None, help='In case comparison data contains all positions, randomly sample one position per comparison')
    argparser.add_argument('--n_comparisons', required=False, type=int, default=None, help='Randomly sample only this many comparison per word.')
    argparser.add_argument('--scale', required=False, type=str, help='start,end of the scale, to which to map the scores', default='1,5')
    argparser.add_argument('--seed', required=False, type=int, default=None, help='seed, for sampling --n_comparisons or --n_positions only')

    args = argparser.parse_args()
    scale_start, scale_end = (int(i) for i in args.scale.split(','))

    converters = {
        'choices': lambda x: tuple(x.split(';')),
        'proba': lambda x: tuple(float(s) for s in x.split(';')),
    }

    df = pandas.read_csv(args.file, index_col=None, converters=converters)

    if args.n_positions is not None or args.n_comparisons is not None:
        args.seed = args.seed or random.randint(0, 999999)
        logging.info(f'Seed: {args.seed}')
        np.random.seed(args.seed)

    if args.n_positions is not None:
        df = df.groupby(['target_id', 'comparison_id']).sample(n=args.n_positions)

    if args.n_comparisons is not None:
        df = df.set_index(['target_id', 'comparison_id']).pivot(columns=['position']).groupby('target_id').sample(args.n_comparisons).stack(level=1).reset_index()
    df['entropy'] = [-sum(prob * math.log2(prob) for prob in scores) for scores in df['proba']]

    df_agg_per_word = df.groupby(['target_id', 'target']).agg({'result': 'mean', 'entropy': 'mean'})

    df_agg_per_word['result'] = df_agg_per_word['result'] * (scale_end - scale_start) + scale_start
    df_agg_per_word.reset_index()[['target_id', 'target', 'result', 'entropy']].to_csv(sys.stdout, index=None)



if __name__ == '__main__':
    main()