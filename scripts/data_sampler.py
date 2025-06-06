import ROOT
import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(1, str(Path('..').resolve()))
from utils import *

ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']

def main(args):
    input_path = Path(args.input)
    output_path = args.output if args.output else input_path.parent 
    new_filename = ''.join([input_path.stem,(f'_{args.label}' if args.label else '_sampled'),input_path.suffix])
    new_output_path = output_path / new_filename

    if new_output_path.is_file():
        response = input(f"The file '{new_output_path}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()

        if response == 'y':
            print('overwriting...')
            new_output_path.unlink()
        elif response == 'n':
            print('aborting...')
            return 1
        else:
            print('invalid input. aborting...')
            return 1

    rdf = ROOT.RDataFrame(args.tree, args.input)
    event_sum = rdf.Sum(args.weight_branch).GetValue() if args.isMC else rdf.Count().GetValue()
    
    if args.cuts:   
        event_sum_cuts = rdf.Filter(args.cuts).Sum(args.weight_branch).GetValue() if args.isMC else rdf.Count().GetValue()
        cut_eff = event_sum_cuts / event_sum
        try:
            sf = args.value / event_sum / cut_eff
        except ZeroDivisionError:
            print('Zero count')
            sf = 0
    else:
        sf = args.value / event_sum
    
    rdf_sampled = rdf.Define('final_wgt',f'trig_wgt*{sf}')
    rdf_sampled.Snapshot('mytree', str(new_output_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', dest='cfg', type=str, default='fit_cfg.yml', help='fit configuration')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True, help='input file to sample')
    parser.add_argument('-o', '--output', dest='output', type=Path, help='output path')
    parser.add_argument('-m', '--mode', dest='mode', type=str, choices=ALLOWED_MODES, help='which fit region')
    parser.add_argument('-t', '--tree', dest='tree', type=str, default='mytree', help='input file tree')
    parser.add_argument('-w', '--weight_branch', dest='weight_branch', type=str, default='trig_wgt', help='event weight branch for mc files')
    parser.add_argument('-x', '--cuts', dest='cuts', type=str, default=None, help='cut string to filter sample')
    parser.add_argument('-v', '--value', dest='value', type=float, required=True, help='skim value')
    parser.add_argument('-mc', '--mc', dest='isMC', action='store_true', help='mc input file flag')
    parser.add_argument('-l', '--label', dest='label', type=str, help='output file label')
    args, _ = parser.parse_known_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])
    
    if args.mode:
        set_mode(dataset_params, output_params, fit_params, args)

    if args.cuts or args.mode:
        args.cuts = args.cuts if args.cuts else ('{}>{}&&{}>{}&&{}<{}&&{}>{}&&{}<{}'.format(
            dataset_params.score_branch, fit_params.bdt_score_cut,
            dataset_params.ll_mass_branch, fit_params.ll_mass_range[0],
            dataset_params.ll_mass_branch, fit_params.ll_mass_range[1],
            dataset_params.b_mass_branch, fit_params.fit_range[0],
            dataset_params.b_mass_branch, fit_params.fit_range[1],
        ))

    main(args)
