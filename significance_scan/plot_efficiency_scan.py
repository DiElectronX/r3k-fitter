import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from uncertainties import unumpy


def efficiency_plotter(data, path, show=False):
    score = data['score']
    eff_eek = data['eff_eek']
    eff_eek_err = data['eff_eek_err']
    eff_jpsik = data['eff_jpsik']
    eff_jpsik_err = data['eff_jpsik_err']
    
    _eff_eek = unumpy.uarray(eff_eek,eff_eek_err)
    _eff_jpsik = unumpy.uarray(eff_jpsik,eff_jpsik_err)
    eff_ratio = _eff_eek / _eff_jpsik

    fig, ax = plt.subplots(figsize=(8,8))
    ax.errorbar(score, unumpy.nominal_values(eff_ratio), yerr=unumpy.std_devs(eff_ratio), label=r'$\frac{B \rightarrow eeK}{B \rightarrow J/\psi(ee) K}$', ls='none', marker='.')
    ax.errorbar(score, unumpy.nominal_values(_eff_eek), yerr=unumpy.std_devs(_eff_eek), label=r'$B \rightarrow eeK$', ls='none', marker='.')
    ax.errorbar(score, unumpy.nominal_values(_eff_jpsik), yerr=unumpy.std_devs(_eff_jpsik), label=r'$B \rightarrow J/\psi(ee) K$', ls='none', marker='.')
    
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$\alpha \times \epsilon \ (\frac{N^{Cuts + BDT}_{MC}}{N^{tot}_{MC}})$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='center right',fontsize=16)
    ax.set_yscale('log')

    
    if show:
        fig.show()

    fig.savefig(path, bbox_inches='tight')

def main(args):
    if args.input_file:
        data_file = Path(args.input_file)
        assert data_file.is_file(), 'Cannot find data file'
    else:
        data_file = Path('.') / 'significance_scan_data.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('.') / 'efficiency_scan.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    efficiency_plotter(score_data, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='input_file', 
        type=str, help='pickle data file')
    parser.add_argument('-o', '--output', dest='output', 
        type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    args, _ = parser.parse_known_args()

    main(args)
