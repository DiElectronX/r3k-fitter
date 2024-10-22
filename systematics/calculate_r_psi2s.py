import os
import sys
import yaml
import csv
import argparse
import pickle
from tqdm import tqdm
from uncertainties import ufloat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
from pathlib import Path
import ROOT
import ROOT.RooFit as rf

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from utils import *


def get_eff_cut_and_count(file_name, tree_name, cut_string, weights='trig_wgt', denom_string=None, denom=500_000_000, get_comps=False):
    df = ROOT.RDataFrame(tree_name, file_name)
    if weights:
        k = df.Filter(cut_string).Sum(weights).GetValue()
        n = df.Filter(denom_string).Sum(weights).GetValue() if denom_string else denom
    else:
        k = df.Filter(cut_string).Count().GetValue()
        n = df.Filter(denom_string).Count().GetValue() if denom_string else denom

    try:
        eff = k / n
        unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
        # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
    except ZeroDivisionError:
        eff = 0
        unc = 0
    
    if get_comps:
        return eff, unc, (k, np.sqrt(k), n, np.sqrt(n))
    else:
        return eff, unc


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    args.mode = 'jpsi'
    set_mode(dataset_params, output_params, fit_params, args)
    eff_jpsi_mc, eff_jpsi_mc_err, eff_jpsi_mc_comps = get_eff_cut_and_count(
        dataset_params.jpsi_file, 
        dataset_params.tree_name, 
        f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(fit_params.bdt_score_cut)}', 
        denom=509825327, 
        weights='trig_wgt', 
        get_comps=True,
    )
    args.mode = 'psi2s'
    set_mode(dataset_params, output_params, fit_params, args)
    eff_psi2s_mc, eff_psi2s_mc_err, eff_psi2s_mc_comps = get_eff_cut_and_count(
        dataset_params.psi2s_file, 
        dataset_params.tree_name, 
        f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(fit_params.bdt_score_cut)}', 
        denom=49245742, 
        weights='trig_wgt', 
        get_comps=True,
    )

    # print(f'J/\u03C8 MC Efficiency = {eff_jpsi_mc} \u00B1 {eff_jpsi_mc_err}')   
    # print(f'\u03C8(2s) MC Efficiency = {eff_psi2s_mc} \u00B1 {eff_psi2s_mc_err}')   
    
    _n_jpsi    = ufloat(62075.82,297.82)
    _n_psi2s   = ufloat(4419.40,74.54)
    _eff_jpsi  = ufloat(eff_jpsi_mc, eff_jpsi_mc_err)
    _eff_psi2s = ufloat(eff_psi2s_mc, eff_psi2s_mc_err)
    _r_psi2s = (_n_psi2s * _eff_jpsi) / (_n_jpsi * _eff_psi2s)

    _br_b_jpsik  = ufloat(1.02E-3,0.019E-3)
    _br_b_psi2sk = ufloat(6.24E-4,0.21E-4)
    _br_jpsi_ee = ufloat(.05971,.00032)
    _br_psi2s_ee = ufloat(7.94E-3,0.22E-3)
    _r_psi2s_sm = (_br_b_psi2sk * _br_psi2s_ee) / (_br_b_jpsik * _br_jpsi_ee)
    
    print(f'R(\u03C8(2s)) = {round(_r_psi2s.n,3)} \u00B1 {round(_r_psi2s.std_dev,6)}')   
    print(f'R(\u03C8(2s)) [SM] = {round(_r_psi2s_sm.n,3)} \u00B1 {round(_r_psi2s_sm.std_dev,6)}')   

    if args.plot:
        fig, ax = plt.subplots(figsize=(8,8))

        x_positions = ['R(\u03C8(2s))']
        ax.errorbar(x_positions, [_r_psi2s.n], yerr=[_r_psi2s.std_dev], fmt='o', label='2022 Estimate', markersize=8, capsize=5)
        ax.errorbar(x_positions, [_r_psi2s_sm.n], yerr=[_r_psi2s_sm.std_dev], fmt='o', markerfacecolor='none', markersize=8, label='SM Prediction', capsize=5)
        #plt.xticks(x_positions, [x_value])

        ax.set_xlabel('Category', loc='right')
        ax.set_ylabel('Ratio', loc='top')
        ax.legend(fontsize=14, markerscale=1.5)
        ax.set_ylim(.05,.1)
        ax.set_xticks([0])
        ax.set_xticklabels(x_positions)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
        ax.grid(True, axis='x', which='major', linestyle='-', linewidth=1)
        ax.grid(True, axis='y', which='major', linestyle='-', color='black')
        ax.grid(True, axis='y', which='minor', linestyle='--', color='gray')
        
        path = Path('.') / 'plots' / 'r_psi2s.pdf'
        fig.savefig(path, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', help='make a plot for R(psi(2s))')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    args = parser.parse_args()

    main(args)
