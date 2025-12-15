import os
import sys
import yaml
import argparse
import pickle
from uncertainties import ufloat, UFloat
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import ROOT

from fit_models import *
from utils import *

def round_val(val, sig_fig=2):
    return int(val) if val>1 else float(f'%.{sig_fig}g' % val)

def get_yield(filename, dataset_params, output_params, fit_params, args, cut_string=None, mass_range=None, plot=None):
    set_mode(dataset_params, output_params, fit_params, args)
    fit_params.fit_range = fit_params.fit_range if mass_range is None else mass_range

    df = ROOT.RDataFrame(dataset_params.tree_name, filename)
    cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]})' \
                 f' && (Bmass > {fit_params.fit_range[0]} && Bmass < {fit_params.fit_range[1]})' \
                 f' && bdt_score>{str(fit_params.bdt_score_cut)}' \
                 if cut_string is None else cut_string
    
    df = df.Filter(cut_string)
    k = df.Sum(dataset_params.mc_weight_branch).GetValue()
    
    if plot:
        fig, ax = plt.subplots(figsize=(8,8))
        bins = np.linspace(fit_params.fit_range[0],fit_params.fit_range[1], 30)
        k_arr = df.AsNumpy(columns=[dataset_params.b_mass_branch,dataset_params.mc_weight_branch])
        ax.hist(k_arr[dataset_params.b_mass_branch], bins=bins, weights=k_arr[dataset_params.mc_weight_branch], histtype='step')
        # leg_title = fig_labels[label].replace('J/ \psi','\psi(2s)') if mode=='psi2s' else fig_labels[label]
        # ax.legend(loc='upper left', title=leg_title+f'\nBDT Score > {str(round(bdt_cut,2))}', title_fontsize='14', fontsize='14')
        ax.set_xlabel('m(B Candidate) [GeV]', fontsize=16, loc='right')
        ax.set_ylabel('$N_{events}$', fontsize=16, loc='top')
        
        outpath = Path(plot)
        fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
        plt.close(fig)

    return round_val(k)


def get_weighted_yield_from_dict(file_dict, dataset_params, output_params, fit_params, args, cut_string=None, mass_range=None, plot=None):
    set_mode(dataset_params, output_params, fit_params, args)
    fit_params.fit_range = fit_params.fit_range if mass_range is None else mass_range
    
    k_sum = 0
    plot_dict = {}
    for sample, sample_dict in file_dict.items():
        df = ROOT.RDataFrame(dataset_params.tree_name, sample_dict['file'])
        cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]})' \
                     f' && (Bmass > {fit_params.fit_range[0]} && Bmass < {fit_params.fit_range[1]})' \
                     f' && bdt_score>{str(fit_params.bdt_score_cut)}' \
                     if cut_string is None else cut_string
        
        df = df.Filter(cut_string)
        k_sum += df.Count().GetValue() * sample_dict['sf']
        plot_dict[sample] = df.AsNumpy(columns=[dataset_params.b_mass_branch])[dataset_params.b_mass_branch]
        
    if plot:
        fig, ax = plt.subplots(figsize=(8,8))
        bins = np.linspace(fit_params.fit_range[0],fit_params.fit_range[1], 30)

        hist_sum = np.zeros_like(bins[1:])
        for sample, arr in plot_dict.items():
            sfs = file_dict[sample]['sf'] if isinstance(file_dict[sample]['sf'],np.ndarray) else np.ones_like(arr)*file_dict[sample]['sf']
            weights = [(sf.n if isinstance(sf,UFloat) else sf) for sf in sfs]
            hist, bin_edges = np.histogram(arr, bins=bins, weights=weights)
            bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])
            ax.errorbar(
                bin_centers,
                hist,
                yerr=0,
                marker='',
                drawstyle='steps-mid',
                label=file_dict[sample]['label'],
            )
            hist_sum += hist if len(plot_dict)>1 else hist_sum
       
        if len(plot_dict)>1:
            ax.errorbar(
                bin_centers,
                hist_sum,
                yerr=0,
                marker='',
                drawstyle='steps-mid',
                label='Sum of Hists',
            )
        
        ax.legend()
        ax.set_xlabel('m(B Candidate) [GeV]', fontsize=16, loc='right')
        ax.set_ylabel('$N_{Candidates}$', fontsize=16, loc='top')
            
        outpath = Path(plot)
        fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
        plt.close(fig)

    return k_sum


def get_eff(filename, dataset_params, output_params, fit_params, args, denom=None, cut_string=None, mass_range=None, get_comps=None):
    set_mode(dataset_params, output_params, fit_params, args)
    fit_params.fit_range = fit_params.fit_range if mass_range is None else mass_range

    df = ROOT.RDataFrame(dataset_params.tree_name, filename)
    cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]})' \
                 f' && (Bmass > {fit_params.fit_range[0]} && Bmass < {fit_params.fit_range[1]})' \
                 f' && bdt_score>{str(fit_params.bdt_score_cut)}' \
                 if cut_string is None else cut_string

    k = df.Filter(cut_string).Sum(dataset_params.mc_weight_branch).GetValue()
    n = df.Sum(dataset_params.mc_weight_branch).GetValue() if denom is None else denom

    try:
        eff = k / n
        unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
        # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
    except ZeroDivisionError:
        eff = 0
        unc = 0

    if get_comps:
        return round_val(eff), round_val(unc), (round_val(k), round_val(np.sqrt(k)), round_val(n), round_val(np.sqrt(n)))
    else:
        return round_val(eff), round_val(unc)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])
    
    args.mode = 'lowq2'
    low_q2_yield = get_yield(dataset_params.rare_file, dataset_params, output_params, fit_params, cut_string=None, mass_range=None)
    low_q2_eff = get_eff(dataset_params.rare_file, dataset_params, output_params, fit_params, denom=None, cut_string=None, mass_range=None, get_comps=True)
    print(f'lowq2 yield = {low_q2_yield}')
    print(f'lowq2 eff = {low_q2_eff}')

    args.mode = 'jpsi'
    jpsi_yield = get_yield(dataset_params.rare_file, dataset_params, output_params, fit_params, cut_string=None, mass_range=None)
    jpsi_eff = get_eff(dataset_params.rare_file, dataset_params, output_params, fit_params, denom=509825327, cut_string=None, mass_range=None, get_comps=True)
    print(f'jpsi yield = {jpsi_yield}')
    print(f'jpsi eff = {jpsi_eff}')

    args.mode = 'psi2s'
    psi2s_yield = get_yield(dataset_params.rare_file, dataset_params, output_params, fit_params, cut_string=None, mass_range=None)
    psi2s_eff = get_eff(dataset_params.rare_file, dataset_params, output_params, fit_params, denom=49245742, cut_string=None, mass_range=None, get_comps=True)
    print(f'psi2s yield = {psi2s_yield}')
    print(f'psi2s eff = {psi2s_eff}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    args = parser.parse_args()

    main(args)
