import os
import sys
import yaml
import argparse
import pickle
from tqdm import tqdm
from tqdm.contrib import tzip
from uncertainties import ufloat
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import ROOT
import ROOT.RooFit as rf

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from utils import *
from do_fit import *
from do_mc_yields_and_effs import *

ALLOWED_MODES = ['jpsi', 'psi2s']
INT_LUMI = 33.85
INT_LUMI_ERR = 0
XS_BB = 4.7E11
XS_BB_ERR = 0.
FRAG_FRAC = 0.4
BR_B_JPSIK = 1.02E-3
BR_B_JPSIK_ERR = 1.9E-5
BR_B_JPSIKSTAR = 1.43E-3
BR_B_JPSIKSTAR_ERR = 8E-5
BR_B0_JPSIK0STAR = 1.27E-3
BR_B0_JPSIK0STAR_ERR = 5E-5
BR_B_PSI2SK = 6.24E-4
BR_B_PSI2SK_ERR = 2.1E-5
BR_B_PSI2SKSTAR = 6.7E-4
BR_B_PSI2SKSTAR_ERR = 1.4E-4
BR_JPSI_EE = 5.97E-2
BR_JPSI_EE_ERR = 3.2E-4
BR_PSI2S_EE = 7.94E-3
BR_PSI2S_EE_ERR = 2.2E-4
BR_B0_PSI2SK0STAR = 5.9E-4
BR_B0_PSI2SK0STAR_ERR = 4E-5
BR_KSTAR_KPI0 = 0.3323
BR_KSTAR_KPI0_ERR = 0.
BR_KSTAR_K0PI = 0.6657
BR_KSTAR_K0PI_ERR = 0.
BR_K0STAR_KPI = 0.6657
BR_K0STAR_KPI_ERR = 0.
NEVTGEN_B_JPSIK = 506005157
NEVTGEN_B_JPSIKSTAR_KEE = 166228045
NEVTGEN_B_JPSIKSTAR_PIEE = 81224467
NEVTGEN_B0_JPSIK0STAR_KEE = 81083868
NEVTGEN_B0_JPSIK0STAR_PIEE = 81083868
NEVTGEN_B_PSI2SK = 49175225
# NEVTGEN_B_PSI2SKSTAR_KEE = 
NEVTGEN_B_PSI2SKSTAR_PIEE = 7244072
NEVTGEN_B0_PSI2SK0STAR_KEE = 783325359
NEVTGEN_B0_PSI2SK0STAR_PIEE = 783325359

_int_lumi = ufloat(INT_LUMI, INT_LUMI_ERR)
_xs_bb = ufloat(XS_BB, XS_BB_ERR)
_frag_frac = ufloat(FRAG_FRAC, 0.)
_br_b_jpsik = ufloat(BR_B_JPSIK,BR_B_JPSIK_ERR)
_br_b_jpsikstar = ufloat(BR_B_JPSIKSTAR,BR_B_JPSIKSTAR_ERR)
_br_b_jpsik0star = ufloat(BR_B0_JPSIK0STAR,BR_B0_JPSIK0STAR_ERR)
_br_b_psi2sk = ufloat(BR_B_PSI2SK,BR_B_PSI2SK_ERR)
_br_b_psi2skstar = ufloat(BR_B_PSI2SKSTAR,BR_B_PSI2SKSTAR_ERR)
_br_b_psi2sk0star = ufloat(BR_B0_PSI2SK0STAR,BR_B0_PSI2SK0STAR_ERR)
_br_jpsi_ee = ufloat(BR_JPSI_EE, BR_JPSI_EE_ERR)
_br_psi2s_ee = ufloat(BR_PSI2S_EE, BR_PSI2S_EE_ERR)
_br_kstar_kee = ufloat(BR_KSTAR_KPI0,BR_KSTAR_KPI0_ERR)
_br_kstar_piee = ufloat(BR_KSTAR_K0PI,BR_KSTAR_K0PI_ERR)
_br_k0star_kee = ufloat(BR_K0STAR_KPI,BR_K0STAR_KPI_ERR)
_br_k0star_piee = ufloat(BR_K0STAR_KPI,BR_K0STAR_KPI_ERR)

_xs_b_jpsik           = _xs_bb * _frag_frac * (1-(1-(_br_b_jpsik*_br_jpsi_ee))**2)
_xs_b_jpsikstar_kee   = _xs_bb * _frag_frac * (1-(1-(_br_b_jpsikstar*_br_kstar_kee*_br_jpsi_ee))**2)
_xs_b_jpsikstar_piee  = _xs_bb * _frag_frac * (1-(1-(_br_b_jpsikstar*_br_kstar_piee*_br_jpsi_ee))**2)
_xs_b_jpsik0star_kee  = _xs_bb * _frag_frac * (1-(1-(_br_b_jpsik0star*_br_k0star_kee*_br_jpsi_ee))**2)
_xs_b_jpsik0star_piee = _xs_bb * _frag_frac * (1-(1-(_br_b_jpsik0star*_br_k0star_piee*_br_jpsi_ee))**2)

_xs_b_psi2sk           = _xs_bb * _frag_frac * (1-(1-(_br_b_psi2sk*_br_psi2s_ee))**2)
_xs_b_psi2skstar_kee   = _xs_bb * _frag_frac * (1-(1-(_br_b_psi2skstar*_br_kstar_kee*_br_psi2s_ee))**2)
_xs_b_psi2skstar_piee  = _xs_bb * _frag_frac * (1-(1-(_br_b_psi2skstar*_br_kstar_piee*_br_psi2s_ee))**2)
_xs_b_psi2sk0star_kee  = _xs_bb * _frag_frac * (1-(1-(_br_b_psi2sk0star*_br_k0star_kee*_br_psi2s_ee))**2)
_xs_b_psi2sk0star_piee = _xs_bb * _frag_frac * (1-(1-(_br_b_psi2sk0star*_br_k0star_piee*_br_psi2s_ee))**2)

_sf_b_jpsik = _int_lumi * _xs_b_jpsik / NEVTGEN_B_JPSIK
_sf_b_jpsikstar_kee = _int_lumi * _xs_b_jpsikstar_kee / NEVTGEN_B_JPSIKSTAR_KEE
_sf_b_jpsikstar_piee = _int_lumi * _xs_b_jpsikstar_piee / NEVTGEN_B_JPSIKSTAR_PIEE
_sf_b_jpsik0star_kee = _int_lumi * _xs_b_jpsik0star_kee / NEVTGEN_B0_JPSIK0STAR_KEE
_sf_b_jpsik0star_piee = _int_lumi * _xs_b_jpsik0star_piee / NEVTGEN_B0_JPSIK0STAR_PIEE

_sf_b_psi2sk = _int_lumi * _xs_b_psi2sk / NEVTGEN_B_PSI2SK
#_sf_b_psi2skstar_kee = _int_lumi * _xs_b_psi2skstar_kee / NEVTGEN_B_PSI2SKSTAR_KEE
_sf_b_psi2skstar_piee = _int_lumi * _xs_b_psi2skstar_piee / NEVTGEN_B_PSI2SKSTAR_PIEE
_sf_b_psi2sk0star_kee = _int_lumi * _xs_b_psi2sk0star_kee / NEVTGEN_B0_PSI2SK0STAR_KEE
_sf_b_psi2sk0star_piee = _int_lumi * _xs_b_psi2sk0star_piee / NEVTGEN_B0_PSI2SK0STAR_PIEE


def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    if isinstance(iterable,zip):
        unzipped = list(iterable)
        return iterable if args.verbose else tqdm(unzipped, total=len(unzipped), unit=unit)
    else:
        return iterable if args.verbose else tqdm(iterable, total=len(iterable), unit=unit)


def jpsi_kstar_k_scan(dataset_params, output_params, fit_params, args):
    set_mode(dataset_params,output_params,fit_params,args)
    scan_range = np.linspace(1,7,5)
    mass_window = fit_params.full_mass_range
    scan_plot_path = Path('.') / 'kstar_scan_plots'
    output_params.output_dir = scan_plot_path

    outputs = {
        'metadata' : {
            'mass_window' : mass_window,
            'nominal_bdt_cut' : fit_params.bdt_score_cut,
        },
        'score' : np.array([]),
        'n_jpsi_data_sig' : np.array([]),
        'n_jpsi_data_sig_err' : np.array([]),
        'n_jpsi_data_kstar' : np.array([]),
        'n_jpsi_data_kstar_err' : np.array([]),
        'n_jpsi_data_comb' : np.array([]),
        'n_jpsi_data_comb_err' : np.array([]),
        'n_jpsi_mc_sig' : np.array([]),
        'n_jpsi_mc_sig_err' : np.array([]),
        'n_jpsi_mc_kstar' : np.array([]),
        'n_jpsi_mc_kstar_err' : np.array([]),
        'r_jpsi_data' : np.array([]),
        'r_jpsi_data_err' : np.array([]),
        'r_jpsi_mc' : np.array([]),
        'r_jpsi_mc_err' : np.array([]),
    }

    for bdt_cut in loop_wrapper(scan_range, args, title='Calculating R(K*)'):
        fit_params.bdt_score_cut = bdt_cut

        _n_sig_mc = get_weighted_yield_from_dict(
            {
                'b+_jpsik+' : {
                    'file'  : dataset_params.jpsi_file, 
                    'label' : '$B^{+} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{+}$',
                    'sf'    : _sf_b_jpsik,
                },
            }, 
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            cut_string=None, 
            mass_range=mass_window,
            plot=scan_plot_path / f'jpsi_sig_mc_bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}.pdf'
        )

        _n_kstar_mc = get_weighted_yield_from_dict(
            {
                'b+_jpsik*+_kee' : {
                    'file'  : dataset_params.kstar_jpsi_kaon_file, 
                    'label' : '$B^{+} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{*+} \ [K^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_jpsikstar_kee,
                },
                'b+_jpsik*+_piee' : {
                    'file'  : dataset_params.kstar_jpsi_pion_file, 
                    'label' : '$B^{+} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{*+} \ [\\pi^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_jpsikstar_piee,
                },
                'b0_jpsik*0_kee' : {
                    'file'  : dataset_params.k0star_jpsi_kaon_file, 
                    'label' : '$B^{0} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{*0} \ [K^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_jpsik0star_kee,
                },
                'b0_jpsik*0_piee' : {
                    'file'  : dataset_params.k0star_jpsi_pion_file, 
                    'label' : '$B^{0} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{*0} \ [\\pi^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_jpsik0star_piee,
                },
            },
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            cut_string=None, 
            mass_range=mass_window,
            plot=scan_plot_path / f'jpsi_kstar_mc_bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}.pdf'
        )
        _r_jpsi_mc = _n_kstar_mc / _n_sig_mc
        data_fit_dict = do_jpsi_control_region_fit(
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            get_yields=True,
            # extra_text=f'\nBDT Score > {str(round(bdt_cut,2))}',
            file_label=f'bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}',
        )

        _n_sig_data = ufloat(data_fit_dict['yield_sig'], data_fit_dict['yield_sig_err'])
        _n_kstar_data = ufloat(data_fit_dict['yield_part_bkg_kstar'], data_fit_dict['yield_part_bkg_kstar_err'])
        _n_comb_data = ufloat(data_fit_dict['yield_comb_bkg'], data_fit_dict['yield_comb_bkg_err'])
        _r_jpsi_data = _n_kstar_data / _n_sig_data

        outputs['score'] = np.append(outputs['score'], bdt_cut)
        outputs['n_jpsi_data_sig'] = np.append(outputs['n_jpsi_data_sig'], _n_sig_data.n)
        outputs['n_jpsi_data_sig_err'] = np.append(outputs['n_jpsi_data_sig_err'], _n_sig_data.std_dev)
        outputs['n_jpsi_data_kstar'] = np.append(outputs['n_jpsi_data_kstar'], _n_kstar_data.n)
        outputs['n_jpsi_data_kstar_err'] = np.append(outputs['n_jpsi_data_kstar_err'], _n_kstar_data.std_dev)
        outputs['n_jpsi_data_comb'] = np.append(outputs['n_jpsi_data_comb'], _n_comb_data.n)
        outputs['n_jpsi_data_comb_err'] = np.append(outputs['n_jpsi_data_comb_err'], _n_comb_data.std_dev)
        outputs['n_jpsi_mc_sig'] = np.append(outputs['n_jpsi_mc_sig'], _n_sig_mc.n)
        outputs['n_jpsi_mc_sig_err'] = np.append(outputs['n_jpsi_mc_sig_err'], _n_sig_mc.std_dev)
        outputs['n_jpsi_mc_kstar'] = np.append(outputs['n_jpsi_mc_kstar'], _n_kstar_mc.n)
        outputs['n_jpsi_mc_kstar_err'] = np.append(outputs['n_jpsi_mc_kstar_err'], _n_kstar_mc.std_dev)
        
        outputs['r_jpsi_data'] = np.append(outputs['r_jpsi_data'], _r_jpsi_data.n)
        outputs['r_jpsi_data_err'] = np.append(outputs['r_jpsi_data_err'], _r_jpsi_data.std_dev)
        outputs['r_jpsi_mc'] = np.append(outputs['r_jpsi_mc'], _r_jpsi_mc.n)
        outputs['r_jpsi_mc_err'] = np.append(outputs['r_jpsi_mc_err'], _r_jpsi_mc.std_dev)
    
    path = Path('.') / 'jpsi_kstar_k_scan_data.pkl'
    with open(path,'wb') as pkl_file:
        pickle.dump(outputs, pkl_file)

def psi2s_kstar_k_scan(dataset_params, output_params, fit_params, args):
    set_mode(dataset_params,output_params,fit_params,args)
    scan_range = np.linspace(1,7,5)
    mass_window = None #[5.1,5.4]
    scan_plot_path = Path('.') / 'kstar_scan_plots'
    output_params.output_dir = scan_plot_path

    outputs = {
        'metadata' : {'mass_window' : mass_window},
        'score' : np.array([]),
        'n_psi2s_data_sig' : np.array([]),
        'n_psi2s_data_sig_err' : np.array([]),
        'n_psi2s_data_kstar' : np.array([]),
        'n_psi2s_data_kstar_err' : np.array([]),
        'n_psi2s_data_comb' : np.array([]),
        'n_psi2s_data_comb_err' : np.array([]),
        'n_psi2s_mc_sig' : np.array([]),
        'n_psi2s_mc_sig_err' : np.array([]),
        'n_psi2s_mc_kstar' : np.array([]),
        'n_psi2s_mc_kstar_err' : np.array([]),
        'r_psi2s_data' : np.array([]),
        'r_psi2s_data_err' : np.array([]),
        'r_psi2s_mc' : np.array([]),
        'r_psi2s_mc_err' : np.array([]),
    }

    for bdt_cut in loop_wrapper(scan_range, args, title='Calculating R(K*)'):
        fit_params.bdt_score_cut = bdt_cut

        _n_sig_mc = get_weighted_yield_from_dict(
            {
                'b+_psi2sk+' : {
                    'file'  : dataset_params.psi2s_file, 
                    'label' : '$B^{+} \\rightarrow J/\\psi(\\rightarrow e^{+}e^{-})K^{+}$',
                    'sf'    : _sf_b_psi2sk,
                },
            }, 
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            cut_string=None, 
            mass_range=mass_window,
            plot=scan_plot_path / f'psi2s_sig_mc_bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}.pdf'
        )

        _n_kstar_mc = get_weighted_yield_from_dict(
            {
                'b+_psi2sk*+_kee' : {
                    'file'  : dataset_params.k0star_psi2s_kaon_file, 
                    'label' : '$B^{+} \\rightarrow \\psi(2S)(\\rightarrow e^{+}e^{-})K^{*+} \ [K^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_psi2sk0star_kee*0.5,
                },
                'b+_psi2sk*+_piee' : {
                    'file'  : dataset_params.kstar_psi2s_pion_file, 
                    'label' : '$B^{+} \\rightarrow \\psi(2S)(\\rightarrow e^{+}e^{-})K^{*+} \ [\\pi^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_psi2skstar_piee,
                },
                'b0_psi2sk*0_kee' : {
                    'file'  : dataset_params.k0star_psi2s_kaon_file, 
                    'label' : '$B^{0} \\rightarrow \\psi(2S)(\\rightarrow e^{+}e^{-})K^{*0} \ [K^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_psi2sk0star_kee,
                },
                'b0_psi2sk*0_piee' : {
                    'file'  : dataset_params.k0star_psi2s_pion_file, 
                    'label' : '$B^{0} \\rightarrow \\psi(2S)(\\rightarrow e^{+}e^{-})K^{*0} \ [\\pi^{+}e^{+}e^{-}$ Candidate]',
                    'sf'    : _sf_b_psi2sk0star_piee,
                },
            },
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            cut_string=None, 
            mass_range=mass_window,
            plot=scan_plot_path / f'psi2s_kstar_mc_bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}.pdf'
        )
        _r_psi2s_mc = _n_kstar_mc / _n_sig_mc

        data_fit_dict = do_psi2s_control_region_fit(
            dataset_params, 
            output_params, 
            fit_params, 
            args, 
            get_yields=True,
            # extra_text=f'\nBDT Score > {str(round(bdt_cut,2))}',
            file_label=f'bdt_cut_{str(round(bdt_cut,2)).replace(".","p")}',
        )

        _n_sig_data = ufloat(data_fit_dict['yield_sig'], data_fit_dict['yield_sig_err'])
        _n_kstar_data = ufloat(data_fit_dict['yield_part_bkg_kstar'], data_fit_dict['yield_part_bkg_kstar_err'])
        _n_comb_data = ufloat(data_fit_dict['yield_comb_bkg'], data_fit_dict['yield_comb_bkg_err'])
        _r_psi2s_data = _n_kstar_data / _n_sig_data

        outputs['score'] = np.append(outputs['score'], bdt_cut)
        outputs['n_psi2s_data_sig'] = np.append(outputs['n_psi2s_data_sig'], _n_sig_data.n)
        outputs['n_psi2s_data_sig_err'] = np.append(outputs['n_psi2s_data_sig_err'], _n_sig_data.std_dev)
        outputs['n_psi2s_data_kstar'] = np.append(outputs['n_psi2s_data_kstar'], _n_kstar_data.n)
        outputs['n_psi2s_data_kstar_err'] = np.append(outputs['n_psi2s_data_kstar_err'], _n_kstar_data.std_dev)
        outputs['n_psi2s_data_comb'] = np.append(outputs['n_psi2s_data_comb'], _n_comb_data.n)
        outputs['n_psi2s_data_comb_err'] = np.append(outputs['n_psi2s_data_comb_err'], _n_comb_data.std_dev)
        outputs['n_psi2s_mc_sig'] = np.append(outputs['n_psi2s_mc_sig'], _n_sig_mc.n)
        outputs['n_psi2s_mc_sig_err'] = np.append(outputs['n_psi2s_mc_sig_err'], _n_sig_mc.std_dev)
        outputs['n_psi2s_mc_kstar'] = np.append(outputs['n_psi2s_mc_kstar'], _n_kstar_mc.n)
        outputs['n_psi2s_mc_kstar_err'] = np.append(outputs['n_psi2s_mc_kstar_err'], _n_kstar_mc.std_dev)
        
        outputs['r_psi2s_data'] = np.append(outputs['r_psi2s_data'], _r_psi2s_data.n)
        outputs['r_psi2s_data_err'] = np.append(outputs['r_psi2s_data_err'], _r_psi2s_data.std_dev)
        outputs['r_psi2s_mc'] = np.append(outputs['r_psi2s_mc'], _r_psi2s_mc.n)
        outputs['r_psi2s_mc_err'] = np.append(outputs['r_psi2s_mc_err'], _r_psi2s_mc.std_dev)
    
    path = Path('.') / 'psi2s_kstar_k_scan_data.pkl'
    with open(path,'wb') as pkl_file:
        pickle.dump(outputs, pkl_file)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    if args.mode=='all':
        jpsi_kstar_k_scan(dataset_params, output_params, fit_params, args)
        psi2s_kstar_k_scan(dataset_params, output_params, fit_params, args)
    elif args.mode=='jpsi':
        jpsi_kstar_k_scan(dataset_params, output_params, fit_params, args)
    elif args.mode=='psi2s':
        psi2s_kstar_k_scan(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which scan to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    args = parser.parse_args()

    main(args)
