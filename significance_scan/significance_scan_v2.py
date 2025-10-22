import os
import sys
import yaml
import argparse
import pickle
from tqdm import tqdm
from tqdm.contrib import tzip
from uncertainties import ufloat
import numpy as np
from pprint import pprint
from pathlib import Path
import ROOT
import ROOT.RooFit as rf

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from utils import *
from do_fit import *
from do_mc_yields_and_effs import *

BR_BKEE = 4.5E-7 # Using Mu BR from PDG instead of electron (5.6E-7)
BR_BJPSI = 1.02E-3
BR_JPSIEE = 5.97E-2


def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    if isinstance(iterable,zip):
        unzipped = list(iterable)
        return iterable if args.verbose else tqdm(unzipped, total=len(unzipped), unit=unit)
    else:
        return iterable if args.verbose else tqdm(iterable, total=len(iterable), unit=unit)


def estimate_lowq2_signal(n_jpsik, eff_eek, eff_jpsik, n_jpsik_err=None, eff_eek_err=None, eff_jpsik_err=None):
    try:
        n_signal = n_jpsik * BR_BKEE * eff_eek / (BR_BJPSI * BR_JPSIEE * eff_jpsik)
    except ZeroDivisionError:
        n_signal = 0
        n_signal_err = 0

    if n_signal and (n_jpsik_err is not None) and (eff_eek_err is not None) and (eff_jpsik_err is not None):
        _n_jpsik = ufloat(n_jpsik,n_jpsik_err)
        _eff_eek = ufloat(eff_eek,eff_eek_err)
        _eff_jpsik = ufloat(eff_jpsik,eff_jpsik_err)
        _BR_BKEE = ufloat(BR_BKEE,0)
        _BR_BJPSI = ufloat(BR_BJPSI,0)
        _BR_JPSIEE = ufloat(BR_JPSIEE,0)

        try:
            n_signal = _n_jpsik * _BR_BKEE * _eff_eek / (_BR_BJPSI * _BR_JPSIEE * _eff_jpsik)
        except ZeroDivisionError:
            return 0, 0

        return n_signal.n, n_signal.std_dev
    else:
        return n_signal


def estimate_significance(n_sig, n_bkg, n_sig_err=None, n_bkg_err=None):
    significance = n_sig / np.sqrt(n_sig + n_bkg)
    if (n_sig_err is not None) and (n_bkg_err is not None):
        significance_err = 0.5 * np.sqrt((n_sig**2*n_bkg_err**2+n_sig_err**2*(2*n_bkg+n_sig)**2)/(n_sig + n_bkg)**3)
        return significance, significance_err
    else:
        return significance

def significance_scan(dataset_params, output_params, fit_params, args):

    outputs = {
        'score' : np.array([]),
        'significance' : np.array([]),
        'significance_err' : np.array([]),
        'n_eek_bkg' : np.array([]),
        'n_eek_bkg_err' : np.array([]),
        'n_eek_sig' : np.array([]),
        'n_eek_sig_err' : np.array([]),
        'n_jpsik_sig' : np.array([]),
        'n_jpsik_sig_err' : np.array([]),
        'eff_eek' : np.array([]),
        'eff_eek_err' : np.array([]),
        'eff_jpsik' : np.array([]),
        'eff_jpsik_err' : np.array([]),
    }

    scan_range = np.linspace(0,8,30)

    for bdt_cut in loop_wrapper(scan_range, args, title='Calculating Significances'):
        fit_params.bdt_score_cut = bdt_cut

        args.mode = 'lowq2'
        cut_string = f'(Mll > 1.05 && Mll < 2.45) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(bdt_cut)}'
        eff_eek, eff_eek_err = get_eff(dataset_params.rare_file, dataset_params, output_params, fit_params, args, cut_string=cut_string)
        lowq2_yields = do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, get_yields=True, write=False ,toy_fit=False)

        args.mode = 'jpsi'
        cut_string = f'(Mll > 2.95 && Mll < 3.2) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(bdt_cut)}'
        eff_jpsik, eff_jpsik_err = get_eff(dataset_params.jpsi_file, dataset_params, output_params, fit_params, args, cut_string=cut_string)
        jpsi_yields = do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, get_yields=True, write=False)

        _n_lowq2_bkg = (ufloat(lowq2_yields['yield_comb_bkg'][0], lowq2_yields['yield_comb_bkg'][1])
                        + ufloat(lowq2_yields['yield_part_bkg'][0], lowq2_yields['yield_part_bkg'][1])
                        + ufloat(lowq2_yields['yield_jpsi_bkg'][0], lowq2_yields['yield_jpsi_bkg'][1])
        )

        n_lowq2_bkg, n_lowq2_bkg_err = _n_lowq2_bkg.n, _n_lowq2_bkg.s
        n_jpsi_sig, n_jpsi_sig_err = jpsi_yields['yield_sig'][0], jpsi_yields['yield_sig'][1]

        n_lowq2_sig, n_lowq2_sig_err = estimate_lowq2_signal(n_jpsi_sig, eff_eek, eff_jpsik, n_jpsik_err=n_jpsi_sig_err, eff_eek_err=eff_eek_err, eff_jpsik_err=eff_jpsik_err)
        significance, significance_err = estimate_significance(n_lowq2_sig, n_lowq2_bkg, n_sig_err=n_lowq2_sig_err, n_bkg_err=n_lowq2_bkg_err)

        outputs['score'] = np.append(outputs['score'], bdt_cut)
        outputs['significance'] = np.append(outputs['significance'], significance)
        outputs['significance_err'] = np.append(outputs['significance_err'], significance_err)
        outputs['n_eek_sig'] = np.append(outputs['n_eek_sig'], n_lowq2_sig)
        outputs['n_eek_sig_err'] = np.append(outputs['n_eek_sig_err'], n_lowq2_sig_err)
        outputs['n_eek_bkg'] = np.append(outputs['n_eek_bkg'], n_lowq2_bkg)
        outputs['n_eek_bkg_err'] = np.append(outputs['n_eek_bkg_err'], n_lowq2_bkg_err)
        outputs['n_jpsik_sig'] = np.append(outputs['n_jpsik_sig'], n_jpsi_sig)
        outputs['n_jpsik_sig_err'] = np.append(outputs['n_jpsik_sig_err'], n_jpsi_sig_err)
        outputs['eff_eek'] = np.append(outputs['eff_eek'], eff_eek)
        outputs['eff_eek_err'] = np.append(outputs['eff_eek_err'], eff_eek_err)
        outputs['eff_jpsik'] = np.append(outputs['eff_jpsik'], eff_jpsik)
        outputs['eff_jpsik_err'] = np.append(outputs['eff_jpsik_err'], eff_jpsik_err)

    path = Path('.') / 'significance_scan_data.pkl'
    with open(path,'wb') as pkl_file:
        pickle.dump(outputs, pkl_file)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    significance_scan(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-minos', '--minos', dest='minos', action='store_true', help='use MINOS minimizer')
    args = parser.parse_args()

    main(args)
