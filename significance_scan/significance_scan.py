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


def get_eff_eek(bdt_cut, dataset_params, output_params, fit_params, total_mc=500_000_000):
    args.mode = 'lowq2'
    set_mode(dataset_params, output_params, fit_params, args)
    df = ROOT.RDataFrame(dataset_params.tree_name, dataset_params.rare_file)
    cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(bdt_cut)}'
    k = df.Filter(cut_string).Sum(dataset_params.mc_weight_branch).GetValue()
    n = total_mc

    try:
        eff = k / n
        unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
        # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
    except ZeroDivisionError:
        eff = 0
        unc = 0

    return eff, unc


def get_eff_jpsik(bdt_cut, dataset_params, output_params, fit_params, total_mc=500_000_000):
    args.mode = 'jpsi'
    set_mode(dataset_params, output_params, fit_params, args)
    df = ROOT.RDataFrame(dataset_params.tree_name, dataset_params.jpsi_file)
    cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(bdt_cut)}'
    k = df.Filter(cut_string).Sum(dataset_params.mc_weight_branch).GetValue()
    n = total_mc

    try:
        eff = k / n
        unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
        # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
    except ZeroDivisionError:
        eff = 0
        unc = 0

    return eff, unc


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


def do_lowq2_quickfits(bdt_cuts, fit_defaults, dataset_params, output_params, fit_params, args, integral=None):
    args.mode = 'lowq2'
    fit_params.bdt_score_cut = 0.
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
   
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    # model_final.add_signal_model('sig_pdf', 'dcb', fit_defaults, let_float=True)
    model_final.add_background_model('bkg_pdf', 'exp', fit_defaults, let_float=True)

    # sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 0, 0, dataset_data.numEntries())
    bkg_coeff = ROOT.RooRealVar('bkg_coeff'+fit_params.channel_label, 'Background PDF Coefficient', 42343, 0, 2*dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            # model_final.sig_pdf,
            model_final.bkg_pdf,
        ),
        ROOT.RooArgList(
            # sig_coeff,
            bkg_coeff,
        ),
    )
    
    # Fit model to data
    final_yields = np.array([])
    final_yield_errs = np.array([])
    for bdt_cut in loop_wrapper(bdt_cuts, args, title='Performing Low-q2 Fit Scan'):
        cut = ROOT.TCut(f'{dataset_params.score_branch}>{bdt_cut}')
        dataset_data = dataset_data.reduce(cut.GetTitle())

        model_final.fit(dataset_data, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)
        params = model_final.fit_result.floatParsFinal()
       
        # Plot fit result
        model_final.plot_fit(
            b_mass_branch,
            dataset_data,
            Path('.') / 'scan_fits' / f'lowq2_fit_bdt>{str(round(bdt_cut,2))}.pdf',
            fit_components = [
                # model_final.sig_pdf,
                # model_final.bkg_pdf,
            ],
            fit_range='full',
            fit_norm_range='sb1,sb2',
            file_formats=['pdf']
        )

        if integral:
            final_yield, final_yield_err = integrate(b_mass_branch, model_final.fit_model, bkg_coeff, integral, model_final.fit_result)
        else:
            final_yield, final_yield_err = bkg_coeff.getVal(), bkg_coeff.getError()
        
        final_yields = np.append(final_yields,final_yield)
        final_yield_errs = np.append(final_yield_errs,final_yield_err)

    return final_yields, final_yield_errs


def do_jpsi_quickfits(bdt_cuts, fit_defaults, dataset_params, output_params, fit_params, args, integral=None):
    args.mode = 'jpsi'
    fit_params.bdt_score_cut = 1.
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
  
    # Use J/Psi MC for signal template 
    b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)
    model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
    model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    model_sig_template.fit_model = model_sig_template.sig_pdf
    model_sig_template.fit(dataset_mc, printlevel=printlevel)
    
    # Use K* MC for background template
    b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_jpsi_pion_file)
    b_mass_branch, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_pion_file)#, extra_weight=.1)
    b_mass_branch, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_kaon_file)#, extra_weight=.1)
    dataset_kstar_pion_comb = dataset_kstar_pion.Clone('dataset_kstar_pion_comb'+fit_params.channel_label)
    dataset_kstar_pion_comb.append(dataset_k0star_pion)

    model_kstar_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar_pion_comb, 'channel_label' : fit_params.channel_label})
    model_kstar_pion_template.add_background_model('part_bkg_pdf_1', 'kde', fit_defaults, let_float=True)
    model_kstar_pion_template.fit_model = model_kstar_pion_template.part_bkg_pdf_1

    model_kstar_kaon_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_kaon, 'channel_label' : fit_params.channel_label})
    model_kstar_kaon_template.add_background_model('part_bkg_pdf_2', 'kde', fit_defaults, let_float=True)
    model_kstar_kaon_template.fit_model = model_kstar_kaon_template.part_bkg_pdf_2

    # Fit composite model to data
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', model_sig_template.signal_models['sig_pdf'])
    model_final.add_background_model('comb_bkg_pdf', 'exp', fit_defaults, let_float=True)
    model_final.add_background_model('part_bkg_pdf_1', model_kstar_pion_template.background_models['part_bkg_pdf_1'])
    model_final.add_background_model('part_bkg_pdf_2', model_kstar_kaon_template.background_models['part_bkg_pdf_2'])

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 124395, 0, dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Background PDF Coefficient', 961773, 0, dataset_data.numEntries())
    part_bkg_1_coeff = ROOT.RooRealVar('part_bkg_1_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 1 Coefficient', 1569, 0, dataset_data.numEntries())
    part_bkg_2_coeff = ROOT.RooRealVar('part_bkg_2_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 2 Coefficient', 8943, 0, dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ),
        ROOT.RooArgList(
            sig_coeff,
            comb_bkg_coeff,
            part_bkg_1_coeff,
            part_bkg_2_coeff,
        )
    )
    
    part_bkg_1_coeff.setConstant(False)
    part_bkg_2_coeff.setConstant(False)
    kstar_ratio = ROOT.RooFormulaVar('kstar_ratio', 'Ratio of K* decay channels', '@0/@1', ROOT.RooArgList(part_bkg_1_coeff, part_bkg_2_coeff))
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_n1.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_n1.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_n2.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_n2.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_alpha1.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_alpha1.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_alpha2.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_alpha2.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_coeff.setConstant(False)
    dcb_ratio = ROOT.RooFormulaVar('dcb_ratio', 'Ratio of DCB Contributions', '@0/@1', ROOT.RooArgList(model_final.signal_models['sig_pdf'].dcb1_coeff, model_final.signal_models['sig_pdf'].dcb2_coeff))

    model_final.add_constraints({
        'kstar_ratio_constraint' : ROOT.RooGaussian('kstar_ratio_constraint', 'kstar_ratio_constraint', kstar_ratio, ROOT.RooFit.RooConst(kstar_ratio.getVal()), ROOT.RooFit.RooConst(.01)),
        'dcb_ratio_constraint' : ROOT.RooGaussian('dcb_ratio_constraint', 'dcb_ratio_constraint', dcb_ratio, ROOT.RooFit.RooConst(dcb_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
    })

    # Fit model to data
    final_yields = np.array([])
    final_yield_errs = np.array([])
    for bdt_cut in loop_wrapper(bdt_cuts, args, title='Performing J/Psi Fit Scan'):
        cut = ROOT.TCut(f'{dataset_params.score_branch}>{bdt_cut}')
        dataset_data = dataset_data.reduce(cut.GetTitle())

        model_final.fit(dataset_data, printlevel=printlevel)
        params = model_final.fit_result.floatParsFinal()
        
        # Plot fit result
        model_final.plot_fit(
            b_mass_branch,
            dataset_data,
            Path('.') / 'scan_fits' / f'jpsi_fit_bdt>{str(round(bdt_cut,2))}.pdf',
            fit_components = [
                model_final.sig_pdf,
                model_final.comb_bkg_pdf,
                model_final.part_bkg_pdf_1,
                model_final.part_bkg_pdf_2,
            ],
            file_formats=['pdf']
        )

        if integral:
            final_yield, final_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, integral, model_final.fit_result)
        else:
            final_yield, final_yield_err = sig_coeff.getVal(), sig_coeff.getError()

        final_yields = np.append(final_yields,final_yield)
        final_yield_errs = np.append(final_yield_errs,final_yield_err)

    return final_yields, final_yield_errs


def significance_scan(dataset_params, output_params, fit_params, args):

    lowq2_fit_defaults = {
        'dcb_mean_sig_pdf'    : [5.27, 5.2, 5.3],
        'dcb_sigma_sig_pdf'   : [0.046, 0.03, 0.07],
        'dcb_alphaL_sig_pdf'  : [0.77, 0.5, 2.],
        'dcb_nL_sig_pdf'      : [2.07, 1., 10.],
        'dcb_alphaR_sig_pdf'  : [1.62, 0.5, 3.],
        'dcb_nR_sig_pdf'      : [3.48, 1., 10.],
        'exp_slope_bkg_pdf'   : [-2.3, -5., 5.],
        'poly_offset_bkg_pdf' : [5.3, -10, 10], 
        'poly_a0_bkg_pdf'     : [-4, -100, 100], 
        'poly_a1_bkg_pdf'     : [-3, -100, 100], 
        'poly_a2_bkg_pdf'     : [22, -100, 100], 
        'poly_a3_bkg_pdf'     : [17, -100, 100], 
        'poly_a4_bkg_pdf'     : [-49, -100, 100], 
    }

    jpsi_fit_defaults = {
        'dcb1_alpha1_sig_pdf' : [2.78,1,10],
        'dcb1_alpha2_sig_pdf' : [3.40,1,10],
        'dcb1_coeff_sig_pdf'  : [94052.59,0,200000],
        'dcb1_mean_sig_pdf'   : [5.27,5.1,5.4],
        'dcb1_n1_sig_pdf'     : [17.54,1,100],
        'dcb1_n2_sig_pdf'     : [8.03,1,100],
        'dcb1_sigma_sig_pdf'  : [0.05,.001,.1],
        'dcb2_alpha1_sig_pdf' : [2.27,1,10],
        'dcb2_alpha2_sig_pdf' : [0.80,1.,10.],
        'dcb2_coeff_sig_pdf'  : [22640.84,2,200000],
        'dcb2_mean_sig_pdf'   : [5.16,5.1,5.4],
        'dcb2_n1_sig_pdf'     : [1.59,1,100],
        'dcb2_n2_sig_pdf'     : [28.95,1,100],
        'dcb2_sigma_sig_pdf'  : [0.04,.001,.1],
        'exp_slope_comb_bkg_pdf'   : [-3.45, -5., 5.],
        'kde_mirror_sig_pdf': 'NoMirror',
        'kde_rho_sig_pdf':    2,
        'kde_mirror_part_bkg_pdf_1': 'MirrorLeft',
        'kde_rho_part_bkg_pdf_1':    2,
        'kde_mirror_part_bkg_pdf_2': 'MirrorLeft',
        'kde_rho_part_bkg_pdf_2':    2,
        'poly_offset_bkg_pdf' : [5.3, -10, 10], 
        'poly_a0_bkg_pdf'     : [-4, -100, 100], 
        'poly_a1_bkg_pdf'     : [-3, -100, 100], 
        'poly_a2_bkg_pdf'     : [22, -100, 100], 
        'poly_a3_bkg_pdf'     : [17, -100, 100], 
        'poly_a4_bkg_pdf'     : [-49, -100, 100], 
    }

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

    scan_range = np.linspace(1,10,30)
    n_lowq2_bkg_list, n_lowq2_bkg_err_list = do_lowq2_quickfits(scan_range, lowq2_fit_defaults, dataset_params, output_params, fit_params, args, integral=(5.1,5.4))

    n_jpsi_sig_list, n_jpsi_sig_err_list = do_jpsi_quickfits(scan_range, jpsi_fit_defaults, dataset_params, output_params, fit_params, args, integral=(5.1,5.4))

    for bdt_cut,n_lowq2_bkg,n_lowq2_bkg_err,n_jpsi_sig,n_jpsi_sig_err in loop_wrapper(zip(scan_range,n_lowq2_bkg_list, n_lowq2_bkg_err_list,n_jpsi_sig_list, n_jpsi_sig_err_list), args, title='Calculating Significances'):
        eff_eek, eff_eek_err = get_eff_eek(bdt_cut, dataset_params, output_params, fit_params)
        eff_jpsik, eff_jpsik_err = get_eff_jpsik(bdt_cut, dataset_params, output_params, fit_params)
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
    args = parser.parse_args()

    main(args)
