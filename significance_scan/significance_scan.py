import os
import yaml
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from pprint import pprint
import ROOT
import ROOT.RooFit as rf

from fit_models import FitModel
from utils import *

BR_BKEE = 4.5E-7 # Using Mu BR from PDG instead of electron (5.6E-7)
BR_BJPSI = 1.02E-3
BR_JPSIEE = 5.97E-2

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

def estimateKEE(x, eff_eek, eff_jpsik):
    try:
        return x * BR_BKEE * eff_eek / (BR_BJPSI * BR_JPSIEE * eff_jpsik)
    except ZeroDivisionError:
        return 0

def integrate(var, model, model_yield, integral_range, fit_result):
    var.setRange('int_range', *integral_range)
    integral_unscaled = model.createIntegral(ROOT.RooArgSet(var),ROOT.RooArgSet(var),'int_range')
    integral = integral_unscaled.getVal() * model_yield.getVal() if (integral_unscaled.getVal()*model_yield.getVal())>0.1 else 0
    integral_err = integral * np.linalg.norm([integral_unscaled.getPropagatedError(fit_result, ROOT.RooArgSet(var))/integral_unscaled.getVal(), model_yield.getError()/model_yield.getVal()])

    return integral, integral_err

def do_lowq2_quickfit(bdt_cut, dataset_params, output_params, fit_params, args, integral=None):
    args.mode = 'lowq2'
    fit_params.bdt_score_cut = bdt_cut
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
   
    fit_defaults = {
        'dcb_mean_sig_pdf'    : [5.27, 5.2, 5.3],
        'dcb_sigma_sig_pdf'   : [0.046, 0.03, 0.07],
        'dcb_alphaL_sig_pdf'  : [0.77, 0.5, 2.],
        'dcb_nL_sig_pdf'      : [2.07, 1., 10.],
        'dcb_alphaR_sig_pdf'  : [1.62, 0.5, 3.],
        'dcb_nR_sig_pdf'      : [3.48, 1., 10.],
        'exp_slope_bkg_pdf'   : [-2.3, -5., 5.],
        'erfc_mean_bkg_pdf'   : [5., 4, 7],
        'erfc_sigma_bkg_pdf'  : [0.03, 0.001, 100.],
        'poly_offset_bkg_pdf' : [5.3, -10, 10], 
        'poly_a0_bkg_pdf'     : [-4, -100, 100], 
        'poly_a1_bkg_pdf'     : [-3, -100, 100], 
        'poly_a2_bkg_pdf'     : [22, -100, 100], 
        'poly_a3_bkg_pdf'     : [17, -100, 100], 
        'poly_a4_bkg_pdf'     : [-49, -100, 100], 
    }


    '''
    b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)
    sig_model_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
    sig_model_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    sig_model_template.fit_model = sig_model_template.sig_pdf
        
    sig_model_template.fit(dataset_mc, printlevel=printlevel)
    params = sig_model_template.fit_result.floatParsFinal()

    for param in params:
        for key in fit_defaults.keys(): 
            if key in param.GetName():
                fit_defaults[key] = param.getVal()
    '''

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
    model_final.fit(dataset_data, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    for param in params:
        for key in fit_defaults.keys(): 
            if key in param.GetName():
                fit_defaults[key] = param.getVal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join('scan_fits','lowq2_fit_bdt>'+str(bdt_cut)+'.pdf'),
        fit_components = [
            # model_final.sig_pdf,
            # model_final.bkg_pdf,
        ],
        fit_range='full',
        fit_norm_range='sb1,sb2',
        file_formats=['pdf']
    )

    if integral:
        final_yield, _ = integrate(b_mass_branch, model_final.fit_model, bkg_coeff, integral, model_final.fit_result)
    else:
        final_yield = bkg_coeff.getVal()

    return final_yield

def do_jpsi_quickfit(bdt_cut, dataset_params, output_params, fit_params, args, integral=None):
    args.mode = 'jpsi'
    fit_params.bdt_score_cut = bdt_cut
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
   
    fit_defaults = {
        'dcb1_coeff_sig_pdf'  : [1., 0., 100000],
        'dcb1_mean_sig_pdf'   : [5.23, 5.1, 5.4],
        'dcb1_sigma_sig_pdf'  : [0.068, 0.01, 0.5],
        'dcb1_alpha1_sig_pdf' : [2.85, 1., 10.],
        'dcb1_n1_sig_pdf'     : [1.32, 0.1, 100.],
        'dcb1_alpha2_sig_pdf' : [3.18, 1., 10.],
        'dcb1_n2_sig_pdf'     : [59.8, 0.1, 100.],
        'dcb2_coeff_sig_pdf'  : [1., 0., 100000],
        'dcb2_mean_sig_pdf'   : [5.28, 5.1, 5.4],
        'dcb2_sigma_sig_pdf'  : [0.04, 0.01, 0.5],
        'dcb2_alpha1_sig_pdf' : [5.94, 1., 6.],
        'dcb2_n1_sig_pdf'     : [65.01, 0.5, 100.],
        'dcb2_alpha2_sig_pdf' : [2.66, 1., 6.],
        'dcb2_n2_sig_pdf'     : [2.52, 0.5, 100.],
        'exp_slope_bkg_pdf'   : [-2.3, -5., 5.],
        'erfc_mean_bkg_pdf'   : [5., 4, 7],
        'erfc_sigma_bkg_pdf'  : [0.03, 0.001, 100.],
        'poly_offset_bkg_pdf' : [5.3, -10, 10], 
        'poly_a0_bkg_pdf'     : [-4, -100, 100], 
        'poly_a1_bkg_pdf'     : [-3, -100, 100], 
        'poly_a2_bkg_pdf'     : [22, -100, 100], 
        'poly_a3_bkg_pdf'     : [17, -100, 100], 
        'poly_a4_bkg_pdf'     : [-49, -100, 100], 
    }


    '''
    b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)
    sig_model_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
    sig_model_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    sig_model_template.fit_model = sig_model_template.sig_pdf
        
    sig_model_template.fit(dataset_mc, printlevel=printlevel)
    params = sig_model_template.fit_result.floatParsFinal()

    for param in params:
        for key in fit_defaults.keys(): 
            if key in param.GetName():
                fit_defaults[key] = param.getVal()
    '''

    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    model_final.add_background_model('bkg_pdf', 'exp', fit_defaults, let_float=True)

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 52159, 0, dataset_data.numEntries())
    bkg_coeff = ROOT.RooRealVar('bkg_coeff'+fit_params.channel_label, 'Background PDF Coefficient', 42343, 0, dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            model_final.sig_pdf,
            model_final.bkg_pdf,
        ),
        ROOT.RooArgList(
            sig_coeff,
            bkg_coeff,
        ),
    )
    
    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    for param in params:
        for key in fit_defaults.keys(): 
            if key in param.GetName():
                fit_defaults[key] = param.getVal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join('scan_fits','jpsi_fit_bdt>'+str(bdt_cut)+'.pdf'),
        fit_components = [
            model_final.sig_pdf,
            model_final.bkg_pdf,
        ],
        file_formats=['pdf']
    )

    if integral:
        final_yield, _ = integrate(b_mass_branch, model_final.fit_model, sig_coeff, integral, model_final.fit_result)
    else:
        final_yield = sig_coeff.getVal()

    return final_yield


def significance_scan(dataset_params, output_params, fit_params, args):
    outputs = {
        'score' : np.array([])
        'significance' : np.array([]),
        'eff_eek' : np.array([]),
        'eff_eek_err' : np.array([]),
        'eff_jpsik' : np.array([]),
        'eff_jpsik_err' : np.array([]),
    }
    scan_range = tqdm(np.linspace(1,6,20))
    for bdt_cut in scan_range:
        n_lowq2_bkg = do_lowq2_quickfit(bdt_cut, dataset_params, output_params, fit_params, args, integral=(5.1,5.4))
        eff_eek, eff_eek_unc = get_eff_eek(bdt_cut, dataset_params, output_params, fit_params)
        n_jpsi_sig = do_jpsi_quickfit(bdt_cut, dataset_params, output_params, fit_params, args, integral=(5.1,5.4))
        eff_jpsik, eff_jpsik_unc = get_eff_jpsik(bdt_cut, dataset_params, output_params, fit_params)
        n_lowq2_sig = estimateKEE(n_jpsi_sig, eff_eek, eff_jpsik)
        significance = n_lowq2_sig / np.sqrt(n_lowq2_sig + n_lowq2_bkg)

        outputs['score'] = np.append(outputs['score'], score)
        outputs['significance'] = np.append(outputs['significance'], score)
        outputs['eff_eek'] = np.append(outputs['eff_eek'], score)
        outputs['eff_eek_err'] = np.append(outputs['eff_eek_err'], score)
        outputs['eff_jpsik'] = np.append(outputs['eff_jpsik'], score)
        outputs['eff_jpsik_err'] = np.append(outputs['eff_jpsik_err'], score)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    significance_scan(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    args = parser.parse_args()

    main(args)
