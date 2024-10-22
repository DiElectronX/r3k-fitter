import os
import sys
import yaml
import csv
import argparse
import pickle
from tqdm import tqdm
from uncertainties import ufloat
import numpy as np
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


def get_eff_jpsi_fit(cut_num, cut_denom, fit_defaults, dataset_params, output_params, fit_params, args, get_comps=False):
    args.mode = 'jpsi'
    fit_params.bdt_score_cut = cut_denom
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
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_data_mc_syst_jpsi_fit_data_denom.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    denom_yield, denom_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)
    _denom_yield = ufloat(denom_yield, denom_yield_err)

    cut = ROOT.TCut(f'{dataset_params.score_branch}>{cut_num}')
    dataset_data = dataset_data.reduce(cut.GetTitle())

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_data_mc_syst_jpsi_fit_data_num.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    num_yield, num_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)
    _num_yield = ufloat(num_yield, num_yield_err)
    eff = _num_yield / _denom_yield

    if get_comps:
        return eff.n, eff.std_dev, (num_yield, num_yield_err, denom_yield, denom_yield_err)
    else:
        return eff.n, eff.std_dev


def get_eff_psi2s_fit(cut_num, cut_denom, fit_defaults, dataset_params, output_params, fit_params, args, get_comps=False):
    args.mode = 'psi2s'
    fit_params.bdt_score_cut = cut_denom
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
  
    # Use J/Psi MC for signal template 
    b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)
    model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
    model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    model_sig_template.fit_model = model_sig_template.sig_pdf
    model_sig_template.fit(dataset_mc, printlevel=printlevel)
    
    # Use K* MC for background template
    b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_psi2s_pion_file)
    b_mass_branch, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_pion_file)#, extra_weight=.1)
    b_mass_branch, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_kaon_file)#, extra_weight=.1)
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
        # 'dcb_ratio_constraint' : ROOT.RooGaussian('dcb_ratio_constraint', 'dcb_ratio_constraint', dcb_ratio, ROOT.RooFit.RooConst(dcb_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_data_mc_syst_psi2s_fit_data_denom.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    denom_yield, denom_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)
    _denom_yield = ufloat(denom_yield, denom_yield_err)

    cut = ROOT.TCut(f'{dataset_params.score_branch}>{cut_num}')
    dataset_data = dataset_data.reduce(cut.GetTitle())

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_data_mc_syst_psi2s_fit_data_num.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    num_yield, num_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)
    _num_yield = ufloat(num_yield, num_yield_err)
    eff = _num_yield / _denom_yield

    if get_comps:
        return eff.n, eff.std_dev, (num_yield, num_yield_err, denom_yield, denom_yield_err)
    else:
        return eff.n, eff.std_dev


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    cut_nominal = 4.1
    cut_loose = -10.

    args.mode = 'jpsi'
    set_mode(dataset_params, output_params, fit_params, args)

    eff_jpsi_mc, eff_jpsi_mc_err, eff_jpsi_mc_comps = get_eff_cut_and_count(
        dataset_params.jpsi_file, 
        dataset_params.tree_name, 
        f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(cut_nominal)}', 
        denom_string=f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(cut_loose)}', 
        weights='trig_wgt', 
        get_comps=True,
    )

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
    }
    eff_jpsi_data, eff_jpsi_data_err, eff_jpsi_data_comps = get_eff_jpsi_fit(
        cut_nominal,
        cut_loose,
        jpsi_fit_defaults,
        dataset_params,
        output_params,
        fit_params,
        args,
        get_comps=True,
    )
    
    args.mode = 'psi2s'
    set_mode(dataset_params, output_params, fit_params, args)

    eff_psi2s_mc, eff_psi2s_mc_err, eff_psi2s_mc_comps = get_eff_cut_and_count(
        dataset_params.psi2s_file, 
        dataset_params.tree_name, 
        f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(cut_nominal)}', 
        denom_string=f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && (Bmass > 5.1 && Bmass < 5.4) && bdt_score>{str(cut_loose)}', 
        weights='trig_wgt', 
        get_comps=True,
    )

    psi2s_fit_defaults = {
        'dcb1_alpha1_sig_pdf' : [2.53,1,10],
        'dcb1_alpha2_sig_pdf' : [3.11,1,10],
        'dcb1_coeff_sig_pdf'  : [1109.97,0,200000],
        'dcb1_mean_sig_pdf'   : [5.17,5.1,5.4],
        'dcb1_n1_sig_pdf'     : [1.14,1,100],
        'dcb1_n2_sig_pdf'     : [3.47,1,100],
        'dcb1_sigma_sig_pdf'  : [0.03,.001,.1],
        'dcb2_alpha1_sig_pdf' : [2.65,1,10],
        'dcb2_alpha2_sig_pdf' : [3.01,1.,10.],
        'dcb2_coeff_sig_pdf'  : [8475.42,2,200000],
        'dcb2_mean_sig_pdf'   : [5.27,5.1,5.4],
        'dcb2_n1_sig_pdf'     : [34.11,1,100],
        'dcb2_n2_sig_pdf'     : [2.33,1,100],
        'dcb2_sigma_sig_pdf'  : [0.05,.001,.1],
        'exp_slope_comb_bkg_pdf'   : [-1.04, -5., 5.],
        'kde_mirror_sig_pdf': 'NoMirror',
        'kde_rho_sig_pdf':    2,
        'kde_mirror_part_bkg_pdf_1': 'MirrorLeft',
        'kde_rho_part_bkg_pdf_1':    2,
        'kde_mirror_part_bkg_pdf_2': 'MirrorLeft',
        'kde_rho_part_bkg_pdf_2':    2,
    }
    eff_psi2s_data, eff_psi2s_data_err, eff_psi2s_data_comps = get_eff_psi2s_fit(
        cut_nominal,
        cut_loose,
        psi2s_fit_defaults,
        dataset_params,
        output_params,
        fit_params,
        args,
        get_comps=True,
    )

    _eff_jpsi_data = ufloat(eff_jpsi_data, eff_jpsi_data_err)
    _eff_jpsi_mc = ufloat(eff_jpsi_mc, eff_jpsi_mc_err)
    _jpsi_ratio = _eff_jpsi_data / _eff_jpsi_mc
    
    _eff_psi2s_data = ufloat(eff_psi2s_data, eff_psi2s_data_err)
    _eff_psi2s_mc = ufloat(eff_psi2s_mc, eff_psi2s_mc_err)
    _psi2s_ratio = _eff_psi2s_data / _eff_psi2s_mc

    _double_ratio = _jpsi_ratio / _psi2s_ratio

    print(f'J/\u03C8 Efficiency Ratio = {_jpsi_ratio.n} \u00B1 {_jpsi_ratio.std_dev}')   
    print(f'\u03C8(2s) Efficiency Ratio = {_psi2s_ratio.n} \u00B1 {_psi2s_ratio.std_dev}')   
    print(f'Double Ratio = {_double_ratio.n} \u00B1 {_double_ratio.std_dev}')   

    output_dict = {
        'double_ratio'           : {'value' : _double_ratio.n, 'uncertainty' : _double_ratio.std_dev},
        'jpsi_ratio'             : {'value' : _jpsi_ratio.n, 'uncertainty' : _jpsi_ratio.std_dev},
        'jpsi_eff_data'          : {'value' : eff_jpsi_data, 'uncertainty' : eff_jpsi_data_err},
        'jpsi_eff_mc'            : {'value' : eff_jpsi_mc, 'uncertainty' : eff_jpsi_mc_err},
        'yield_jpsi_num_data'    : {'value' : eff_jpsi_data_comps[0], 'uncertainty' : eff_jpsi_data_comps[1]},
        'yield_jpsi_denom_data'  : {'value' : eff_jpsi_data_comps[2], 'uncertainty' : eff_jpsi_data_comps[3]},
        'yield_jpsi_num_mc'      : {'value' : eff_jpsi_mc_comps[0], 'uncertainty' : eff_jpsi_mc_comps[1]},
        'yield_jpsi_denom_mc'    : {'value' : eff_jpsi_mc_comps[2], 'uncertainty' : eff_jpsi_mc_comps[3]},
        'psi2s_ratio'            : {'value' : _psi2s_ratio.n, 'uncertainty' : _psi2s_ratio.std_dev},
        'psi2s_eff_data'         : {'value' : eff_psi2s_data, 'uncertainty' : eff_psi2s_data_err},
        'psi2s_eff_mc'           : {'value' : eff_psi2s_mc, 'uncertainty' : eff_psi2s_mc_err},
        'yield_psi2s_num_data'   : {'value' : eff_psi2s_data_comps[0], 'uncertainty' : eff_psi2s_data_comps[1]},
        'yield_psi2s_denom_data' : {'value' : eff_psi2s_data_comps[2], 'uncertainty' : eff_psi2s_data_comps[3]},
        'yield_psi2s_num_mc'     : {'value' : eff_psi2s_mc_comps[0], 'uncertainty' : eff_psi2s_mc_comps[1]},
        'yield_psi2s_denom_mc'   : {'value' : eff_psi2s_mc_comps[2], 'uncertainty' : eff_psi2s_mc_comps[3]},
    }
    with open('bdt_data_mc_systematic_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'value', 'uncertainty'])
        for name, info in output_dict.items():
            writer.writerow([name, info['value'], info['uncertainty']])
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    args = parser.parse_args()

    main(args)
