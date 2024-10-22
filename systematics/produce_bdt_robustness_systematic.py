import os
import sys
import yaml
import csv
import argparse
import numpy as np
from pathlib import Path
import ROOT

sys.path.insert(1, str(Path('..').resolve()))
from fit_models import FitModel
from utils import *


def get_yield_jpsi_fit(cut_low, cut_opt, cut_high, fit_defaults, dataset_params, output_params, fit_params, args):
    args.mode = 'jpsi'
    fit_params.bdt_score_cut = cut_low
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
   
    # Incorporate contraints
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
        Path('.') / 'plots' / f'bdt_robustness_syst_jpsi_fit_low.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    low_yield, low_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)

    cut = ROOT.TCut(f'{dataset_params.score_branch}>{cut_opt}')
    dataset_data = dataset_data.reduce(cut.GetTitle())

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_robustness_syst_jpsi_fit_opt.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    opt_yield, opt_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)

    cut = ROOT.TCut(f'{dataset_params.score_branch}>{cut_high}')
    dataset_data = dataset_data.reduce(cut.GetTitle())

    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'plots' / f'bdt_robustness_syst_jpsi_fit_high.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        file_formats=['pdf']
    )

    high_yield, high_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, [5.1,5.4], model_final.fit_result)

    output_dict = {
        'low_yield'  : {'value' : low_yield, 'uncertainty' : low_yield_err},
        'opt_yield'  : {'value' : opt_yield, 'uncertainty' : opt_yield_err},
        'high_yield' : {'value' : high_yield, 'uncertainty' : high_yield_err},
    }

    return output_dict


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    cut_nominal = 3.9
    cut_loose = 0.

    args.mode = 'jpsi'
    set_mode(dataset_params, output_params, fit_params, args)
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
    yield_jpsi_opt, yield_jpsi_opt_err = get_yield_jpsi_fit(
        cut_low,
        cut_opt,
        cut_high,
        jpsi_fit_defaults,
        dataset_params,
        output_params,
        fit_params,
        args,
    )
    
    print(f'Efficiency Ratio = {eff_ratio.n} \u00B1 {eff_ratio.std_dev}')   
    output_dict = {
        'yield_opt'  : {'value' : eff_ratio.n, 'uncertainty' : eff_ratio.std_dev},
        'yield_low'  : {'value' : eff_jpsi_data, 'uncertainty' : eff_jpsi_data_err},
        'yield_high' : {'value' : eff_jpsi_mc, 'uncertainty' : eff_jpsi_mc_err},
    }
    with open('bdt_data_mc_systematic_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'value', 'uncertainty'])
        for name, info in data.items():
            writer.writerow([name, info['value'], info['uncertainty']])
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='../fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    args = parser.parse_args()

    main(args)
