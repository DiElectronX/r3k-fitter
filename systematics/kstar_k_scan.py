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

ALLOWED_MODES = ['jpsi', 'psi2s']
BR_B_JPSIK = 1.02E-3
BR_B_JPSIK_ERR = 1.9E-5
BR_B_JPSIKPLUSSTAR = 1.43E-3
BR_B_JPSIKPLUSSTAR_ERR = 8E-5
BR_B0_JPSIK0STAR = 1.27E-3
BR_B0_JPSIK0STAR_ERR = 5E-5
BR_B_PSI2SK = 6.24E-4
BR_B_PSI2SK_ERR = 2.1E-5
BR_B_PSI2SKPLUSSTAR = 6.7E-4
BR_B_PSI2SKPLUSSTAR_ERR = 1.4E-4
BR_B0_PSI2SK0STAR = 5.9E-4
BR_B0_PSI2SK0STAR_ERR = 4E-5


def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    if isinstance(iterable,zip):
        unzipped = list(iterable)
        return iterable if args.verbose else tqdm(unzipped, total=len(unzipped), unit=unit)
    else:
        return iterable if args.verbose else tqdm(iterable, total=len(iterable), unit=unit)


def get_mc_eff(bdt_cut, mode, file, dataset_params, output_params, fit_params, total_mc=500_000_000, mass_window=None, label=None):
    args.mode = mode
    set_mode(dataset_params, output_params, fit_params, args)

    fig, ax = plt.subplots(figsize=(8,8))
    fig_labels = {
        'sig' : r'$B^{+} \rightarrow J/ \psi (e^{+}e^{-}) K^{+}$',
        'k0star' :r'$B^{0} \rightarrow J/ \psi (e^{+}e^{-}) K^{*0}$',
        'kplusstar' :r'$B^{+} \rightarrow J/ \psi (e^{+}e^{-}) K^{*+}$',
    }
    nbins = 60
    bins = np.linspace(fit_params.full_mass_range[0],fit_params.full_mass_range[1], nbins)

    if isinstance(file,list):
        _effs = np.array([])
        for f in file:
            df = ROOT.RDataFrame(dataset_params.tree_name, f)
            cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && bdt_score>{str(bdt_cut)}'
            cut_string_sig = cut_string+f' && (Bmass > {mass_window[0]} && Bmass < {mass_window[1]})' if mass_window else cut_string
            k_arr = df.Filter(cut_string).AsNumpy(columns=[dataset_params.b_mass_branch,dataset_params.mc_weight_branch])
            k_arr_sig = df.Filter(cut_string_sig).AsNumpy(columns=[dataset_params.b_mass_branch,dataset_params.mc_weight_branch])
            k = df.Filter(cut_string_sig).Sum(dataset_params.mc_weight_branch).GetValue()
            n = total_mc

            try:
                eff = k / n
                unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
                # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
            except ZeroDivisionError:
                eff = 0
                unc = 0

            _effs = np.append(_effs,ufloat(eff,unc))

            ax.hist(k_arr[dataset_params.b_mass_branch], bins=bins, weights=k_arr[dataset_params.mc_weight_branch], histtype='step', label='All Events')
            ax.hist(k_arr_sig[dataset_params.b_mass_branch], bins=bins, weights=k_arr_sig[dataset_params.mc_weight_branch], histtype='stepfilled', label='Events in mass window')

        eff = _effs.sum().n
        unc = _effs.sum().std_dev

    else:
        df = ROOT.RDataFrame(dataset_params.tree_name, file)
        cut_string = f'(Mll > {fit_params.ll_mass_range[0]} && Mll < {fit_params.ll_mass_range[1]}) && bdt_score>{str(bdt_cut)}'
        cut_string_sig = cut_string+f' && (Bmass > {mass_window[0]} && Bmass < {mass_window[1]})' if mass_window else cut_string
        k_arr = df.Filter(cut_string).AsNumpy(columns=[dataset_params.b_mass_branch,dataset_params.mc_weight_branch])
        k_arr_sig = df.Filter(cut_string_sig).AsNumpy(columns=[dataset_params.b_mass_branch,dataset_params.mc_weight_branch])
        k = df.Filter(cut_string_sig).Sum(dataset_params.mc_weight_branch).GetValue()
        n = total_mc

        try:
            eff = k / n
            unc = np.sqrt(eff * (1 - eff) / n) # Binomial stats
            # unc = eff * np.sqrt((1 / k) + (1 / n)) # Poisson stats
        except ZeroDivisionError:
            eff = 0
            unc = 0

        ax.hist(k_arr[dataset_params.b_mass_branch], bins=bins, weights=k_arr[dataset_params.mc_weight_branch], histtype='step', label='All Events')
        ax.hist(k_arr_sig[dataset_params.b_mass_branch], bins=bins, weights=k_arr_sig[dataset_params.mc_weight_branch], histtype='stepfilled', label='Events in mass window')

    leg_title = fig_labels[label].replace('J/ \psi','\psi(2s)') if mode=='psi2s' else fig_labels[label]
    ax.legend(loc='upper left', title=leg_title+f'\nBDT Score > {str(round(bdt_cut,2))}', title_fontsize='14', fontsize='14')
    ax.set_xlabel('m(B Candidate) [GeV]', fontsize=16, loc='right')
    ax.set_ylabel('$N_{events}$', fontsize=16, loc='top')
    fig.savefig(Path('.') / 'scan_fits' / f'{mode}_{label}_mc_events_bdt>{str(round(bdt_cut,2))}.pdf', bbox_inches='tight')
    plt.close(fig)

    return eff, unc, k


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
    _, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_jpsi_pion_file)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_pion_file)#, extra_weight=.1)
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_kaon_file)#, extra_weight=.1)
    dataset_kstar_pion_comb = dataset_kstar_pion.Clone('dataset_kstar_pion_comb'+fit_params.channel_label)
    dataset_kstar_pion_comb.append(dataset_k0star_pion)

    model_kstar_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar_pion_comb, 'channel_label' : fit_params.channel_label})
    model_kstar_pion_template.add_background_model('part_bkg_pdf_1', 'kde', fit_defaults, let_float=True)
    model_kstar_pion_template.fit_model = model_kstar_pion_template.part_bkg_pdf_1

    model_kstar_kaon_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_kaon, 'channel_label' : fit_params.channel_label})
    model_kstar_kaon_template.add_background_model('part_bkg_pdf_2', 'kde', fit_defaults, let_float=True)
    model_kstar_kaon_template.fit_model = model_kstar_kaon_template.part_bkg_pdf_2

    # Fit composite model to data
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)
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
    signal_yields = np.array([])
    signal_yield_errs = np.array([])
    comb_yields = np.array([])
    comb_yield_errs = np.array([])
    kstar_yields = np.array([])
    kstar_yield_errs = np.array([])
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
            extra_text=f'BDT Score > {str(round(bdt_cut,2))}',
            file_formats=['pdf']
        )

        if integral:
            signal_yield, signal_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, integral, model_final.fit_result)
            comb_yield, comb_yield_err = integrate(b_mass_branch, model_final.comb_bkg_pdf, comb_bkg_coeff, integral, model_final.fit_result)
            kstar_yield_1, kstar_yield_err_1 = integrate(b_mass_branch, model_final.part_bkg_pdf_1, part_bkg_1_coeff, integral, model_final.fit_result)
            kstar_yield_2, kstar_yield_err_2 = integrate(b_mass_branch, model_final.part_bkg_pdf_2, part_bkg_2_coeff, integral, model_final.fit_result)
            kstar_yield, kstar_yield_err = kstar_yield_1 + kstar_yield_2, np.sqrt(kstar_yield_err_1**2 + kstar_yield_err_2**2)

        else:
            signal_yield, signal_yield_err = sig_coeff.getVal(), sig_coeff.getError()
            comb_yield, comb_yield_err = comb_bkg_coeff.getVal(), comb_bkg_coeff.getError()
            kstar_yield, kstar_yield_err = part_bkg_1_coeff.getVal()+part_bkg_2_coeff.getVal(), np.sqrt(part_bkg_1_coeff.getError()**2+part_bkg_2_coeff.getError()**2)

        signal_yields = np.append(signal_yields,signal_yield)
        signal_yield_errs = np.append(signal_yield_errs,signal_yield_err)
        comb_yields = np.append(comb_yields,comb_yield)
        comb_yield_errs = np.append(comb_yield_errs,comb_yield_err)
        kstar_yields = np.append(kstar_yields,kstar_yield)
        kstar_yield_errs = np.append(kstar_yield_errs,kstar_yield_err)

    return signal_yields, signal_yield_errs, comb_yields, comb_yield_errs, kstar_yields, kstar_yield_errs


def do_psi2s_quickfits(bdt_cuts, fit_defaults, dataset_params, output_params, fit_params, args, integral=None):
    args.mode = 'psi2s'
    fit_params.bdt_score_cut = bdt_cuts[0]
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)

    # Use J/Psi MC for signal template 
    b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)
    model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
    model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_defaults, let_float=True)
    model_sig_template.fit_model = model_sig_template.sig_pdf
    model_sig_template.fit(dataset_mc, printlevel=printlevel)
    
    # Use K* MC for background template
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_pion_file, score_cut=bdt_cuts[0])
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_kaon_file, score_cut=bdt_cuts[0])

    model_kstar_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_pion, 'channel_label' : fit_params.channel_label})
    model_kstar_pion_template.add_background_model('part_bkg_pdf_1', 'kde', fit_defaults, let_float=True)
    model_kstar_pion_template.fit_model = model_kstar_pion_template.part_bkg_pdf_1

    model_kstar_kaon_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_kaon, 'channel_label' : fit_params.channel_label})
    model_kstar_kaon_template.add_background_model('part_bkg_pdf_2', 'kde', fit_defaults, let_float=True)
    model_kstar_kaon_template.fit_model = model_kstar_kaon_template.part_bkg_pdf_2

    # Fit composite model to data
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', model_sig_template.signal_models['sig_pdf'])
    model_final.add_background_model('comb_bkg_pdf', 'exp', fit_defaults, let_float=True)
    model_final.add_background_model('part_bkg_pdf_1', model_kstar_pion_template.background_models['part_bkg_pdf_1'])
    model_final.add_background_model('part_bkg_pdf_2', model_kstar_kaon_template.background_models['part_bkg_pdf_2'])
    
    # Give initial part.bkg. yield mixture from templates
    part_bkg_yield_init = 10500
    part_bkg_1_yield_init = part_bkg_yield_init * dataset_k0star_pion.sumEntries() / (dataset_k0star_pion.sumEntries() + dataset_k0star_kaon.sumEntries())
    part_bkg_2_yield_init = part_bkg_yield_init * dataset_k0star_kaon.sumEntries() / (dataset_k0star_pion.sumEntries() + dataset_k0star_kaon.sumEntries()) 

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 124395, 0, dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Background PDF Coefficient', 961773, 0, dataset_data.numEntries())
    part_bkg_1_coeff = ROOT.RooRealVar('part_bkg_1_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 1 Coefficient', part_bkg_1_yield_init, 0, dataset_data.numEntries())
    part_bkg_2_coeff = ROOT.RooRealVar('part_bkg_2_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 2 Coefficient', part_bkg_2_yield_init, 0, dataset_data.numEntries())
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

    model_kstar = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_kstar.fit_model = ROOT.RooAddPdf(
        'pdf_sum_kstar',
        'Sum of Kstar0 Background PDFs',
        ROOT.RooArgList(
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ),
        ROOT.RooArgList(
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
        'dcb_ratio_constraint' : ROOT.RooGaussian('dcb_ratio_constraint', 'dcb_ratio_constraint', dcb_ratio, ROOT.RooFit.RooConst(dcb_ratio.getVal()), ROOT.RooFit.RooConst(.2)),
    })

    # Optimize inital fit to loosest bdt cut
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path('.') / 'scan_fits' / f'psi2s_fit_bdt>{str(round(bdt_cuts[0],2))}.pdf',
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf_1,
            model_final.part_bkg_pdf_2,
        ],
        extra_text=f'BDT > {str(round(bdt_cuts[0],2))}',
        file_formats=['pdf']
    )

    if integral:
        signal_yield, signal_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, integral, model_final.fit_result)
        comb_yield, comb_yield_err = integrate(b_mass_branch, model_final.comb_bkg_pdf, comb_bkg_coeff, integral, model_final.fit_result)
        kstar_yield_1, kstar_yield_err_1 = integrate(b_mass_branch, model_final.part_bkg_pdf_1, part_bkg_1_coeff, integral, model_final.fit_result)
        kstar_yield_2, kstar_yield_err_2 = integrate(b_mass_branch, model_final.part_bkg_pdf_2, part_bkg_2_coeff, integral, model_final.fit_result)
        kstar_yield, kstar_yield_err = kstar_yield_1 + kstar_yield_2, np.sqrt(kstar_yield_err_1**2 + kstar_yield_err_2**2)

    else:
        signal_yield, signal_yield_err = sig_coeff.getVal(), sig_coeff.getError()
        comb_yield, comb_yield_err = comb_bkg_coeff.getVal(), comb_bkg_coeff.getError()
        kstar_yield, kstar_yield_err = part_bkg_1_coeff.getVal()+part_bkg_2_coeff.getVal(), np.sqrt(part_bkg_1_coeff.getError()**2+part_bkg_2_coeff.getError()**2)

    # Initialize output list for scan
    signal_yields = np.array([signal_yield])
    signal_yield_errs = np.array([signal_yield_err])
    comb_yields = np.array([comb_yield])
    comb_yield_errs = np.array([comb_yield_err])
    kstar_yields = np.array([kstar_yield])
    kstar_yield_errs = np.array([kstar_yield_err])

    # Fit model to data for successive bdt cuts
    for bdt_cut in loop_wrapper(bdt_cuts[1:], args, title='Performing Psi(2s) Fit Scan'):
        # Apply tighter bdt cut to data
        cut = ROOT.TCut(f'{dataset_params.score_branch}>{bdt_cut}')
        dataset_data = dataset_data.reduce(cut.GetTitle())
        
        # Reshape MC part. bkg. templates with new bdt cut
        _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_pion_file, score_cut=bdt_cut)
        _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_kaon_file, score_cut=bdt_cut)
        model_kstar_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_pion, 'channel_label' : fit_params.channel_label})
        model_kstar_pion_template.add_background_model('part_bkg_pdf_1', 'kde', fit_defaults, let_float=True)
        model_kstar_pion_template.fit_model = model_kstar_pion_template.part_bkg_pdf_1

        model_kstar_kaon_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_k0star_kaon, 'channel_label' : fit_params.channel_label})
        model_kstar_kaon_template.add_background_model('part_bkg_pdf_2', 'kde', fit_defaults, let_float=True)
        model_kstar_kaon_template.fit_model = model_kstar_kaon_template.part_bkg_pdf_2

        _model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
        _model_final.add_signal_model('sig_pdf', model_final.sig_models['sig_pdf'])
        _model_final.add_background_model('comb_bkg_pdf', model_final.bkg_models['comb_bkg_pdf'])
        _model_final.add_background_model('part_bkg_pdf_1', model_final.bkg_models['part_bkg_pdf_1'])
        _model_final.add_background_model('part_bkg_pdf_2', model_final.bkg_models['part_bkg_pdf_2'])
        print(dir(model_final))
        model_final = _model_final
        print(dir(model_final))
        exit()
        # Give initial part.bkg. yield mixture from templates
        part_bkg_yield_init = part_bkg_1_coeff.getVal()+part_bkg_2_coeff.getVal()
        part_bkg_1_yield_init = part_bkg_yield_init * dataset_k0star_pion.sumEntries() / (dataset_k0star_pion.sumEntries() + dataset_k0star_kaon.sumEntries())
        part_bkg_2_yield_init = part_bkg_yield_init * dataset_k0star_kaon.sumEntries() / (dataset_k0star_pion.sumEntries() + dataset_k0star_kaon.sumEntries()) 

        sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', sig_coeff.getVal(), 0, dataset_data.numEntries())
        comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Background PDF Coefficient', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
        part_bkg_1_coeff = ROOT.RooRealVar('part_bkg_1_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 1 Coefficient', part_bkg_1_yield_init, 0, dataset_data.numEntries())
        part_bkg_2_coeff = ROOT.RooRealVar('part_bkg_2_coeff'+fit_params.channel_label, 'Partially Reconstructed Background 2 Coefficient', part_bkg_2_yield_init, 0, dataset_data.numEntries())
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
        
        # Refit to new dataset
        model_final.fit(dataset_data, printlevel=printlevel)
        params = model_final.fit_result.floatParsFinal()
        
        # Plot fit result
        model_final.plot_fit(
            b_mass_branch,
            dataset_data,
            Path('.') / 'scan_fits' / f'psi2s_fit_bdt>{str(round(bdt_cut,2))}.pdf',
            fit_components = [
                model_final.sig_pdf,
                model_final.comb_bkg_pdf,
                model_final.part_bkg_pdf_1,
                model_final.part_bkg_pdf_2,
            ],
            extra_text=f'BDT > {str(round(bdt_cut,2))}',
            file_formats=['pdf']
        )

        if integral:
            signal_yield, signal_yield_err = integrate(b_mass_branch, model_final.sig_pdf, sig_coeff, integral, model_final.fit_result)
            comb_yield, comb_yield_err = integrate(b_mass_branch, model_final.comb_bkg_pdf, comb_bkg_coeff, integral, model_final.fit_result)
            kstar_yield_1, kstar_yield_err_1 = integrate(b_mass_branch, model_final.part_bkg_pdf_1, part_bkg_1_coeff, integral, model_final.fit_result)
            kstar_yield_2, kstar_yield_err_2 = integrate(b_mass_branch, model_final.part_bkg_pdf_2, part_bkg_2_coeff, integral, model_final.fit_result)
            kstar_yield, kstar_yield_err = kstar_yield_1 + kstar_yield_2, np.sqrt(kstar_yield_err_1**2 + kstar_yield_err_2**2)

        else:
            signal_yield, signal_yield_err = sig_coeff.getVal(), sig_coeff.getError()
            comb_yield, comb_yield_err = comb_bkg_coeff.getVal(), comb_bkg_coeff.getError()
            kstar_yield, kstar_yield_err = part_bkg_1_coeff.getVal()+part_bkg_2_coeff.getVal(), np.sqrt(part_bkg_1_coeff.getError()**2+part_bkg_2_coeff.getError()**2)

        signal_yields = np.append(signal_yields,signal_yield)
        signal_yield_errs = np.append(signal_yield_errs,signal_yield_err)
        comb_yields = np.append(comb_yields,comb_yield)
        comb_yield_errs = np.append(comb_yield_errs,comb_yield_err)
        kstar_yields = np.append(kstar_yields,kstar_yield)
        kstar_yield_errs = np.append(kstar_yield_errs,kstar_yield_err)

    return signal_yields, signal_yield_errs, comb_yields, comb_yield_errs, kstar_yields, kstar_yield_errs

def jpsi_kstar_k_scan(dataset_params, output_params, fit_params, args):
    
    scan_range = np.linspace(1,7,15)
    mass_window = None #[5.1,5.4]

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

    outputs = {
        'metadata' : {'mass_window' : mass_window},
        'score' : np.array([]),
        'r_jpsi_data' : np.array([]),
        'r_jpsi_data_err' : np.array([]),
        'r_jpsi_mc' : np.array([]),
        'r_jpsi_mc_err' : np.array([]),
        'n_jpsi_sig' : np.array([]),
        'n_jpsi_sig_err' : np.array([]),
        'n_jpsi_comb' : np.array([]),
        'n_jpsi_comb_err' : np.array([]),
        'n_jpsi_kstar' : np.array([]),
        'n_jpsi_kstar_err' : np.array([]),
        'n_mc_jpsi_sig' : np.array([]),
        'n_mc_jpsi_kstar' : np.array([]),
        'eff_jpsi_sig' : np.array([]),
        'eff_jpsi_sig_err' : np.array([]),
        'eff_jpsi_kplusstar' : np.array([]),
        'eff_jpsi_kplusstar_err' : np.array([]),
        'eff_jpsi_k0star' : np.array([]),
        'eff_jpsi_k0star_err' : np.array([]),
    }

    jpsi_fit_yields = do_jpsi_quickfits(scan_range, jpsi_fit_defaults, dataset_params, output_params, fit_params, args, integral=mass_window)

    for bdt_cut,n_jpsi_sig,n_jpsi_sig_err,n_jpsi_comb,n_jpsi_comb_err,n_jpsi_kstar,n_jpsi_kstar_err in loop_wrapper(zip(scan_range,*jpsi_fit_yields), args, title='Calculating R(K*)'):
        eff_jpsi_sig, eff_jpsi_sig_err, n_mc_jpsi_sig = get_mc_eff(
            bdt_cut,
            'jpsi',
            dataset_params.jpsi_file,
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=500_000_000,
            label='sig'
        )

        eff_jpsi_kplusstar, eff_jpsi_kplusstar_err, n_mc_jpsi_kstar = get_mc_eff(
            bdt_cut,
            'jpsi',
            [dataset_params.k0star_jpsi_kaon_file,dataset_params.kstar_jpsi_pion_file],
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=196_000_000,
            label='k0star'
        )

        eff_jpsi_k0star, eff_jpsi_k0star_err, n_mc_jpsi_kstar = get_mc_eff(
            bdt_cut,
            'jpsi',
            [dataset_params.k0star_jpsi_kaon_file,dataset_params.k0star_jpsi_pion_file],
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=196_000_000,
            label='k0star'
        )

        _n_jpsi_sig = ufloat(n_jpsi_sig, n_jpsi_sig_err)
        _n_jpsi_kstar = ufloat(n_jpsi_kstar, n_jpsi_kstar_err)
        _eff_jpsi_sig = ufloat(eff_jpsi_sig, eff_jpsi_sig_err)
        _eff_jpsi_kplusstar = ufloat(eff_jpsi_kplusstar, eff_jpsi_kplusstar_err)
        _eff_jpsi_k0star = ufloat(eff_jpsi_k0star, eff_jpsi_k0star_err)
        _br_jpsi_sig = ufloat(BR_B_JPSIK, BR_B_JPSIK_ERR)
        _br_jpsi_kplusstar = ufloat(BR_B_JPSIKPLUSSTAR, BR_B_JPSIKPLUSSTAR_ERR)
        _br_jpsi_k0star = ufloat(BR_B0_JPSIK0STAR, BR_B0_JPSIK0STAR_ERR)

        _r_jpsi_data = _n_jpsi_kstar / _n_jpsi_sig
        _r_jpsi_mc = (_br_jpsi_k0star * _eff_jpsi_k0star + _br_jpsi_kplusstar * _eff_jpsi_kplusstar) / (_br_jpsi_sig * _eff_jpsi_sig)
        
        outputs['score'] = np.append(outputs['score'], bdt_cut)
        outputs['n_jpsi_sig'] = np.append(outputs['n_jpsi_sig'], n_jpsi_sig)
        outputs['n_jpsi_sig_err'] = np.append(outputs['n_jpsi_sig_err'], n_jpsi_sig_err)
        outputs['n_jpsi_comb'] = np.append(outputs['n_jpsi_comb'], n_jpsi_comb)
        outputs['n_jpsi_comb_err'] = np.append(outputs['n_jpsi_comb_err'], n_jpsi_comb_err)
        outputs['n_jpsi_kstar'] = np.append(outputs['n_jpsi_kstar'], n_jpsi_kstar)
        outputs['n_jpsi_kstar_err'] = np.append(outputs['n_jpsi_kstar_err'], n_jpsi_kstar_err)
        outputs['n_mc_jpsi_sig'] = np.append(outputs['n_mc_jpsi_sig'], n_mc_jpsi_sig)
        outputs['n_mc_jpsi_kstar'] = np.append(outputs['n_mc_jpsi_kstar'], n_mc_jpsi_kstar)
        outputs['eff_jpsi_sig'] = np.append(outputs['eff_jpsi_sig'], eff_jpsi_sig)
        outputs['eff_jpsi_sig_err'] = np.append(outputs['eff_jpsi_sig_err'], eff_jpsi_sig_err)
        outputs['eff_jpsi_kplusstar'] = np.append(outputs['eff_jpsi_kplusstar'], eff_jpsi_kplusstar)
        outputs['eff_jpsi_kplusstar_err'] = np.append(outputs['eff_jpsi_kplusstar_err'], eff_jpsi_kplusstar_err)
        outputs['eff_jpsi_k0star'] = np.append(outputs['eff_jpsi_k0star'], eff_jpsi_k0star)
        outputs['eff_jpsi_k0star_err'] = np.append(outputs['eff_jpsi_k0star_err'], eff_jpsi_k0star_err)
        
        outputs['r_jpsi_data'] = np.append(outputs['r_jpsi_data'], _r_jpsi_data.n)
        outputs['r_jpsi_data_err'] = np.append(outputs['r_jpsi_data_err'], _r_jpsi_data.std_dev)
        outputs['r_jpsi_mc'] = np.append(outputs['r_jpsi_mc'], _r_jpsi_mc.n)
        outputs['r_jpsi_mc_err'] = np.append(outputs['r_jpsi_mc_err'], _r_jpsi_mc.std_dev)

    path = Path('.') / 'jpsi_kstar_k_scan_data.pkl'
    with open(path,'wb') as pkl_file:
        pickle.dump(outputs, pkl_file)


def psi2s_kstar_k_scan(dataset_params, output_params, fit_params, args):

    scan_range = np.linspace(1,7,15)
    mass_window = None#[4.9,5.15]

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

    outputs = {
        'metadata' : {'mass_window' : mass_window},
        'score' : np.array([]),
        'r_psi2s_data' : np.array([]),
        'r_psi2s_data_err' : np.array([]),
        'r_psi2s_mc' : np.array([]),
        'r_psi2s_mc_err' : np.array([]),
        'n_psi2s_sig' : np.array([]),
        'n_psi2s_sig_err' : np.array([]),
        'n_psi2s_comb' : np.array([]),
        'n_psi2s_comb_err' : np.array([]),
        'n_psi2s_kstar' : np.array([]),
        'n_psi2s_kstar_err' : np.array([]),
        'n_mc_psi2s_sig' : np.array([]),
        'n_mc_psi2s_kstar' : np.array([]),
        'eff_psi2s_sig' : np.array([]),
        'eff_psi2s_sig_err' : np.array([]),
        'eff_psi2s_kplusstar' : np.array([]),
        'eff_psi2s_kplusstar_err' : np.array([]),
        'eff_psi2s_k0star' : np.array([]),
        'eff_psi2s_k0star_err' : np.array([]),
    }

    psi2s_fit_yields = do_psi2s_quickfits(scan_range, psi2s_fit_defaults, dataset_params, output_params, fit_params, args, integral=mass_window)

    for bdt_cut,n_psi2s_sig,n_psi2s_sig_err,n_psi2s_comb,n_psi2s_comb_err,n_psi2s_kstar,n_psi2s_kstar_err in loop_wrapper(zip(scan_range,*psi2s_fit_yields), args, title='Calculating R(K*)'):
        eff_psi2s_sig, eff_psi2s_sig_err, n_mc_psi2s_sig = get_mc_eff(
            bdt_cut,
            'psi2s',
            dataset_params.psi2s_file,
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=500_000_000,
            label='sig'
        )

        eff_psi2s_kplusstar, eff_psi2s_kplusstar_err, n_mc_psi2s_kstar = get_mc_eff(
            bdt_cut,
            'psi2s',
            [dataset_params.k0star_psi2s_kaon_file,dataset_params.kstar_psi2s_pion_file],
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=196_000_000,
            label='k0star'
        )

        eff_psi2s_k0star, eff_psi2s_k0star_err, n_mc_psi2s_kstar = get_mc_eff(
            bdt_cut,
            'psi2s',
            [dataset_params.k0star_psi2s_kaon_file,dataset_params.k0star_psi2s_pion_file],
            dataset_params,
            output_params,
            fit_params,
            mass_window=mass_window,
            total_mc=196_000_000,
            label='k0star'
        )

        _n_psi2s_sig = ufloat(n_psi2s_sig, n_psi2s_sig_err)
        _n_psi2s_kstar = ufloat(n_psi2s_kstar, n_psi2s_kstar_err)
        _eff_psi2s_sig = ufloat(eff_psi2s_sig, eff_psi2s_sig_err)
        _eff_psi2s_kplusstar = ufloat(eff_psi2s_kplusstar, eff_psi2s_kplusstar_err)
        _eff_psi2s_k0star = ufloat(eff_psi2s_k0star, eff_psi2s_k0star_err)
        _br_psi2s_sig = ufloat(BR_B_PSI2SK, BR_B_PSI2SK_ERR)
        _br_psi2s_kplusstar = ufloat(BR_B_PSI2SKPLUSSTAR, BR_B_PSI2SKPLUSSTAR_ERR)
        _br_psi2s_k0star = ufloat(BR_B0_PSI2SK0STAR, BR_B0_PSI2SK0STAR_ERR)

        _r_psi2s_data = _n_psi2s_kstar / _n_psi2s_sig
        _r_psi2s_mc = (_br_psi2s_k0star * _eff_psi2s_k0star + _br_psi2s_kplusstar * _eff_psi2s_kplusstar) / (_br_psi2s_sig * _eff_psi2s_sig)
        
        outputs['score'] = np.append(outputs['score'], bdt_cut)
        outputs['n_psi2s_sig'] = np.append(outputs['n_psi2s_sig'], n_psi2s_sig)
        outputs['n_psi2s_sig_err'] = np.append(outputs['n_psi2s_sig_err'], n_psi2s_sig_err)
        outputs['n_psi2s_comb'] = np.append(outputs['n_psi2s_comb'], n_psi2s_comb)
        outputs['n_psi2s_comb_err'] = np.append(outputs['n_psi2s_comb_err'], n_psi2s_comb_err)
        outputs['n_psi2s_kstar'] = np.append(outputs['n_psi2s_kstar'], n_psi2s_kstar)
        outputs['n_psi2s_kstar_err'] = np.append(outputs['n_psi2s_kstar_err'], n_psi2s_kstar_err)
        outputs['n_mc_psi2s_sig'] = np.append(outputs['n_mc_psi2s_sig'], n_mc_psi2s_sig)
        outputs['n_mc_psi2s_kstar'] = np.append(outputs['n_mc_psi2s_kstar'], n_mc_psi2s_kstar)
        outputs['eff_psi2s_sig'] = np.append(outputs['eff_psi2s_sig'], eff_psi2s_sig)
        outputs['eff_psi2s_sig_err'] = np.append(outputs['eff_psi2s_sig_err'], eff_psi2s_sig_err)
        outputs['eff_psi2s_kplusstar'] = np.append(outputs['eff_psi2s_kplusstar'], eff_psi2s_kplusstar)
        outputs['eff_psi2s_kplusstar_err'] = np.append(outputs['eff_psi2s_kplusstar_err'], eff_psi2s_kplusstar_err)
        outputs['eff_psi2s_k0star'] = np.append(outputs['eff_psi2s_k0star'], eff_psi2s_k0star)
        outputs['eff_psi2s_k0star_err'] = np.append(outputs['eff_psi2s_k0star_err'], eff_psi2s_k0star_err)
        
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
