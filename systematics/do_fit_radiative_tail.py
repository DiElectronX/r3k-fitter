import ROOT
import yaml
import argparse
import copy
from pprint import pprint

import sys
import numpy as np
from pathlib import Path
import ROOT
import ROOT.RooFit as rf
import csv
sys.path.insert(1, str(Path('..').resolve()))

from utils import *
from fit_models import FitModel

ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']

def save_yields_err(yields, output_params, filename):
    os.makedirs(output_params.output_dir, exist_ok=True)
    filepath = os.path.join(output_params.output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Yield", "Value"])
        for key, value in yields.items():
            writer.writerow([key, value])

def get_MC_yield_cut_and_count(file_name, tree_name, cut_string, weights='trig_wgt', denom_string=None, denom=500000000, get_comps=False):
    df = ROOT.RDataFrame(tree_name, file_name)
    if weights:
        k = df.Filter(cut_string).Sum(weights).GetValue()
        # n = df.Filter(denom_string).Sum(weights).GetValue() if denom_string else denom
    else:
        k = df.Filter(cut_string).Count().GetValue()
        # n = df.Filter(denom_string).Count().GetValue() if denom_string else denom
    return k

def get_eff_cut_and_count(output_params, file_name, tree_name, cut_string, weights='trig_wgt', denom_string=None, denom=500_000_000, get_comps=False, get_yields=False):
    '''
    Getting the efficiency from the MC Jpsi files
    '''
    df = ROOT.RDataFrame(tree_name, file_name)
    cut_nominal_num = str(cut_string.split('>')[-1])
    cut_loose_denom = str(denom_string.split('>')[-1])
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

    output_dict = {'jpsi_mc_num': k,
                'jpsi_mc_denom': n,
                'eff_jpsi_mc': eff,
                'eff_jpsi_mc_unc': unc,
                }
    # saving yields and eff
    save_yields_err(output_dict, output_params, filename='mc_eff_yield_'+ cut_nominal_num + '_' + cut_loose_denom +'.csv')
    
    if get_comps:
        # print(f'eff_jpsi_mc:{eff}')
        # print(f'eff_jpsi_mc_err:{unc}')
        # print(f'k:{k}')
        # print(f'sqrt(k):{np.sqrt(k)}')
        # print(f'n:{n}')
        # print(f'sqrt(n):{np.sqrt(n)}')
        pprint(output_dict)
        return eff, unc, (k, np.sqrt(k), n, np.sqrt(n)),
    else:
        # print(f'eff_jpsi_mc:{eff}')
        # print(f'eff_jpsi_mc_err:{unc}')
        pprint(output_dict)
        return eff, unc



def jpsi_fit_dcbdcb_sig(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False, coeffCon=False, isMinos=False):
    '''
    The standard full fitting with 
        signal: dcb+dcb
        combinatorial: exp
        partial k(0)star: kde
        partial jpsipi_jpsi_pion: dcb
    '''
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))
        # if args.verbose:
        #     print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_mc = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        # model_sig_template.fit(dataset_mc, printlevel=printlevel)
        model_sig_template.fit(dataset_mc, printlevel=printlevel, isMinos=isMinos)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_sig_template.pdf'),
            file_label=file_label,
            fit_components = [
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
            legend=True,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))
        # if args.verbose:
        #     print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        # model_comb_template.fit(dataset_data, printlevel=printlevel)
        model_comb_template.fit(dataset_data, printlevel=printlevel, isMinos=isMinos)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_comb_template.pdf'),
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
            legend=True,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))
    # if args.verbose:
    #     print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))
    
    # Look at partial shape files
    tmp_b_mass_branch, dataset_kstar_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_kstar_pion   = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_pion_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_kaon  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_pion_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_chic1_kaon   = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.chic1_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    dataset_kstar_comb = dataset_kstar_kaon.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_kstar_pion)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)
    dataset_kstar_comb.append(dataset_chic1_kaon)
    # dataset_kstar_comb.append(dataset_jpsipi_pion)
    
    mc_yield_tot = dataset_kstar_comb.sumEntries()
    
    kstar_kaon_yield_frac = dataset_kstar_kaon.sumEntries() / mc_yield_tot
    kstar_pion_yield_frac = dataset_kstar_pion.sumEntries() / mc_yield_tot
    k0star_kaon_yield_frac = dataset_k0star_kaon.sumEntries() / mc_yield_tot
    k0star_pion_yield_frac = dataset_k0star_pion.sumEntries() / mc_yield_tot
    chic1_kaon_yield_frac = dataset_chic1_kaon.sumEntries() / mc_yield_tot
    kstar_yield_frac = ((dataset_kstar_kaon.sumEntries() + 
                        dataset_kstar_pion.sumEntries() +
                        dataset_k0star_kaon.sumEntries() + 
                        dataset_k0star_pion.sumEntries()) / 
                        mc_yield_tot)

    if args.verbose: 
        print('nEvents for B+ -> J/ψ K*+ - Kee cand = {}'.format(dataset_kstar_kaon.sumEntries()))
        print('nEvents for B+ -> J/ψ K*+ - πee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for B0 -> J/ψ K*0 - Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))
        print('nEvents for B0 -> J/ψ K*0 - πee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for B+ -> χc1 K+  - Kee cand = {}'.format(dataset_chic1_kaon.sumEntries()))
        print('nEvents for B+ -> J/ψ π+  - πee cand = {}'.format(dataset_jpsipi_pion.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6,.5,.85,.85)
    tmp_frame = tmp_b_mass_branch.frame()
    
    dataset_kstar_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_kaon'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.MarkerColor(ROOT.kOrange))
    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_chic1_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('chic1_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kMagenta), ROOT.RooFit.MarkerColor(ROOT.kMagenta))
    dataset_jpsipi_pion.plotOn(tmp_frame, ROOT.RooFit.Name('jpsipi_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.MarkerColor(ROOT.kCyan))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination','combination','lpe'); l1.SetLineColor(ROOT.kBlack); l1.SetMarkerColor(ROOT.kBlack)
    l2 = leg.AddEntry('kstar_kaon','kstar_kaon','lpe'); l2.SetLineColor(ROOT.kOrange); l2.SetMarkerColor(ROOT.kOrange)
    l3 = leg.AddEntry('kstar_pion','kstar_pion','lpe'); l3.SetLineColor(ROOT.kBlue); l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon','k0star_kaon','lpe'); l4.SetLineColor(ROOT.kRed); l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion','k0star_pion','lpe'); l5.SetLineColor(ROOT.kGreen); l5.SetMarkerColor(ROOT.kGreen)
    l6 = leg.AddEntry('chic1_kaon','chic1_kaon','lpe'); l6.SetLineColor(ROOT.kMagenta); l6.SetMarkerColor(ROOT.kMagenta)
    l7 = leg.AddEntry('jpsipi_pion','jpsipi_pion','lpe'); l7.SetLineColor(ROOT.kCyan); l7.SetMarkerColor(ROOT.kCyan)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(os.path.join(output_params.output_dir,'jpsi_dataset_partial_bkgs.pdf'))
    tmp_c.Close()
    
    # Import ROOT file dataset
    model_part_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar_comb, 'channel_label' : fit_params.channel_label})
    model_part_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_part_template.fit_model = model_part_template.part_bkg_pdf

    # Plot fit result
    model_part_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_partial_template.pdf'),
        file_label=file_label,
        bins=30,
    )

    # Fit partial background shape to jpsipi MC
    if not args.cache:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))
        # if args.verbose:
        #     print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))
        
        _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
        model_jpsipi_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_jpsipi_pion, 'channel_label' : fit_params.channel_label})
        model_jpsipi_pion_template.add_background_model('part_bkg_pdf_jpsipi_pion', 'dcb', fit_params.fit_defaults, let_float=True)
        model_jpsipi_pion_template.fit_model = model_jpsipi_pion_template.part_bkg_pdf_jpsipi_pion
        
        # Fit model to data
        # model_jpsipi_pion_template.fit(dataset_jpsipi_pion, printlevel=printlevel)
        model_jpsipi_pion_template.fit(dataset_jpsipi_pion, printlevel=printlevel, isMinos=isMinos)
        params = model_jpsipi_pion_template.fit_result.floatParsFinal()

        # Plot fit result
        model_jpsipi_pion_template.plot_fit(
            b_mass_branch,
            dataset_jpsipi_pion,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_jpsipi_template.pdf'),
            file_label=file_label,
            fit_result=model_jpsipi_pion_template.fit_result,
            legend=True,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))
    # if args.verbose:
    #     print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            print('Load template: ', file.name)
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('part_bkg_pdf_jpsipi_pion', 'dcb', template, let_float=False)

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 60000, 0, dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 2000, 0, dataset_data.numEntries())
    part_bkg_coeff  = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Part. Bkg. PDF Coeff.', 3000, 0, dataset_data.numEntries())
    #part_bkg_jpsipi_pion_coeff = ROOT.RooRealVar('part_bkg_jpsipi_pion_coeff'+fit_params.channel_label, 'Part. Bkg. PDF Coeff.', 2280, 0, dataset_data.numEntries())
    part_bkg_jpsipi_pion_coeff = ROOT.RooFormulaVar('part_bkg_jpsipi_pion_coeff', 'Part. Bkg. PDF Coeff.', '0.038*@0', ROOT.RooArgList(sig_coeff))

    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf,
            model_final.part_bkg_pdf_jpsipi_pion,
        ),
        ROOT.RooArgList(
            sig_coeff,
            comb_bkg_coeff,
            part_bkg_coeff,
            part_bkg_jpsipi_pion_coeff,
        )
    )

    # Add gaussian contraints to fit parameters
    sig_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    part_bkg_coeff.setConstant(False)
    comb_bkg_coeff.setConstant(False)
    #part_bkg_jpsipi_pion_coeff.setConstant(False)
    #jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        #'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        # 'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(2.)),

        'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
    })
    if coeffCon:
        model_final.add_constraints({
            'dcb1_coeff_constraint' : ROOT.RooGaussian('dcb1_coeff_constraint', 'dcb1_coeff_constraint', model_final.signal_models['sig_pdf'].dcb1_coeff, ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']*.05)),
            'dcb2_coeff_constraint' : ROOT.RooGaussian('dcb2_coeff_constraint', 'dcb2_coeff_constraint', model_final.signal_models['sig_pdf'].dcb2_coeff, ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']*.05)),
        })

    # Fit model to data
    # model_final.fit(dataset_data, printlevel=printlevel)
    model_final.fit(dataset_data, printlevel=printlevel, isMinos=isMinos)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        file_label=file_label,
        fit_components = {
            'Signal' : model_final.sig_pdf,
            # model_final.signal_models['sig_pdf'].dcb1_pdf,
            # model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg.' : model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg.' : model_final.part_bkg_pdf,
            'B #rightarrow J/#psi #pi Bkg.' : model_final.part_bkg_pdf_jpsipi_pion,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text='N_{{J/#psi}} = {} #pm {}'.format(round(sig_coeff.getValV()), round(sig_coeff.getError(),1)),
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_jpsipi_pion_norm = ROOT.RooRealVar('part_bkg_pdf_jpsipi_pion'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_jpsipi_pion_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    _dcb1_coeff = model_final.signal_models['sig_pdf'].dcb1_coeff.getVal()
    _dcb2_coeff = model_final.signal_models['sig_pdf'].dcb2_coeff.getVal()
    _norm_sf = 1 / (_dcb1_coeff + _dcb2_coeff)
    model_final.signal_models['sig_pdf'].dcb1_coeff.setVal(_dcb1_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].dcb2_coeff.setVal(_dcb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, part_bkg_pdf_jpsipi_pion_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, part_bkg_pdf_jpsipi_pion_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    # Use function to grab yields
    yields = {
        'yield_sig' : round(sig_coeff.getValV(),2),
        'yield_sig_err' : round(sig_coeff.getError(),2),
        'yield_comb_bkg' : round(comb_bkg_coeff.getValV(),2),
        'yield_comb_bkg_err' : round(comb_bkg_coeff.getError(),2),
        'yield_part_bkg' : round(part_bkg_coeff.getValV(),2),
        'yield_part_bkg_err' : round(part_bkg_coeff.getError(),2),
        'yield_part_bkg_jpsipi_pion' : round(part_bkg_jpsipi_pion_coeff.getValV(),2),
        # 'yield_part_bkg_jpsipi_pion_err' : round(part_bkg_jpsipi_pion_coeff.getError(),2), 
        'yield_part_bkg_kstar' : round(part_bkg_coeff.getValV() * kstar_yield_frac,2),
        'yield_part_bkg_kstar_err' : round(part_bkg_coeff.getError(),2),
        'yield_part_bkg_kstar_kaon' : round(part_bkg_coeff.getValV() * kstar_kaon_yield_frac,2),
        'yield_part_bkg_kstar_kaon_err' : round(part_bkg_coeff.getError() * kstar_kaon_yield_frac,2),
        'yield_part_bkg_kstar_pion' : round(part_bkg_coeff.getValV() * kstar_pion_yield_frac,2),
        'yield_part_bkg_kstar_pion_err' : round(part_bkg_coeff.getError() * kstar_pion_yield_frac,2),
        'yield_part_bkg_k0star_kaon' : round(part_bkg_coeff.getValV() * k0star_kaon_yield_frac,2),
        'yield_part_bkg_k0star_kaon_err' : round(part_bkg_coeff.getError() * k0star_kaon_yield_frac,2),
        'yield_part_bkg_k0star_pion' : round(part_bkg_coeff.getValV() * k0star_pion_yield_frac,2),
        'yield_part_bkg_k0star_pion_err' : round(part_bkg_coeff.getError() * k0star_pion_yield_frac,2),
        'yield_part_bkg_chic1_kaon' : round(part_bkg_coeff.getValV() * chic1_kaon_yield_frac,2),
        'yield_part_bkg_chic1_kaon_err' : round(part_bkg_coeff.getError() * chic1_kaon_yield_frac,2),
    }

    if get_yields:
        save_yields_err(yields, output_params, filename=f'yield.csv')
        return yields
    else:
        pprint(yields)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])
    # args.mode = 'jpsi'
    set_mode(dataset_params, output_params, fit_params, args)

    relaxed_windows = [2.5, 2.6, 2.7, 2.8]

# #===========================================================================================================
# Jpsi Radiative Tail Systemics study 
# #===========================================================================================================
    # 1.) Jpsi Yield: from the data fiting using do_jpsi_control_region_fit function
    do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, get_yields=True)

    # 2.) With analysis BDT cut, run cut and count in MC to get yield_1_mc
    set_mode(dataset_params, output_params, fit_params, args)

    FILE_NAME
    dataset_params.samesign_data_file # same sign ee combinatorial background
    dataset_params.kstar_jpsi_kaon_file # kstar_kaon
    dataset_params.kstar_jpsi_pion_file # kstar_pion
    dataset_params.k0star_jpsi_kaon_file # k0star_kaon
    dataset_params.k0star_jpsi_pion_file # k0star_pion
    dataset_params.chic1_jpsi_kaon_file chic1_kaon
    dataset_params.jpsipi_jpsi_kaon_file jpsipi_kaon

    file_name = [dataset_params.jpsi_file, # mc_signal
                 dataset_params.samesign_data_file, # samesign ee combinatorial (no mc_weight_branch -> trig_wgt)
                 dataset_params.kstar_jpsi_kaon_file,
                 dataset_params.kstar_jpsi_pion_file,
                 dataset_params.k0star_jpsi_kaon_file,
                 dataset_params.k0star_jpsi_pion_file,
                 dataset_params.chic1_jpsi_kaon_file,
                 dataset_params.jpsipi_jpsi_kaon_file 
                ]
    treename = dataset_params.tree_name
    score_cut = None
    cutstring = '{}>{}&&{}>{}&&{}<{}'.format(
                                            dataset_params.score_branch,
                                            fit_params.bdt_score_cut if score_cut is None else score_cut,
                                            dataset_params.ll_mass_branch,
                                            fit_params.region['ll_mass_range'][0],
                                            dataset_params.ll_mass_branch,
                                            fit_params.region['ll_mass_range'][1],
                                            )
    # extract yields ino dicts
    yields_dict = {}
    for file in file_name:
        filename_str = str(file).split('/')[-1]  # Extracts the filename
        # if (filename_str)
        mc_decay = filename_str.replace("measurement_", "").replace(".root","").replace("_D0_cut","").replace("_wide_fit", "").replace("_slimmed", "").replace("_corrected", "").replace("_recorrected","")
        yield_name = f"yield_{mc_decay}"
        if "same_sign_electrons" in yield_name: # no mc_trig_wgt
            yield_num = get_MC_yield_cut_and_count(file, tree_name=treename, cut_string=cutstring, weights=None)
        else:
            yield_num = get_MC_yield_cut_and_count(file, tree_name=treename, cut_string=cutstring)
        print(yield_name, yield_num)
        yields_dict[yield_name] = yield_num
    # partial background
    yields_dict['yield_part_bkg_kstar'] = yields_dict['yield_kstar_jpsi_kaon'] + yields_dict['yield_kstar_jpsi_pion'] + \
                                    yields_dict['yield_k0star_jpsi_kaon'] + yields_dict['yield_k0star_jpsi_pion'] # all kstar

    yields_dict['yield_part_bkg'] = yields_dict['yield_part_bkg_kstar'] + yields_dict['yield_chic1_jpsi_kaon'] # kstar + chic1_kaon

    # save to csv
    save_yields_err(yields_dict, output_params, 'mc_yields_2.9.csv')


    # V2?
    ### MC Cut and Count ###
    file_name = [dataset_params.jpsi_file, # mc_signal
                 dataset_params.kstar_jpsi_kaon_file,
                 dataset_params.kstar_jpsi_pion_file,
                 dataset_params.k0star_jpsi_kaon_file,
                 dataset_params.k0star_jpsi_pion_file,
                 dataset_params.chic1_jpsi_kaon_file,
                 dataset_params.jpsipi_jpsi_kaon_file
                ]
    treename = dataset_params.tree_name
    score_cut = None

    low_q2_window = [2.9, 2.8 , 2.7, 2.6, 2.5]
    for q2 in low_q2_window:
        fit_params.ll_mass_range[0] = q2
        # print(output_params.output_dir)
        tmp_output_params = copy.deepcopy(output_params)
        tmp_output_params.output_dir = output_params.output_dir+str(q2)
        print(tmp_output_params.output_dir)

        cutstring = '{}>{}&&{}>{}&&{}<{}'.format(
                                                dataset_params.score_branch,
                                                fit_params.bdt_score_cut if score_cut is None else score_cut,
                                                dataset_params.ll_mass_branch,
                                                fit_params.region['ll_mass_range'][0],
                                                dataset_params.ll_mass_branch,
                                                fit_params.region['ll_mass_range'][1],
                                                )
        # extract yields ino dicts
        yields_dict = {}
        for file in file_name:
            filename_str = str(file).split('/')[-1]  # Extracts the filename
            # if (filename_str)
            mc_decay = filename_str.replace("measurement_", "").replace(".root","").replace("_D0_cut","").replace("_wide_fit", "").replace("_slimmed", "").replace("_corrected", "").replace("_recorrected","")
            yield_name = f"yield_{mc_decay}"
            yield_num = get_MC_yield_cut_and_count(file, tree_name=treename, cut_string=cutstring)
            print(yield_name, yield_num)
            yields_dict[yield_name] = yield_num
        # partial background
        yields_dict['yield_part_bkg'] = yields_dict['yield_kstar_jpsi_kaon'] + yields_dict['yield_kstar_jpsi_pion'] + \
                                        yields_dict['yield_k0star_jpsi_kaon'] + yields_dict['yield_k0star_jpsi_pion'] + yields_dict['yield_chic1_jpsi_kaon']

        # save to csv
        save_yields_err(yields_dict, tmp_output_params, 'mc_yield.csv')






















# #===========================================================================================================
# Fit Parameterization systematics: SIGNAL dcb+dcb --> cbcb, cbgauss, gausscb + COMBINATORIAL exp --> 2nd/4th polynomial 
# #===========================================================================================================
    # tmp_output_params = copy.deepcopy(output_params)
    # tmp_output_params.output_dir = 'dcbdcb_sig/fitter_outputs/'
    # jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

    # tmp_output_params = copy.deepcopy(output_params)
    # tmp_output_params.output_dir = 'dcbdcb_sig/fitter_outputs_wCoeffCon/'
    # jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=False)

    # tmp_output_params = copy.deepcopy(output_params)
    # tmp_output_params.output_dir = 'gausscb_sig/fitter_outputs/'
    # jpsi_fit_cbgauss_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)
    
    # tmp_output_params = copy.deepcopy(output_params)
    # tmp_output_params.output_dir = 'cbcb_sig/fitter_outputs/'
    # jpsi_fit_cbcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

    # tmp_output_params = copy.deepcopy(output_params)
    # tmp_output_params.output_dir = 'poly_2nd_comb/fitter_outputs/'
    # jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

# #===========================================================================================================
# Fit Parameterization systematics: same shape script but wCoeff/noCoeff minos/noMinos runs
# #===========================================================================================================
    dir_path = output_params.output_dir

    print(dir_path)
    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = dir_path+'fitter_outputs/'
    jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)
    print(dir_path)
    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = dir_path+'fitter_outputs_wCoeffCon/'
    jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=False)
    print(dir_path)
    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = dir_path+'fitter_outputs_minos/'
    jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=True)
    print(dir_path)
    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = dir_path+'fitter_outputs_wCoeffCon_minos/'
    jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=True)
    
# #===========================================================================================================
# Fit Parameterization systematics: partial reconstructed background --> up/down sampled for k(0)star --> pion/kaon
# #===========================================================================================================
    '''
    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = 'partial_bkg/kaon_pion_switched/fitter_outputs'
    jpsi_fit_kde_partial_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = 'partial_bkg/kaon_pion_switched/fitter_outputs_wCoeffCon'
    jpsi_fit_kde_partial_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=False)

    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = 'partial_bkg/kaon_pion_switched/fitter_outputs_minos'
    jpsi_fit_kde_partial_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=True)

    tmp_output_params = copy.deepcopy(output_params)
    tmp_output_params.output_dir = 'partial_bkg/kaon_pion_switched/fitter_outputs_wCoeffCon_minos'
    jpsi_fit_kde_partial_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=True)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which fit to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-t', '--toy_fit', dest='toy_fit', action='store_true', help='fit toy data in low-q2')
    args = parser.parse_args()

    main(args)
