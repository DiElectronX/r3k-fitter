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

def jpsi_fit_cbgauss_sig(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False, coeffCon=False, isMinos=False):
    '''
    The standard full fitting with 
        signal: cb+gaauss (NOTE: whether which one is the main/tail depends on the initial guess in fit_cfg_parameterization.yml)
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
        model_sig_template.add_signal_model('sig_pdf', 'cb+gauss', fit_params.fit_defaults, let_float=True)
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
                model_sig_template.signal_models['sig_pdf'].cb_pdf,
                model_sig_template.signal_models['sig_pdf'].gauss_pdf,
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
        file_path = 'gausscb_sig/fitter_outputs_wCoeffCons/fit_jpsi_template.yml'
        with open(file_path, 'r') as file:
        # with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            print('Load template: ', file.name)
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'cb+gauss', template, let_float=False)
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
    model_final.signal_models['sig_pdf'].gauss_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].gauss_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].gauss_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].cb_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].cb_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].cb_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    part_bkg_coeff.setConstant(False)
    comb_bkg_coeff.setConstant(False)

    #part_bkg_jpsipi_pion_coeff.setConstant(False)
    #jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        #'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        # 'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(2.)),
        
        'gauss_mean_constraint' : ROOT.RooGaussian('gauss_mean_constraint', 'gauss_mean_constraint', model_final.signal_models['sig_pdf'].gauss_mean, ROOT.RooFit.RooConst(template['gauss_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'gauss_sigma_constraint' : ROOT.RooGaussian('gauss_sigma_constraint', 'gauss_sigma_constraint', model_final.signal_models['sig_pdf'].gauss_sigma, ROOT.RooFit.RooConst(template['gauss_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'cb_mean_constraint' : ROOT.RooGaussian('cb_mean_constraint', 'cb_mean_constraint', model_final.signal_models['sig_pdf'].cb_mean, ROOT.RooFit.RooConst(template['cb_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'cb_sigma_constraint' : ROOT.RooGaussian('cb_sigma_constraint', 'cb_sigma_constraint', model_final.signal_models['sig_pdf'].cb_sigma, ROOT.RooFit.RooConst(template['cb_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
    })
    if coeffCon:
        model_final.add_constraints({
            'gauss_coeff_constraint' : ROOT.RooGaussian('gauss_coeff_constraint', 'gauss_coeff_constraint', model_final.signal_models['sig_pdf'].gauss_coeff, ROOT.RooFit.RooConst(template['gauss_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['gauss_coeff_sig_pdf']*.05)),
            'cb_coeff_constraint' : ROOT.RooGaussian('cb_coeff_constraint', 'cb_coeff_constraint', model_final.signal_models['sig_pdf'].cb_coeff, ROOT.RooFit.RooConst(template['cb_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['cb_coeff_sig_pdf']*.05)),
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
            # model_final.signal_models['sig_pdf'].cb_pdf,
            # model_final.signal_models['sig_pdf'].gauss_pdf,
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
    _cb_coeff = model_final.signal_models['sig_pdf'].cb_coeff.getVal()
    _gauss_coeff = model_final.signal_models['sig_pdf'].gauss_coeff.getVal()
    _norm_sf = 1 / (_cb_coeff + _gauss_coeff)
    model_final.signal_models['sig_pdf'].cb_coeff.setVal(_cb_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].gauss_coeff.setVal(_gauss_coeff * _norm_sf)

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

def jpsi_fit_cbcb_sig(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False, coeffCon=False, isMinos=False):
    '''
    The standard full fitting with 
        signal: cb+cb
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
        model_sig_template.add_signal_model('sig_pdf', 'cb+cb', fit_params.fit_defaults, let_float=True)
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
                model_sig_template.signal_models['sig_pdf'].cb1_pdf,
                model_sig_template.signal_models['sig_pdf'].cb2_pdf,
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
        filepath = 'cbcb_sig/fitter_outputs_wCoeffCons/fit_jpsi_template.yml'
        with open(filepath, 'r') as file:
        # with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            print('Load template: ', file.name)
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'cb+cb', template, let_float=False)
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
    model_final.signal_models['sig_pdf'].cb1_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].cb2_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].cb1_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].cb2_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].cb1_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].cb2_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    part_bkg_coeff.setConstant(False)
    comb_bkg_coeff.setConstant(False)
    #part_bkg_jpsipi_pion_coeff.setConstant(False)
    #jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        #'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        # 'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(2.)),
        
        'cb1_mean_constraint' : ROOT.RooGaussian('cb1_mean_constraint', 'cb1_mean_constraint', model_final.signal_models['sig_pdf'].cb1_mean, ROOT.RooFit.RooConst(template['cb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'cb1_sigma_constraint' : ROOT.RooGaussian('cb1_sigma_constraint', 'cb1_sigma_constraint', model_final.signal_models['sig_pdf'].cb1_sigma, ROOT.RooFit.RooConst(template['cb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'cb2_mean_constraint' : ROOT.RooGaussian('cb2_mean_constraint', 'cb2_mean_constraint', model_final.signal_models['sig_pdf'].cb2_mean, ROOT.RooFit.RooConst(template['cb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'cb2_sigma_constraint' : ROOT.RooGaussian('cb2_sigma_constraint', 'cb2_sigma_constraint', model_final.signal_models['sig_pdf'].cb2_sigma, ROOT.RooFit.RooConst(template['cb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
    })
    if coeffCon:
        model_final.add_constraints({
            'cb1_coeff_constraint' : ROOT.RooGaussian('cb1_coeff_constraint', 'cb1_coeff_constraint', model_final.signal_models['sig_pdf'].cb1_coeff, ROOT.RooFit.RooConst(template['cb1_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['cb1_coeff_sig_pdf']*.05)),
            'cb2_coeff_constraint' : ROOT.RooGaussian('cb2_coeff_constraint', 'cb2_coeff_constraint', model_final.signal_models['sig_pdf'].cb2_coeff, ROOT.RooFit.RooConst(template['cb2_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['cb2_coeff_sig_pdf']*.05)),
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
            # model_final.signal_models['sig_pdf'].cb1_pdf,
            # model_final.signal_models['sig_pdf'].cb2_pdf,
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
    _cb1_coeff = model_final.signal_models['sig_pdf'].cb1_coeff.getVal()
    _cb2_coeff = model_final.signal_models['sig_pdf'].cb2_coeff.getVal()
    _norm_sf = 1 / (_cb1_coeff + _cb2_coeff)
    model_final.signal_models['sig_pdf'].cb1_coeff.setVal(_cb1_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].cb2_coeff.setVal(_cb2_coeff * _norm_sf)

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

def jpsi_fit_poly_comb_bkg(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False, coeffCon=False, isMinos=False):
    # NOTE: donmt forget to take outt coeffCon and isMinos flag
    '''
    The standard full fitting with 
        signal: dcb+dcb
        combinatorial: 2nd or 4th degree polynomial
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
        model_comb_template.add_background_model('comb_bkg_pdf', 'poly', fit_params.fit_defaults, let_float=True)
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
        filepath = 'poly_comb/fitter_outputs_minos/fit_jpsi_template.yml'
        with open(filepath, 'r') as file:
        # with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            print('Load template: ', file.name)
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'poly', template, let_float=False)
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

    model_final.background_models['comb_bkg_pdf'].poly_a0.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].poly_a1.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].poly_a2.setConstant(False)
    # model_final.background_models['comb_bkg_pdf'].poly_a3.setConstant(False) # cross this out if 2nd order polynomial
    # model_final.background_models['comb_bkg_pdf'].poly_a4.setConstant(False) # cross this out if 2nd order polynomial
    model_final.background_models['comb_bkg_pdf'].poly_offset.setConstant(False)
    
    part_bkg_coeff.setConstant(False)
    comb_bkg_coeff.setConstant(False)
    #part_bkg_jpsipi_pion_coeff.setConstant(False)
    #jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        #'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),    
    })
    if coeffCon: # coeffCon flag
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

# #===========================================================================================================
# Fit Parameterization systematics: SIGNAL dcb+dcb --> cbcb, cbgauss, gausscb + COMBINATORIAL exp --> 2nd/4th polynomial 
# #===========================================================================================================
    tmp_output_params = copy.deepcopy(output_params)

    # tmp_output_params.output_dir = 'dcbdcb_sig/fitter_outputs_wCoeffCons/'
    # jpsi_fit_dcbdcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True)

    # tmp_output_params.output_dir = 'gausscb_sig/fitter_outputs/'
    # jpsi_fit_cbgauss_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)
    
    tmp_output_params.output_dir = 'cbcb_sig/fitter_outputs/'
    jpsi_fit_cbcb_sig(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

    # tmp_output_params.output_dir = 'poly_comb/fitter_outputs_wCoeffCons/'
    # jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True)

# #===========================================================================================================
# Fit Parameterization systematics: same shape script but wCoeff/noCoeff minos/noMinos runs
# #===========================================================================================================
    '''
    tmp_output_params = copy.deepcopy(output_params)

    tmp_output_params.output_dir = 'poly_4ht_comb/fitter_outputs/'
    jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=False)

    tmp_output_params.output_dir = 'poly_4ht_comb/fitter_outputs_wCoeffCons/'
    jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=False)

    tmp_output_params.output_dir = 'poly_4ht_comb/fitter_outputs_minos/'
    jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=False, isMinos=True)

    tmp_output_params.output_dir = 'poly_4ht_comb/fitter_outputs_wCoeffCons_minos/'
    jpsi_fit_poly_comb_bkg(dataset_params, tmp_output_params, fit_params, args, get_yields=True, coeffCon=True, isMinos=True)
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
