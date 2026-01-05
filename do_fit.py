import ROOT
import yaml
import argparse
from pathlib import Path
import numpy as np
from pprint import pprint

from physics_constants import SAMPLES, get_mc_scale_factor
from fit_models import FitModel, PDFDictWrapper
from utils import set_verbosity, makedirs, set_mode, prepare_inputs, save_params, integrate, calculate_yields, write_workspace, load_template_from_file

ROOT.gErrorIgnoreLevel = ROOT.kError
ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']


def do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None,  toy_fit=True, unblinded=False, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)
    b_mass_branch.setRange('semilow', 4.65, 5.7)
    b_mass_branch.setRange('sb1', fit_params.fit_range[0], fit_params.blinded[0])
    b_mass_branch.setRange('sb2', fit_params.blinded[1], fit_params.fit_range[1])

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_rare = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name=dataset_params.mc_weight_branch)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_rare, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_rare, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_rare,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_samesign_data = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0., unblind=True)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_samesign_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_samesign_data, use_minos=True if args.minos else False, printlevel=printlevel, fit_range='semilow', fit_norm_range='semilow')
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_samesign_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            bins=30,
            file_label=file_label,
            fit_range='semilow',
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit jpsi leakage in low-q2 region from MC
    if args.verbose:
        print('\nStarting Fit 3 - J/Psi Leakage Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    _, dataset_jpsi = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.jpsi_file, weight_branch_name=dataset_params.mc_weight_branch)

    # Build Roofit model for exponential background
    model_jpsi_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_jpsi, 'channel_label': fit_params.channel_label})
    model_jpsi_template.add_background_model('jpsi_bkg_pdf', 'gauss', fit_params.fit_defaults, let_float=True)
    model_jpsi_template.fit_model = model_jpsi_template.jpsi_bkg_pdf

    # Fit model to data
    model_jpsi_template.fit(dataset_jpsi, use_minos=True if args.minos else False, fit_range='low', fit_norm_range='low', printlevel=printlevel)
    params = model_jpsi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsi_template.plot_fit(
        b_mass_branch,
        dataset_jpsi,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsi_template.pdf',
        file_label=file_label,
        fit_range='low',
        fit_result=model_jpsi_template.fit_result,
        bins=35,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 4 - KStar Partial Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    _, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.kstar_pion_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.k0star_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.k0star_pion_file, weight_branch_name=dataset_params.mc_weight_branch)
    dataset_kstar_comb = dataset_kstar_pion.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)

    if args.verbose:
        print('nEvents for K*+ -> piee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for K*0 -> piee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for K*0 -> Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6, .5, .85, .85)
    tmp_frame = b_mass_branch.frame()

    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination', 'Combination', 'lpe')
    l1.SetLineColor(ROOT.kBlack)
    l1.SetMarkerColor(ROOT.kBlack)
    l3 = leg.AddEntry('kstar_pion', 'kstar_pion', 'lpe')
    l3.SetLineColor(ROOT.kBlue)
    l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon', 'k0star_kaon + kstar_kaon', 'lpe')
    l4.SetLineColor(ROOT.kRed)
    l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion', 'k0star_pion', 'lpe')
    l5.SetLineColor(ROOT.kGreen)
    l5.SetMarkerColor(ROOT.kGreen)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(str(Path(output_params.output_dir) / f'fit_{args.mode}_kstar_combination_dataset.pdf'))
    tmp_c.Close()

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_kstar_comb, 'channel_label': fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        Path(output_params.output_dir) / f'fit_{args.mode}_kstar_partial_template.pdf',
        file_label=file_label,
        bins=30,
    )

    # Add template for final fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    comb_bkg_norm = 170
    part_bkg_norm = 23
    jpsi_bkg_norm = 62378*0.0003865566637

    if args.cache:
        # Load fit shape templates from file
        with open(Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=True)

    # Use toys to produce expected signal
    if toy_fit:
        # Fit background-only model to data sidebands
        bkg_only_model = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        bkg_only_model.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        bkg_only_model.add_background_model('jpsi_bkg_pdf', 'gauss', template, let_float=False)
        bkg_only_model.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])
        comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', comb_bkg_norm, 0., 1E8)
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm,  0, 1E8)
        part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', part_bkg_norm, 0, 1E8)
        bkg_only_model.fit_model = ROOT.RooAddPdf(
            'bkg_only_pdf',
            'Sum of Background PDFs',
            ROOT.RooArgList(
                bkg_only_model.comb_bkg_pdf,
                bkg_only_model.jpsi_bkg_pdf,
                bkg_only_model.part_bkg_pdf,
            ),
            ROOT.RooArgList(
                comb_bkg_coeff,
                jpsi_bkg_coeff,
                part_bkg_coeff,
            )
        )

        comb_bkg_coeff.setConstant(False)
        # jpsi_bkg_coeff.setConstant(False)
        part_bkg_coeff.setConstant(False)
        bkg_only_model.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)

        bkg_only_model.constraints.update({
            'part_bkg_coeff_constraint': ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_coeff, ROOT.RooFit.RooConst(part_bkg_coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_coeff.getVal()*.2)),
            # 'exp_slope_comb_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_comb_bkg_pdf_constraint', 'exp_slope_comb_bkg_pdf_constraint', bkg_only_model.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(5)),
        })

        bkg_only_model.fit(dataset_data, use_minos=True if args.minos else False, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)
        params = bkg_only_model.fit_result.floatParsFinal()
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

        bkg_only_model.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_bkg_only.pdf',
            file_label=file_label,
            fit_components={
                'Combinatorial Bkg.': bkg_only_model.comb_bkg_pdf,
                'Part.-Reco. Bkg.': bkg_only_model.part_bkg_pdf,
                'B #rightarrow J/#psi K Bkg.': bkg_only_model.jpsi_bkg_pdf,
            },
            fit_range='full',
            fit_norm_range='sb1,sb2',
            fit_result=bkg_only_model.fit_result,
            bins=35,
            legend=True,
            yrange=[0, 100],
        )

        # Generate expected background from sideband fit
        expected_bkg, _ = integrate(
            b_mass_branch,
            bkg_only_model.fit_model,
            [4.5, 5.7],
            coeffs=[comb_bkg_coeff, jpsi_bkg_coeff, part_bkg_coeff],
        )
        toy_background = bkg_only_model.fit_model.generate(ROOT.RooArgSet(b_mass_branch), expected_bkg)

        # Generate expected signal from MC shape and jpsi-extrapolated yield
        if args.cache:
            _, dataset_rare = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name=dataset_params.mc_weight_branch)
            model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_rare, 'channel_label': fit_params.channel_label})
            model_sig_template.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
            model_sig_template.fit_model = model_sig_template.sig_pdf

        toy_signal = model_sig_template.fit_model.generate(ROOT.RooArgSet(b_mass_branch), fit_params.toy_signal_yield)

        # Create toy dataset for final fit
        toy_dataset = dataset_data.emptyClone('dataset_data'+fit_params.channel_label, 'Toy Dataset (S+B)')
        toy_dataset.append(toy_background)
        toy_dataset.append(toy_signal)

        # Toy dataset plot
        tmp_frame = b_mass_branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        dataset_data.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack), ROOT.RooFit.Name('ds'))
        toy_background.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue), ROOT.RooFit.Name('tb'))
        toy_signal.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed), ROOT.RooFit.Name('ts'))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.MarkerColor(ROOT.kCyan), ROOT.RooFit.Name('td'))
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kMagenta), ROOT.RooFit.Name('f'))
        comp = ROOT.RooArgSet(bkg_only_model.comb_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name('f_comb'))
        comp = ROOT.RooArgSet(bkg_only_model.jpsi_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name('f_jpsi'))
        comp = ROOT.RooArgSet(bkg_only_model.part_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid), ROOT.RooFit.LineColor(ROOT.kViolet), ROOT.RooFit.Name('f_part'))

        legend = ROOT.TLegend(0.6, 0.6, 0.9, 0.9)
        legend.AddEntry(tmp_frame.findObject('tb'), 'Toy Background', 'LPE')
        legend.AddEntry(tmp_frame.findObject('ts'), 'Toy Signal', 'LPE')
        legend.AddEntry(tmp_frame.findObject('td'), 'Toy Dataset (S+B)', 'LPE')
        legend.AddEntry(tmp_frame.findObject('ds'), 'Blinded Data', 'LPE')
        legend.AddEntry(tmp_frame.findObject('f'), 'Bkg.-Only Fit', 'L')
        legend.AddEntry(tmp_frame.findObject('f_comb'), 'Bkg.-Only Fit (Comb.)', 'L')
        legend.AddEntry(tmp_frame.findObject('f_jpsi'), 'Bkg.-Only Fit (Jpsi)', 'L')
        legend.AddEntry(tmp_frame.findObject('f_part'), 'Bkg.-Only Fit (Part.-Reco.)', 'L')

        tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
        tmp_frame.Draw()
        tmp_frame.GetYaxis().SetRangeUser(0, 80)
        legend.Draw()
        tmp_c.SaveAs(str(Path(output_params.output_dir) / f'fit_{args.mode}_toy_dataset.pdf'))
        tmp_c.Close()

        dataset_data = toy_dataset

    # Build final Roofit model
    model_final = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})

    if toy_fit:
        model_final.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
    model_final.add_background_model('jpsi_bkg_pdf', 'gauss', template, let_float=False)
    # model_final.add_background_model('jpsi_bkg_pdf', model_jpsi_template.background_models['jpsi_bkg_pdf'])
    model_final.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])

    if toy_fit:
        sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 101., 0., 5*dataset_data.numEntries())
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)
    else:
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)

    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 500, 0., 5*dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 300, 0, dataset_data.numEntries())

    model_comps = {
        'Combinatorial Bkg.': model_final.comb_bkg_pdf,
        'B #rightarrow J/#psi K Leakage': model_final.jpsi_bkg_pdf,
        'Part.-Reco. Bkg.': model_final.part_bkg_pdf,
    }

    model_coeffs = [comb_bkg_coeff, jpsi_bkg_coeff, part_bkg_coeff]
    if toy_fit:
        model_comps['B #rightarrow eeK (toy)'] = model_final.sig_pdf
        model_coeffs.append(sig_coeff)

    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(*model_comps.values()),
        ROOT.RooArgList(*model_coeffs)
    )

    # Add gaussian contraints to fit parameters
    # part_bkg_coeff.setConstant(False)
    # comb_bkg_coeff.setConstant(False)
    # jpsi_bkg_coeff.setConstant(False)
    # model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)

    if toy_fit:
        model_final.signal_models['sig_pdf'].sig_coeff.setConstant(False)
        # model_final.signal_models['sig_pdf'].dcb_mean.setConstant(False)
        # model_final.signal_models['sig_pdf'].dcb_sigma.setConstant(False)
        # jpsi_ratio = ROOT.RooFormulaVar('jpsi_ratio', 'Ratio of JPsi leakage', '0.2413793103*@0', ROOT.RooArgList(sig_coeff))
    else:
        pass
        # jpsi_ratio = ROOT.RooFormulaVar('jpsi_ratio', 'Ratio of JPsi leakage', '@0/@1', ROOT.RooArgList(part_bkg_coeff, jpsi_bkg_coeff))

    # Add gaussian contraints to fit parameters
    model_final.constraints.update({
        # 'part_bkg_coeff_constraint' : ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_coeff, ROOT.RooFit.RooConst(part_bkg_coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_coeff.getVal()*0.1)),
        # 'exp_slope_comb_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_comb_bkg_pdf_constraint', 'exp_slope_comb_bkg_pdf_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(5)),
        # 'exp_slope_jpsi_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_jpsi_bkg_pdf_constraint', 'exp_slope_jpsi_bkg_pdf_constraint', model_final.background_models['jpsi_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_jpsi_bkg_pdf']), ROOT.RooFit.RooConst(0.5)),
    })
    if toy_fit:
        model_final.constraints.update({
            # 'dcb_mean_constraint' : ROOT.RooGaussian('dcb_mean_constraint', 'dcb_mean_constraint', model_final.signal_models['sig_pdf'].dcb_mean, ROOT.RooFit.RooConst(template['dcb_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
            # 'dcb_sigma_constraint' : ROOT.RooGaussian('dcb_sigma_constraint', 'dcb_sigma_constraint', model_final.signal_models['sig_pdf'].dcb_sigma, ROOT.RooFit.RooConst(template['dcb_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
            # 'jpsi_ratio_constraint' : ROOT.RooGaussian('jpsi_ratio_constraint', 'jpsi_ratio_constraint', jpsi_ratio, ROOT.RooFit.RooConst(jpsi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        })

    # Fit model to data
    fit_range = 'full' if toy_fit else 'sb1,sb2'
    fit_norm_range = 'full' if toy_fit else 'sb1,sb2'
    model_final.fit(dataset_data, use_minos=True if args.minos else False, fit_range=fit_range, fit_norm_range=fit_norm_range, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Define the component map specific to the lowq2 fit
    component_map = {
        'yield_comb_bkg': (model_final.comb_bkg_pdf, comb_bkg_coeff),
        'yield_part_bkg': (model_final.part_bkg_pdf, part_bkg_coeff),
        'yield_jpsi_bkg': (model_final.jpsi_bkg_pdf, jpsi_bkg_coeff),
    }
    # Conditionally add the signal component if doing a toy fit
    if toy_fit:
        component_map['yield_sig'] = (model_final.sig_pdf, sig_coeff)

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # 3. Use the results to create plot text and then plot the model
    if toy_fit:
        signal_yield = yields['yield_sig']
        sig_range = (custom_yield_ranges or {}).get('yield_sig')

        yield_text = f'N_{{B #rightarrow eeK}} = {signal_yield[0]} #pm {signal_yield[1]}'
        if sig_range:
            yield_text = f'N_{{B #rightarrow eeK}} [{sig_range[0]}-{sig_range[1]} GeV] = {signal_yield[0]} #pm {signal_yield[1]}'
    else:
        # If not a toy fit, calculate total background in the blinded region for the plot label
        bkg_yields_in_blinded_region = calculate_yields(
            b_mass_branch, component_map, fit_params.blinded, model_final.fit_result
        )
        total_bkg_val = sum(val for val, err in bkg_yields_in_blinded_region.values())
        total_bkg_err = np.sqrt(sum(err**2 for val, err in bkg_yields_in_blinded_region.values()))
        yield_text = f'N_{{Bkg}} [{fit_params.blinded[0]}-{fit_params.blinded[1]} GeV] = {round(total_bkg_val)} #pm {round(total_bkg_err, 2)}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode+("_toy" if toy_fit else "")}_final.pdf',
        file_label=file_label,
        fit_components=model_comps,
        fit_range='full',
        fit_norm_range=fit_norm_range,
        fit_result=model_final.fit_result,
        bins=35,
        legend='ul',
        yrange=[0, 180],
        stat_text_pos='middle',
        extra_text=yield_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar(f'comb_bkg_pdf{fit_params.channel_label}_norm', 'Number of combinatorial background events', yields['yield_comb_bkg'][0], 0, 999999)
    part_bkg_pdf_norm = ROOT.RooRealVar(f'part_bkg_pdf{fit_params.channel_label}_norm', 'Number of partially reconstructed background events', yields['yield_part_bkg'][0], 0, 999999)
    jpsi_bkg_pdf_norm = ROOT.RooRealVar(f'jpsi_bkg_pdf{fit_params.channel_label}_norm', 'Number of jpsi low-q2 background events', yields['yield_jpsi_bkg'][0], 0, 999999)

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsi_bkg_pdf_norm]
        if toy_fit:
            sig_pdf_norm = ROOT.RooRealVar(f'sig_pdf{fit_params.channel_label}_norm', 'Number of signal events', yields['yield_sig'][0], 0, 999999)
            extra_objects.append(sig_pdf_norm)
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    if get_yields:
        return yields
    else:
        pprint(yields)


def do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_mc = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name=dataset_params.mc_weight_branch)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))

    # Look at partial shape files
    tmp_b_mass_branch, dataset_kstar_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_pion_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_pion_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_chic1_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.chic1_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_jpsipi_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    dataset_kstar_comb = dataset_kstar_kaon.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_kstar_pion)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)
    dataset_kstar_comb.append(dataset_chic1_kaon)
    # dataset_kstar_comb.append(dataset_jpsipi_pion)

    mc_yield_tot = dataset_kstar_comb.sumEntries()

    if mc_yield_tot:
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
    else:
        kstar_kaon_yield_frac = 0
        kstar_pion_yield_frac = 0
        k0star_kaon_yield_frac = 0
        k0star_pion_yield_frac = 0
        chic1_kaon_yield_frac = 0
        kstar_yield_frac = 0

    if args.verbose:
        print('nEvents for B+ -> J/ψ K*+ - Kee cand = {}'.format(dataset_kstar_kaon.sumEntries()))
        print('nEvents for B+ -> J/ψ K*+ - πee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for B0 -> J/ψ K*0 - Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))
        print('nEvents for B0 -> J/ψ K*0 - πee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for B+ -> χc1 K+  - Kee cand = {}'.format(dataset_chic1_kaon.sumEntries()))
        print('nEvents for B+ -> J/ψ π+  - πee cand = {}'.format(dataset_jpsipi_pion.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6, .5, .85, .85)
    tmp_frame = tmp_b_mass_branch.frame()

    dataset_kstar_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_kaon'), ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.MarkerColor(ROOT.kOrange))
    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_chic1_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('chic1_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kMagenta), ROOT.RooFit.MarkerColor(ROOT.kMagenta))
    dataset_jpsipi_pion.plotOn(tmp_frame, ROOT.RooFit.Name('jpsipi_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.MarkerColor(ROOT.kCyan))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination', 'combination', 'lpe'); l1.SetLineColor(ROOT.kBlack); l1.SetMarkerColor(ROOT.kBlack)
    l2 = leg.AddEntry('kstar_kaon', 'kstar_kaon', 'lpe'); l2.SetLineColor(ROOT.kOrange); l2.SetMarkerColor(ROOT.kOrange)
    l3 = leg.AddEntry('kstar_pion', 'kstar_pion', 'lpe'); l3.SetLineColor(ROOT.kBlue); l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon', 'k0star_kaon', 'lpe'); l4.SetLineColor(ROOT.kRed); l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion', 'k0star_pion', 'lpe'); l5.SetLineColor(ROOT.kGreen); l5.SetMarkerColor(ROOT.kGreen)
    l6 = leg.AddEntry('chic1_kaon', 'chic1_kaon', 'lpe'); l6.SetLineColor(ROOT.kMagenta); l6.SetMarkerColor(ROOT.kMagenta)
    l7 = leg.AddEntry('jpsipi_pion', 'jpsipi_pion', 'lpe'); l7.SetLineColor(ROOT.kCyan); l7.SetMarkerColor(ROOT.kCyan)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(str(Path(output_params.output_dir) / 'jpsi_dataset_partial_bkgs.pdf'))
    tmp_c.Close()

    # Import ROOT file dataset
    model_part_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_kstar_comb, 'channel_label': fit_params.channel_label})
    model_part_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_part_template.fit_model = model_part_template.part_bkg_pdf

    # Plot fit result
    model_part_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        file_label=file_label,
        bins=30,
    )
    # Fit partial background shape to jpsipi MC
    if args.verbose:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))

    _, dataset_jpsipi_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    model_jpsipi_pion_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_jpsipi_pion, 'channel_label': fit_params.channel_label})
    model_jpsipi_pion_template.add_background_model('part_bkg_pdf_jpsipi_pion', 'dcb', fit_params.fit_defaults, let_float=True)
    model_jpsipi_pion_template.fit_model = model_jpsipi_pion_template.part_bkg_pdf_jpsipi_pion

    # Fit model to data
    model_jpsipi_pion_template.fit(dataset_jpsipi_pion, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_jpsipi_pion_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsipi_pion_template.plot_fit(
        b_mass_branch,
        dataset_jpsipi_pion,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsipi_template.pdf',
        file_label=file_label,
        fit_result=model_jpsipi_pion_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    # dataset_params.score_branch = 'bdt_score_1'
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)
    # dataset_params.score_branch = 'bdt_score'

    # Build final Roofit model
    model_final = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('part_bkg_pdf_jpsipi_pion', 'dcb', template, let_float=False)

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 60000, 0, dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 2000, 0, dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Part. Bkg. PDF Coeff.', 3000, 0, dataset_data.numEntries())
    # part_bkg_jpsipi_pion_coeff = ROOT.RooRealVar('part_bkg_jpsipi_pion_coeff'+fit_params.channel_label, 'Part. Bkg. PDF Coeff.', 2280, 0, dataset_data.numEntries())
    part_bkg_jpsipi_pion_coeff = ROOT.RooFormulaVar('part_bkg_jpsipi_pion_coeff', 'Part. Bkg. PDF Coeff.', '0.0465*@0', ROOT.RooArgList(sig_coeff))

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
    comb_bkg_coeff.setConstant(False)
    part_bkg_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_mean.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_sigma.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_sigma.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb1_coeff.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb_coeff_ratio.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    # part_bkg_jpsipi_pion_coeff.setConstant(False)
    # jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        # 'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        # 'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(2.)),
        # 'dcb_coeff_ratio_constraint' : ROOT.RooGaussian('dcb_coeff_ratio_constraint', 'dcb_coeff_ratio_constraint', model_final.signal_models['sig_pdf'].dcb_coeff_ratio, ROOT.RooFit.RooConst(template['dcb_coeff_ratio']), ROOT.RooFit.RooConst(.05*template['dcb_coeff_ratio'])),
        # 'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.02*template['dcb1_mean_sig_pdf'])),
        # 'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.02*template['dcb1_sigma_sig_pdf'])),
        # 'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.02*template['dcb2_mean_sig_pdf'])),
        # 'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.02*template['dcb2_sigma_sig_pdf'])),
    })

    # Fit model to data
    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map specific to this J/psi fit
    component_map = {
        'yield_sig': (model_final.sig_pdf, sig_coeff),
        'yield_comb_bkg': (model_final.comb_bkg_pdf, comb_bkg_coeff),
        'yield_part_bkg': (model_final.part_bkg_pdf, part_bkg_coeff),
        'yield_part_bkg_jpsipi_pion': (model_final.part_bkg_pdf_jpsipi_pion, part_bkg_jpsipi_pion_coeff),
    }
    # Add fractional components if they exist
    if mc_yield_tot > 0:
        component_map.update({
            'yield_part_bkg_kstar': (model_final.part_bkg_pdf, part_bkg_coeff, kstar_yield_frac),
            'yield_part_bkg_kstar_kaon': (model_final.part_bkg_pdf, part_bkg_coeff, kstar_kaon_yield_frac),
            'yield_part_bkg_kstar_pion': (model_final.part_bkg_pdf, part_bkg_coeff, kstar_pion_yield_frac),
            'yield_part_bkg_k0star_kaon': (model_final.part_bkg_pdf, part_bkg_coeff, k0star_kaon_yield_frac),
            'yield_part_bkg_k0star_pion': (model_final.part_bkg_pdf, part_bkg_coeff, k0star_pion_yield_frac),
            'yield_part_bkg_chic1_kaon': (model_final.part_bkg_pdf, part_bkg_coeff, chic1_kaon_yield_frac),
        })

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')  # Safely check for the custom range
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{J/#psi}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{J/#psi}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':                        model_final.sig_pdf,
            # 'Signal Comp 1':                 model_final.signal_models['sig_pdf'].dcb1_pdf,
            # 'Signal Comp 2':                 model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg.':            model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg.':              model_final.part_bkg_pdf,
            'B #rightarrow J/#psi #pi Bkg.': model_final.part_bkg_pdf_jpsipi_pion,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_jpsipi_pion_norm = ROOT.RooRealVar('part_bkg_pdf_jpsipi_pion'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_jpsipi_pion_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    # _dcb1_coeff = model_final.signal_models['sig_pdf'].dcb1_coeff.getVal()
    # _dcb2_coeff = model_final.signal_models['sig_pdf'].dcb2_coeff.getVal()
    # _norm_sf = 1 / (_dcb1_coeff + _dcb2_coeff)
    # model_final.signal_models['sig_pdf'].dcb1_coeff.setVal(_dcb1_coeff * _norm_sf)
    # model_final.signal_models['sig_pdf'].dcb2_coeff.setVal(_dcb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, part_bkg_pdf_jpsipi_pion_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, part_bkg_pdf_jpsipi_pion_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    # Use function to grab yields
    yields = {
        'yield_sig':                  signal_yield,
        'yield_comb_bkg':             (round(comb_bkg_coeff.getValV(), 2), round(comb_bkg_coeff.getError(), 2)),
        'yield_part_bkg':             (round(part_bkg_coeff.getValV(), 2), round(part_bkg_coeff.getError(), 2)),
        'yield_part_bkg_jpsipi_pion': (round(part_bkg_jpsipi_pion_coeff.getValV(), 2), 0),  # round(part_bkg_jpsipi_pion_coeff.getError(),2),
        'yield_part_bkg_kstar':       (round(part_bkg_coeff.getValV() * kstar_yield_frac, 2), round(part_bkg_coeff.getError(), 2)),
        'yield_part_bkg_kstar_kaon':  (round(part_bkg_coeff.getValV() * kstar_kaon_yield_frac, 2), round(part_bkg_coeff.getError() * kstar_kaon_yield_frac, 2)),
        'yield_part_bkg_kstar_pion':  (round(part_bkg_coeff.getValV() * kstar_pion_yield_frac, 2), round(part_bkg_coeff.getError() * kstar_pion_yield_frac, 2)),
        'yield_part_bkg_k0star_kaon': (round(part_bkg_coeff.getValV() * k0star_kaon_yield_frac, 2), round(part_bkg_coeff.getError() * k0star_kaon_yield_frac, 2)),
        'yield_part_bkg_k0star_pion': (round(part_bkg_coeff.getValV() * k0star_pion_yield_frac, 2), round(part_bkg_coeff.getError() * k0star_pion_yield_frac, 2)),
        'yield_part_bkg_chic1_kaon':  (round(part_bkg_coeff.getValV() * chic1_kaon_yield_frac, 2), round(part_bkg_coeff.getError() * chic1_kaon_yield_frac, 2)),
    }

    if get_yields:
        return yields
    else:
        pprint(yields)


def do_constrained_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        sf = get_mc_scale_factor('jpsi_resonant')
        _, dataset_mc = prepare_inputs(
            dataset_params,
            fit_params,
            b_mass_branch=b_mass_branch,
            isData=False,
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )
        total_expected_signal_yield = dataset_mc.sumEntries()

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    if not args.cache:
        # Fit combinatorial background to same-sign electron data
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape
    if args.verbose:
        print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))

    # 1. Define Components (Must match keys in config/physics_constants.py)
    partial_components = [
        'kstar_jpsi_kaon',
        'kstar_jpsi_pion',
        'k0star_jpsi_kaon',
        'k0star_jpsi_pion',
        'chic1_jpsi_kaon'
    ]

    # Initialize the template model container
    model_part_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})

    # Lists to construct the Sum PDF
    pdf_list = ROOT.RooArgList()
    coeff_list = ROOT.RooArgList()
    component_yields = {}

    # We need a list to keep the RooRealVar/ConstVar objects alive in memory
    model_part_template.memory_store = []

    # 2. Loop to build individual KDEs and calculate weights
    total_expected_partial_yield = 0

    for name in partial_components:
        sample_cfg = SAMPLES[name]

        # A. Load Weighted Dataset
        # sf calculates (Lumi * Sigma * BF) / N_gen
        sf = get_mc_scale_factor(name)

        _, ds_comp = prepare_inputs(
            dataset_params,
            fit_params,
            isData=False,
            b_mass_branch=b_mass_branch,
            set_file=getattr(dataset_params, sample_cfg['file_key']),
            weight_branch_name=dataset_params.mc_weight_branch,
            weight_sf=sf
        )

        # B. Create Individual KDE
        # This adds 'pdf_{name}' to model_part_template, allowing you to plot it later
        pdf_name = f"pdf_{name}"
        comp_params = fit_params.fit_defaults.copy()

        if 'kde_mirror_part_bkg_pdf' in comp_params:
            comp_params[f'kde_mirror_{pdf_name}'] = comp_params['kde_mirror_part_bkg_pdf']

        # For Rho (Smoothing Parameter)
        if 'kde_rho_part_bkg_pdf' in comp_params:
            comp_params[f'kde_rho_{pdf_name}'] = comp_params['kde_rho_part_bkg_pdf']

        model_part_template.add_background_model_from_scratch(
            pdf_name, 'kde', comp_params, dataset=ds_comp
        )

        # Add to list for the Sum PDF
        pdf_obj = getattr(model_part_template, pdf_name)
        pdf_list.add(pdf_obj)

        # C. Calculate Weight for the Sum
        # Since ds_comp is weighted by 'sf', sumEntries() IS the expected yield in data
        expected_yield = ds_comp.sumEntries()
        total_expected_partial_yield += expected_yield
        component_yields[name] = expected_yield

        # We use a RooConstVar because this is a rigid template; relative proportions are fixed to Theory/MC
        # (If you wanted these to float in the template creation, you'd use RooRealVar)
        coeff = ROOT.RooConstVar(f"coef_{name}", f"Expected Yield {name}", expected_yield)
        coeff_list.add(coeff)
        model_part_template.memory_store.append(coeff)

        if args.verbose:
            print(f"  > Component {name:<20}: N_exp = {expected_yield:.2f}")

    # 3. Construct the Sum PDF
    # RooAddPdf(name, title, pdfs, coefficients)
    # When len(pdfs) == len(coeffs), RooFit interprets coeffs as relative weights/yields
    part_bkg_sum_pdf = ROOT.RooAddPdf(
        'part_bkg_pdf',
        'Combined Partial Background',
        pdf_list,
        coeff_list
    )

    # 4. Inject the Sum PDF into the model wrapper
    # We use a wrapper because FitModel expects an object with a .model attribute
    # You might need to add `class PDFDictWrapper: def __init__(self, n, m): self.name=n; self.model=m` to utils.py
    model_part_template.background_models['part_bkg_pdf'] = PDFDictWrapper('part_bkg_pdf', part_bkg_sum_pdf)
    model_part_template.part_bkg_pdf = part_bkg_sum_pdf
    model_part_template.fit_model = part_bkg_sum_pdf

    # 5. Plotting with Decomposition
    # Now this block will work because `model_part_template.pdf_kstar_jpsi_kaon` exists!
    components_to_plot = {
        SAMPLES[name]['label']: getattr(model_part_template, f"pdf_{name}")
        for name in partial_components
    }

    # We need a combined dataset just for the plotting points (to show data agreement)
    # (Re-using the loop logic briefly just to merge datasets for the plot)
    dataset_merged = None
    for name in partial_components:
        sf = get_mc_scale_factor(name)
        _, ds = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=getattr(dataset_params, SAMPLES[name]['file_key']), weight_branch_name=dataset_params.mc_weight_branch, weight_sf=sf)

        if dataset_merged is None:
            dataset_merged = ds.Clone('merged_partial')
        else:
            dataset_merged.append(ds)

    model_part_template.plot_fit(
        b_mass_branch,
        dataset_merged,
        Path(output_params.output_dir) / f'fit_{args.mode}_partial_template.pdf',
        fit_components=components_to_plot,
        legend='ur',
        file_label=file_label,
        bins=30,
    )

    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    mc_yield_tot = sum(component_yields.values())
    if mc_yield_tot > 0:
        kstar_kaon_yield_frac = component_yields.get('kstar_jpsi_kaon', 0) / mc_yield_tot
        kstar_pion_yield_frac = component_yields.get('kstar_jpsi_pion', 0) / mc_yield_tot
        k0star_kaon_yield_frac = component_yields.get('k0star_jpsi_kaon', 0) / mc_yield_tot
        k0star_pion_yield_frac = component_yields.get('k0star_jpsi_pion', 0) / mc_yield_tot
        chic1_kaon_yield_frac = component_yields.get('chic1_jpsi_kaon', 0) / mc_yield_tot

        # Sum of all K* modes (everything except chic1)
        kstar_yield_frac = (component_yields.get('kstar_jpsi_kaon', 0) +
                            component_yields.get('kstar_jpsi_pion', 0) +
                            component_yields.get('k0star_jpsi_kaon', 0) +
                            component_yields.get('k0star_jpsi_pion', 0)) / mc_yield_tot
    else:
        kstar_kaon_yield_frac = 0
        kstar_pion_yield_frac = 0
        k0star_kaon_yield_frac = 0
        k0star_pion_yield_frac = 0
        chic1_kaon_yield_frac = 0
        kstar_yield_frac = 0

    # Fit partial background shape to jpsipi MC
    if args.verbose:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))

    sf = get_mc_scale_factor('jpsipi_jpsi_pion')
    _, dataset_jpsipi = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)
    total_expected_jpsipi_yield = dataset_jpsipi.sumEntries()
    model_jpsipi_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_jpsipi, 'channel_label': fit_params.channel_label})
    model_jpsipi_template.add_background_model('jpsipi_bkg_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
    model_jpsipi_template.fit_model = model_jpsipi_template.jpsipi_bkg_pdf

    # Fit model to data
    model_jpsipi_template.fit(dataset_jpsipi, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_jpsipi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsipi_template.plot_fit(
        b_mass_branch,
        dataset_jpsipi,
        Path(output_params.output_dir) / f'fit_{args.mode}_jpsipi_template.pdf',
        file_label=file_label,
        fit_result=model_jpsipi_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        template = load_template_from_file(output_params, args)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_part_template.background_models['part_bkg_pdf'])
    model_final.add_background_model('jpsipi_bkg_pdf', 'dcb', template, let_float=False)

    # Initialize coefficients
    model_final.set_yield('sig_pdf', total_expected_signal_yield, 0, dataset_data.numEntries())
    model_final.set_yield('comb_bkg_pdf', 2000, 0, dataset_data.numEntries())
    model_final.set_yield('part_bkg_pdf', total_expected_partial_yield, 0, dataset_data.numEntries())
    model_final.set_yield('jpsipi_bkg_pdf', total_expected_jpsipi_yield, 0, dataset_data.numEntries())
    model_final.build_model()

    sig_wrapper = model_final.signal_models['sig_pdf']
    comb_wrapper = model_final.background_models['comb_bkg_pdf']
    part_wrapper = model_final.background_models['part_bkg_pdf']
    jpsipi_wrapper = model_final.background_models['jpsipi_bkg_pdf']

    # Keep constrained to some of the expected yields/shapes
    sig_wrapper.coeff.setConstant(False)
    comb_wrapper.coeff.setConstant(False)
    part_wrapper.coeff.setConstant(False)
    jpsipi_wrapper.coeff.setConstant(False)
    comb_wrapper.exp_slope.setConstant(False)

    target_partial_ratio = total_expected_partial_yield / total_expected_signal_yield
    target_jpsipi_ratio = total_expected_jpsipi_yield / total_expected_signal_yield
    partial_ratio = ROOT.RooFormulaVar('partial_ratio', 'Partial Bkg / Signal', '@0/@1', ROOT.RooArgList(part_wrapper.coeff, sig_wrapper.coeff))
    jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'JpsiPi Bkg / Signal', '@0/@1', ROOT.RooArgList(jpsipi_wrapper.coeff, sig_wrapper.coeff))

    model_final.add_constraints({
        'partial_ratio_constraint': ROOT.RooGaussian('partial_ratio_constraint', 'partial_ratio_constraint', partial_ratio, ROOT.RooFit.RooConst(target_partial_ratio), ROOT.RooFit.RooConst(target_partial_ratio * 0.05)),
        'jpsipi_ratio_constraint': ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(target_jpsipi_ratio), ROOT.RooFit.RooConst(target_jpsipi_ratio * 0.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map specific to this J/psi fit
    component_map = {
        'yield_sig': (sig_wrapper.model, sig_wrapper.coeff),
        'yield_comb_bkg': (comb_wrapper.model, comb_wrapper.coeff),
        'yield_part_bkg': (part_wrapper.model, part_wrapper.coeff),
        'yield_jpsipi_bkg': (jpsipi_wrapper.model, jpsipi_wrapper.coeff),
        'yield_part_bkg_kstar': (part_wrapper.model, part_wrapper.coeff, kstar_yield_frac),
        'yield_part_bkg_kstar_kaon': (part_wrapper.model, part_wrapper.coeff, kstar_kaon_yield_frac),
        'yield_part_bkg_kstar_pion': (part_wrapper.model, part_wrapper.coeff, kstar_pion_yield_frac),
        'yield_part_bkg_k0star_kaon': (part_wrapper.model, part_wrapper.coeff, k0star_kaon_yield_frac),
        'yield_part_bkg_k0star_pion': (part_wrapper.model, part_wrapper.coeff, k0star_pion_yield_frac),
        'yield_part_bkg_chic1_kaon': (part_wrapper.model, part_wrapper.coeff, chic1_kaon_yield_frac),
    }

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{J/#psi}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{J/#psi}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':                        model_final.sig_pdf,
            'Combinatorial Bkg.':            model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg.':              model_final.part_bkg_pdf,
            'B #rightarrow J/#psi #pi Bkg.': model_final.jpsipi_bkg_pdf,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_wrapper.coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_wrapper.coeff.getVal(), 0, dataset_data.numEntries())
    jpsipi_bkg_pdf_norm = ROOT.RooRealVar('jpsipi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', jpsipi_wrapper.coeff.getVal(), 0, dataset_data.numEntries())

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm])

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsipi_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    # Use function to grab yields
    yields = {
        'yield_sig':                  signal_yield,
        'yield_comb_bkg':             (round(comb_wrapper.coeff.getValV(), 2), round(comb_wrapper.coeff.getError(), 2)),
        'yield_part_bkg':             (round(part_wrapper.coeff.getValV(), 2), round(part_wrapper.coeff.getError(), 2)),
        'yield_jpsipi_bkg':           (round(comb_wrapper.coeff.getValV(), 2), round(comb_wrapper.coeff.getError(), 2)),
        'yield_part_bkg_kstar':       (round(part_wrapper.coeff.getValV() * kstar_yield_frac, 2), round(part_wrapper.coeff.getError(), 2)),
        'yield_part_bkg_kstar_kaon':  (round(part_wrapper.coeff.getValV() * kstar_kaon_yield_frac, 2), round(part_wrapper.coeff.getError() * kstar_kaon_yield_frac, 2)),
        'yield_part_bkg_kstar_pion':  (round(part_wrapper.coeff.getValV() * kstar_pion_yield_frac, 2), round(part_wrapper.coeff.getError() * kstar_pion_yield_frac, 2)),
        'yield_part_bkg_k0star_kaon': (round(part_wrapper.coeff.getValV() * k0star_kaon_yield_frac, 2), round(part_wrapper.coeff.getError() * k0star_kaon_yield_frac, 2)),
        'yield_part_bkg_k0star_pion': (round(part_wrapper.coeff.getValV() * k0star_pion_yield_frac, 2), round(part_wrapper.coeff.getError() * k0star_pion_yield_frac, 2)),
        'yield_part_bkg_chic1_kaon':  (round(part_wrapper.coeff.getValV() * chic1_kaon_yield_frac, 2), round(part_wrapper.coeff.getError() * chic1_kaon_yield_frac, 2)),
    }

    if get_yields:
        return yields
    else:
        pprint(yields)


def do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, custom_yield_ranges=None, file_label=None, legend_text=None, param_file_lock=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Set mass branch & additional fit windows
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', 4.5, 5.7)
    b_mass_branch.setRange('full', *fit_params.fit_range)
    b_mass_branch.setRange('low', 4.5, 5.7)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_mc = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name=dataset_params.mc_weight_branch)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_mc, 'channel_label': fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            Path(output_params.output_dir) / f'fit_{args.mode}_sig_template.pdf',
            file_label=file_label,
            fit_components=[
                model_sig_template.signal_models['sig_pdf'].dcb1_pdf,
                model_sig_template.signal_models['sig_pdf'].dcb2_pdf,
            ],
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            Path(output_params.output_dir) / f'fit_{args.mode}_comb_template.pdf',
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - KStar Partial Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    tmp_b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_psi2s_pion_file, weight_branch_name=dataset_params.mc_weight_branch)
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_psi2s_kaon_file, weight_branch_name=dataset_params.mc_weight_branch)  # , extra_weight=.1)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_psi2s_pion_file, weight_branch_name=dataset_params.mc_weight_branch)  # , extra_weight=.1)
    dataset_kstar_comb = dataset_kstar_pion.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)

    mc_yield_tot = dataset_kstar_comb.sumEntries()

    kstar_pion_yield_frac = dataset_kstar_pion.sumEntries() / mc_yield_tot
    k0star_kaon_yield_frac = dataset_k0star_kaon.sumEntries() / mc_yield_tot
    k0star_pion_yield_frac = dataset_k0star_pion.sumEntries() / mc_yield_tot
    kstar_yield_frac = ((dataset_kstar_pion.sumEntries() +
                        dataset_k0star_kaon.sumEntries() +
                        dataset_k0star_pion.sumEntries()) /
                        mc_yield_tot)

    if args.verbose:
        print('nEvents for K*+ -> piee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for K*0 -> piee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for K*0 -> Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6, .5, .85, .85)
    tmp_frame = tmp_b_mass_branch.frame()

    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination', 'Combination', 'lpe'); l1.SetLineColor(ROOT.kBlack); l1.SetMarkerColor(ROOT.kBlack)
    l3 = leg.AddEntry('kstar_pion', 'kstar_pion', 'lpe'); l3.SetLineColor(ROOT.kBlue); l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon', 'k0star_kaon + kstar_kaon', 'lpe'); l4.SetLineColor(ROOT.kRed); l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion', 'k0star_pion', 'lpe'); l5.SetLineColor(ROOT.kGreen); l5.SetMarkerColor(ROOT.kGreen)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(str(Path(output_params.output_dir) / 'psi2s_dataset_kstar_combs.pdf'))

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch': b_mass_branch, 'dataset': dataset_kstar_comb, 'channel_label': fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        Path(output_params.output_dir) / f'fit_{args.mode}_kstar_partial_template_1.pdf',
        file_label=file_label,
        bins=30,
    )

    # Add partial background shape to simplified fit
    if args.verbose:
        print('\nStarting Fit 4 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(Path(output_params.output_dir) / f'fit_{args.mode}_template.yml', 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

    # Build final Roofit model
    model_final = FitModel({'branch': b_mass_branch, 'dataset': dataset_data, 'channel_label': fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb+dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 4500, 0, dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 1792, 0, dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 92, 0, dataset_data.numEntries())

    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of PDFs',
        ROOT.RooArgList(
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf,
        ),
        ROOT.RooArgList(
            sig_coeff,
            comb_bkg_coeff,
            part_bkg_coeff,
        )
    )

    # Add gaussian contraints to fit parameters
    sig_coeff.setConstant(False)
    part_bkg_coeff.setConstant(False)
    comb_bkg_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb1_mean.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_mean.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb1_sigma.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_sigma.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb1_coeff.setConstant(False)
    # model_final.signal_models['sig_pdf'].dcb2_coeff.setConstant(False)

    model_final.add_constraints({
        # 'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(10)),
        # 'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb1_coeff_constraint' : ROOT.RooGaussian('dcb1_coeff_constraint', 'dcb1_coeff_constraint', model_final.signal_models['sig_pdf'].dcb1_coeff, ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']*.05)),
        # 'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb2_coeff_constraint' : ROOT.RooGaussian('dcb2_coeff_constraint', 'dcb2_coeff_constraint', model_final.signal_models['sig_pdf'].dcb2_coeff, ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']*.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, use_minos=True if args.minos else False, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Define the component map specific to this psi2s fit
    component_map = {
        'yield_sig': (model_final.sig_pdf, sig_coeff),
        'yield_comb_bkg': (model_final.comb_bkg_pdf, comb_bkg_coeff),
        'yield_part_bkg': (model_final.part_bkg_pdf, part_bkg_coeff),
    }
    # Add fractional components if they exist
    if mc_yield_tot > 0:
        component_map.update({
            'yield_part_bkg_kstar': (model_final.part_bkg_pdf, part_bkg_coeff, kstar_yield_frac),
            'yield_part_bkg_kstar_pion': (model_final.part_bkg_pdf, part_bkg_coeff, kstar_pion_yield_frac),
            'yield_part_bkg_k0star_kaon': (model_final.part_bkg_pdf, part_bkg_coeff, k0star_kaon_yield_frac),
            'yield_part_bkg_k0star_pion': (model_final.part_bkg_pdf, part_bkg_coeff, k0star_pion_yield_frac),
        })

    # Call the generic calculator from utils.py
    yields = calculate_yields(
        b_mass_branch=b_mass_branch,
        component_map=component_map,
        fit_range=fit_params.fit_range,
        fit_result=model_final.fit_result,
        custom_yield_ranges=custom_yield_ranges
    )

    # Use the results to create plot text and then plot the model
    signal_yield = yields['yield_sig']
    sig_range = (custom_yield_ranges or {}).get('yield_sig')  # Safely check for the custom range
    rounded_yield = [round(y) for y in signal_yield]
    plot_text = f'N_{{#psi(2s)}} = {rounded_yield[0]} #pm {rounded_yield[1]}'
    if sig_range:
        plot_text = f'N_{{#psi(2s)}} [{sig_range[0]}-{sig_range[1]} GeV] = {rounded_yield[0]} #pm {rounded_yield[1]}'

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        Path(output_params.output_dir) / f'fit_{args.mode}_final.pdf',
        file_label=file_label,
        fit_components={
            'Signal':             model_final.sig_pdf,
            # 'Signal Comp 1':      model_final.signal_models['sig_pdf'].dcb1_pdf,
            # 'Signal Comp 2':      model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg.': model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg.':   model_final.part_bkg_pdf,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text=plot_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    # _dcb1_coeff = model_final.signal_models['sig_pdf'].dcb1_coeff.getVal()
    # _dcb2_coeff = model_final.signal_models['sig_pdf'].dcb2_coeff.getVal()
    # _norm_sf = 1 / (_dcb1_coeff + _dcb2_coeff)
    # model_final.signal_models['sig_pdf'].dcb1_coeff.setVal(_dcb1_coeff * _norm_sf)
    # model_final.signal_models['sig_pdf'].dcb2_coeff.setVal(_dcb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])

    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    # Use function to grab yields
    # yields = {
    #     'yield_sig' : (round(sig_coeff.getValV(),2), round(sig_coeff.getError(),2)),
    #     'yield_comb_bkg' : (round(comb_bkg_coeff.getValV(),2), round(comb_bkg_coeff.getError(),2)),
    #     'yield_part_bkg' : (round(part_bkg_coeff.getValV(),2), round(part_bkg_coeff.getError(),2)),
    #     'yield_part_bkg_kstar' : (round(part_bkg_coeff.getValV() * kstar_yield_frac,2), round(part_bkg_coeff.getError(),2)),
    #     'yield_part_bkg_kstar_pion' : (round(part_bkg_coeff.getValV() * kstar_pion_yield_frac,2), round(part_bkg_coeff.getError() * kstar_pion_yield_frac,2)),
    #     'yield_part_bkg_k0star_kaon' : (round(part_bkg_coeff.getValV() * k0star_kaon_yield_frac,2), round(part_bkg_coeff.getError() * k0star_kaon_yield_frac,2)),
    #     'yield_part_bkg_k0star_pion' : (round(part_bkg_coeff.getValV() * k0star_pion_yield_frac,2), round(part_bkg_coeff.getError() * k0star_pion_yield_frac,2)),
    # }

    if get_yields:
        return yields
    else:
        pprint(yields)


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    match args.mode:
        case 'all':
            args.mode = 'lowq2'
            if args.verbose:
                print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
            do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, toy_fit=args.toy_fit)

            args.mode = 'jpsi'
            if args.verbose:
                print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
            do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)

            args.mode = 'psi2s'
            if args.verbose:
                print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
            do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args)

        case 'lowq2':
            if args.constrained_fit:
                raise NotImplementedError('No full constrained fit for lowq2')
            else:
                do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, toy_fit=args.toy_fit)
        case 'jpsi':
            if args.constrained_fit:
                do_constrained_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)
            else:
                do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)
        case 'psi2s':
            if args.constrained_fit:
                raise NotImplementedError('No full constrained fit for psi2s')
            else:
                do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which fit to perform')
    parser.add_argument('-v', '--verbose', nargs='?', const=1, default=0, type=int, help='Set verbosity level. Default is 0. If flag is used without value, sets to 1.')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-t', '--toy_fit', dest='toy_fit', action='store_true', help='fit toy data in low-q2')
    parser.add_argument('-cf', '--constrained_fit', dest='constrained_fit', action='store_true', help='Fit with norm-constrained templates')
    parser.add_argument('-minos', '--minos', dest='minos', action='store_true', help='use MINOS minimizer')
    args = parser.parse_args()

    main(args)
