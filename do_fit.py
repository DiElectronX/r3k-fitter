import ROOT
import yaml
import argparse
import copy
from pprint import pprint
from utils import *
from fit_models import FitModel

ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']
N_SIG_EXP = 0.0018 * 53593

def do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, toy_fit=True, unblinded=False, file_label=None, legend_text=None):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_rare = prepare_inputs(dataset_params, fit_params, isData=False)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_rare, 'channel_label' : fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_rare, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_rare,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_sig_template.pdf'),
            file_label=file_label,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_samesign_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0., unblind=True)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_samesign_data, 'channel_label' : fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_samesign_data, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_samesign_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_comb_template.pdf'),
            bins=30,
            file_label=file_label,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit jpsi leakage in low-q2 region from MC
    if args.verbose:
        print('\nStarting Fit 3 - J/Psi Leakage Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    fit_params.full_mass_range = [4.5,5.7]
    tmp_b_mass_branch, dataset_jpsi = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.jpsi_file)

    # Build Roofit model for exponential background
    model_jpsi_template = FitModel({'branch' : tmp_b_mass_branch, 'dataset' : dataset_jpsi, 'channel_label' : fit_params.channel_label})
    model_jpsi_template.add_background_model('jpsi_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_jpsi_template.fit_model = model_jpsi_template.jpsi_bkg_pdf

    # Fit model to data
    #model_jpsi_template.fit(dataset_jpsi, printlevel=printlevel)
    #params = model_jpsi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsi_template.plot_fit(
        tmp_b_mass_branch,
        dataset_jpsi,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_jpsi_template.pdf'),
        file_label=file_label,
        bins=35,
    )
       
    # Save fit shape parameters
    # template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)
    fit_params.full_mass_range = [4.6,5.7]

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 4 - KStar Partial Template\n{}'.format(50*'~'))
        
    # Import ROOT file dataset
    tmp_b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_pion_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_pion_file, weight_branch_name='trig_wgt_reweighted')
    dataset_kstar_comb = dataset_kstar_pion.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)

    if args.verbose: 
        print('nEvents for K*+ -> piee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for K*0 -> piee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for K*0 -> Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6,.5,.85,.85)
    tmp_frame = tmp_b_mass_branch.frame()
    
    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination','Combination','lpe')
    l1.SetLineColor(ROOT.kBlack)
    l1.SetMarkerColor(ROOT.kBlack)
    l3 = leg.AddEntry('kstar_pion','kstar_pion','lpe')
    l3.SetLineColor(ROOT.kBlue)
    l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon','k0star_kaon + kstar_kaon','lpe')
    l4.SetLineColor(ROOT.kRed)
    l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion','k0star_pion','lpe')
    l5.SetLineColor(ROOT.kGreen)
    l5.SetMarkerColor(ROOT.kGreen)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(os.path.join(output_params.output_dir,'lowq2_dataset_kstar_combs.pdf'))
    tmp_c.Close()

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch' : tmp_b_mass_branch, 'dataset' : dataset_kstar_comb, 'channel_label' : fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        tmp_b_mass_branch,
        dataset_kstar_comb,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_kstar_partial_template.pdf'),
        file_label=file_label,
        bins=30,
    )

    # Add template for final fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))
    
    comb_bkg_norm = 500
    part_bkg_norm = 23
    jpsi_bkg_norm = 223

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Use toys to produce expected signal region
    if toy_fit:
        # Fit background-only model to data sidebands
        bkg_only_model = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
        bkg_only_model.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
        bkg_only_model.add_background_model('jpsi_bkg_pdf', model_jpsi_template.background_models['jpsi_bkg_pdf'])
        bkg_only_model.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])
        comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', comb_bkg_norm, 0., 100000.)
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm,  0, dataset_data.numEntries())
        part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', part_bkg_norm, 0, dataset_data.numEntries())
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
        part_bkg_coeff.setConstant(False)
        bkg_only_model.constraints.update({
            'part_bkg_coeff_constraint' : ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_coeff, ROOT.RooFit.RooConst(part_bkg_coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_coeff.getVal()*.2)),
            # 'kstar_ratio_constraint' : ROOT.RooGaussian('kstar_ratio_constraint', 'kstar_ratio_constraint', kstar_ratio, ROOT.RooFit.RooConst(kstar_ratio.getVal()), ROOT.RooFit.RooConst(.01)),
        })

        bkg_only_model.fit(dataset_data, fit_range='sb1,sb2', fit_norm_range='full', printlevel=printlevel)
        bkg_only_model.fit_model.removeStringAttribute("fitrange")

        # Generate expected background from sideband fit
        print(comb_bkg_coeff.getVal(), jpsi_bkg_coeff.getVal(), part_bkg_coeff.getVal())
        bkg_yield = comb_bkg_coeff.getVal() + jpsi_bkg_coeff.getVal() + part_bkg_coeff.getVal()
        bkg_yield, bkg_yield_err = integrate(b_mass_branch, bkg_only_model.fit_model, [comb_bkg_coeff,jpsi_bkg_coeff,part_bkg_coeff], fit_params.full_mass_range, bkg_only_model.fit_result)
        dataset_data.Print()
        toy_background = bkg_only_model.fit_model.generate(ROOT.RooArgSet(b_mass_branch), bkg_yield)
        
        # Generate expected signal from MC shape and jpsi-extrapolated yield
        if args.cache:
            _, dataset_rare = prepare_inputs(dataset_params, fit_params, isData=False)
            model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_rare, 'channel_label' : fit_params.channel_label})
            model_sig_template.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
            model_sig_template.fit_model = model_sig_template.sig_pdf

        toy_signal = model_sig_template.fit_model.generate(ROOT.RooArgSet(b_mass_branch), N_SIG_EXP)

        # Create toy dataset for final fit
        toy_dataset = dataset_data.emptyClone('dataset_data'+fit_params.channel_label)
        toy_dataset.append(toy_background)
        toy_dataset.append(toy_signal)

        print(dataset_data.sumEntries())
        print(toy_background.sumEntries())
        print(toy_dataset.sumEntries())

        # Temporary plot of just toy dataset 
        tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
        tmp_frame = b_mass_branch.frame()
        toy_background.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
        toy_signal.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.RefreshNorm(),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.Range('sb1,sb2'), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.Range('sb1,sb2'), ROOT.RooFit.NormRange('full'), ROOT.RooFit.LineColor(ROOT.kYellow), ROOT.RooFit.MarkerColor(ROOT.kYellow))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.Range('full'), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.MarkerColor(ROOT.kOrange))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.Range('full'), ROOT.RooFit.NormRange('full'), ROOT.RooFit.LineColor(ROOT.kGray), ROOT.RooFit.MarkerColor(ROOT.kGray))
        dataset_data.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.MarkerColor(ROOT.kCyan))
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.NormRange('sb1,sb2'),ROOT.RooFit.LineStyle(ROOT.kSolid),ROOT.RooFit.LineColor(ROOT.kMagenta))
        # bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Components(ROOT.RooArgSet(bkg_only_model.part_bkg_pdf)), ROOT.RooFit.Binning(50), ROOT.RooFit.Range('sb1,sb2'), ROOT.RooFit.NormRange('sb1,sb2'))
        # bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Components(ROOT.RooArgSet(bkg_only_model.jpsi_bkg_pdf)), ROOT.RooFit.Binning(50), ROOT.RooFit.Range('sb1,sb2'), ROOT.RooFit.NormRange('sb1,sb2'))
        tmp_frame.Draw()
        tmp_c.SaveAs(os.path.join(output_params.output_dir,'tmp_toy_datasets.pdf'))
        tmp_c.Close()

        dataset_data = toy_dataset
    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})

    if toy_fit:
        model_final.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    # model_final.add_background_model('jpsi_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('jpsi_bkg_pdf', model_jpsi_template.background_models['jpsi_bkg_pdf'])
    model_final.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])

    if toy_fit:
        sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 101., 0., 5*dataset_data.numEntries())
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)
    else:
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)
    
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 500, 0., 5*dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 23, 0, dataset_data.numEntries())

    model_comps = [model_final.comb_bkg_pdf, model_final.jpsi_bkg_pdf, model_final.part_bkg_pdf]
    model_coeffs = [comb_bkg_coeff, jpsi_bkg_coeff, part_bkg_coeff]
    if toy_fit:
        model_comps.append(model_final.sig_pdf)
        model_coeffs.append(sig_coeff)
        # model_comps.insert(0,model_final.sig_pdf)
        # model_coeffs.insert(0,sig_coeff)

    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(*model_comps),
        ROOT.RooArgList(*model_coeffs)
    )

    # Add gaussian contraints to fit parameters
    part_bkg_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)

    if toy_fit:
        model_final.signal_models['sig_pdf'].dcb_mean.setConstant(False)
        model_final.signal_models['sig_pdf'].dcb_sigma.setConstant(False)
        #jpsi_ratio = ROOT.RooFormulaVar('jpsi_ratio', 'Ratio of JPsi leakage', '@0/@1', ROOT.RooArgList(sig_coeff, jpsi_bkg_coeff))
    else:
        pass
        # jpsi_ratio = ROOT.RooFormulaVar('jpsi_ratio', 'Ratio of JPsi leakage', '@0/@1', ROOT.RooArgList(part_bkg_coeff, jpsi_bkg_coeff))


    # Add gaussian contraints to fit parameters
    model_final.constraints.update({
        'part_bkg_coeff_constraint' : ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_coeff, ROOT.RooFit.RooConst(part_bkg_coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_coeff.getVal()*.2)),
        #'exp_slope_comb_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_comb_bkg_pdf_constraint', 'exp_slope_comb_bkg_pdf_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(5)),
        # 'exp_slope_jpsi_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_jpsi_bkg_pdf_constraint', 'exp_slope_jpsi_bkg_pdf_constraint', model_final.background_models['jpsi_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_jpsi_bkg_pdf']), ROOT.RooFit.RooConst(0.5)),
    })
    if toy_fit:
        model_final.constraints.update({
            'dcb_mean_constraint' : ROOT.RooGaussian('dcb_mean_constraint', 'dcb_mean_constraint', model_final.signal_models['sig_pdf'].dcb_mean, ROOT.RooFit.RooConst(template['dcb_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
            'dcb_sigma_constraint' : ROOT.RooGaussian('dcb_sigma_constraint', 'dcb_sigma_constraint', model_final.signal_models['sig_pdf'].dcb_sigma, ROOT.RooFit.RooConst(template['dcb_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
            # 'jpsi_ratio_constraint' : ROOT.RooGaussian('jpsi_ratio_constraint', 'jpsi_ratio_constraint', jpsi_ratio, ROOT.RooFit.RooConst(jpsi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        })

    # Fit model to data
    fit_range = 'full' #if toy_fit else 'sb1,sb2'
    fit_norm_range = 'full' #if toy_fit else 'sb1,sb2'
    model_final.fit(dataset_data, fit_range=fit_range, fit_norm_range=fit_norm_range, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()
    
    if toy_fit:
        yield_text = 'N_{{B #rightarrow eeK}} = {} #pm {}'.format(round(sig_coeff.getValV()), round(sig_coeff.getError(),1)) 
    else:
        bkg_yield, bkg_yield_err = integrate(b_mass_branch, model_final.fit_model, model_coeffs, [5.1,5.4], model_final.fit_result)
        yield_text = 'N_{{Bkg}} [5.1-5.4 GeV] = {} #pm {}'.format(round(bkg_yield), round(bkg_yield_err,1))

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+('_toy' if toy_fit else '')+'_final.pdf'),
        file_label=file_label,
        fit_components=model_comps,
        fit_range=fit_range,
        fit_norm_range=fit_norm_range,
        fit_result=model_final.fit_result,
        bins=35,
        extra_text=yield_text,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    jpsi_bkg_pdf_norm = ROOT.RooRealVar('jpsi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of jpsi low-q2 background events', jpsi_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_1_norm = ROOT.RooRealVar('part_bkg_pdf_1'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Write final fit to RooWorkspace
    if write:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, jpsi_bkg_pdf_norm])

    yields = {
        'yield_sig' : round(sig_coeff.getValV()) if toy_fit else 'N/A',
        'yield_sig_err' : round(sig_coeff.getError(),2) if toy_fit else 'N/A',
        'yield_comb_bkg' : round(comb_bkg_coeff.getValV()),
        'yield_comb_bkg_err' : round(comb_bkg_coeff.getError(),2),
        'yield_jpsi_bkg' : round(jpsi_bkg_coeff.getValV()),
        'yield_jpsi_bkg_err' : round(jpsi_bkg_coeff.getError(),2),
        'yield_part_bkg' : round(part_bkg_coeff.getValV()),
        'yield_part_bkg_err' : round(part_bkg_coeff.getError(),2),
    }
    if get_yields:
        return yields
    else:
        pprint(yields)


def do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, printlevel=printlevel)
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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_comb_template.pdf'),
            file_label=file_label,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))
    
    # Look at partial shape files
    tmp_b_mass_branch, dataset_kstar_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_kstar_pion   = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_jpsi_pion_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_kaon  = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_k0star_pion  = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_jpsi_pion_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_chic1_kaon   = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.chic1_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
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
    if args.verbose:
        print('\nStarting Fit 4 - JpsiPi Partial Template \n{}'.format(50*'~'))
    
    _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='trig_wgt_reweighted')
    model_jpsipi_pion_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_jpsipi_pion, 'channel_label' : fit_params.channel_label})
    model_jpsipi_pion_template.add_background_model('part_bkg_pdf_jpsipi_pion', 'dcb', fit_params.fit_defaults, let_float=True)
    model_jpsipi_pion_template.fit_model = model_jpsipi_pion_template.part_bkg_pdf_jpsipi_pion
    
    # Fit model to data
    model_jpsipi_pion_template.fit(dataset_jpsipi_pion, printlevel=printlevel)
    params = model_jpsipi_pion_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsipi_pion_template.plot_fit(
        b_mass_branch,
        dataset_jpsipi_pion,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_jpsipi_template.pdf'),
        file_label=file_label,
    )

    # Save fit shape parameters
    template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

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
    #part_bkg_jpsipi_pion_coeff.setConstant(False)
    #jpsipi_ratio = ROOT.RooFormulaVar('jpsipi_ratio', 'Ratio of B->JpsiPi decay channel', '@0/@1', ROOT.RooArgList(sig_coeff, part_bkg_jpsipi_pion_coeff))

    model_final.add_constraints({
        #'jpsipi_ratio_constraint' : ROOT.RooGaussian('jpsipi_ratio_constraint', 'jpsipi_ratio_constraint', jpsipi_ratio, ROOT.RooFit.RooConst(jpsipi_ratio.getVal()), ROOT.RooFit.RooConst(.05)),
        'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(2.)),
        'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        #'dcb1_coeff_constraint' : ROOT.RooGaussian('dcb1_coeff_constraint', 'dcb1_coeff_constraint', model_final.signal_models['sig_pdf'].dcb1_coeff, ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']*.05)),
        'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        #'dcb2_coeff_constraint' : ROOT.RooGaussian('dcb2_coeff_constraint', 'dcb2_coeff_constraint', model_final.signal_models['sig_pdf'].dcb2_coeff, ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']*.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        file_label=file_label,
        fit_components = {
            'Signal PDF' : model_final.sig_pdf,
            # model_final.signal_models['sig_pdf'].dcb1_pdf,
            # model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg. PDF' : model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg. PDF' : model_final.part_bkg_pdf,
            'B #rightarrow J/#psi #pi Bkg. PDF' : model_final.part_bkg_pdf_jpsipi_pion,
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
        return yields
    else:
        pprint(yields)


def do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)

        # Build Roofit model for signal
        model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc, 'channel_label' : fit_params.channel_label})
        model_sig_template.add_signal_model('sig_pdf', 'dcb+dcb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, printlevel=printlevel)
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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))
        '''
        fit_params_loose = copy.deepcopy(fit_params)
        fit_params_loose.full_mass_range = [4.5,5.7]
        _, dataset_samesign_loose = prepare_inputs(dataset_params, fit_params_loose, isData=True, set_file=dataset_params.samesign_data_file, score_cut=-5.)
        tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
        tmp_frame = b_mass_branch.frame()
        dataset_samesign_loose.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
        tmp_frame.Draw()
        tmp_c.SaveAs(os.path.join(output_params.output_dir,'tmp_samesigndata_loose.pdf'))
        '''
        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_data, printlevel=printlevel)
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_comb_template.pdf'),
            file_label=file_label,
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - KStar Partial Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    tmp_b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.kstar_psi2s_pion_file)
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_kaon_file)#, extra_weight=.1)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.k0star_psi2s_pion_file)#, extra_weight=.1)
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
    leg = ROOT.TLegend(.6,.5,.85,.85)
    tmp_frame = tmp_b_mass_branch.frame()
    
    dataset_kstar_pion.plotOn(tmp_frame, ROOT.RooFit.Name('kstar_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
    dataset_k0star_kaon.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_kaon'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
    dataset_k0star_pion.plotOn(tmp_frame, ROOT.RooFit.Name('k0star_pion'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.MarkerColor(ROOT.kGreen))
    dataset_kstar_comb.plotOn(tmp_frame,  ROOT.RooFit.Name('combination'),ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))

    l1 = leg.AddEntry('combination','Combination','lpe')
    l1.SetLineColor(ROOT.kBlack)
    l1.SetMarkerColor(ROOT.kBlack)
    l3 = leg.AddEntry('kstar_pion','kstar_pion','lpe')
    l3.SetLineColor(ROOT.kBlue)
    l3.SetMarkerColor(ROOT.kBlue)
    l4 = leg.AddEntry('k0star_kaon','k0star_kaon + kstar_kaon','lpe')
    l4.SetLineColor(ROOT.kRed)
    l4.SetMarkerColor(ROOT.kRed)
    l5 = leg.AddEntry('k0star_pion','k0star_pion','lpe')
    l5.SetLineColor(ROOT.kGreen)
    l5.SetMarkerColor(ROOT.kGreen)

    tmp_frame.Draw()
    leg.Draw()
    tmp_c.SaveAs(os.path.join(output_params.output_dir,'psi2s_dataset_kstar_combs.pdf'))

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar_comb, 'channel_label' : fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_kstar_partial_template_1.pdf'),
        file_label=file_label,
        bins=30,
    )
    
    # Add partial background shape to simplified fit
    if args.verbose:
        print('\nStarting Fit 4 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
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
    part_bkg_coeff.setConstant(False)
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_mean.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_sigma.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb1_coeff.setConstant(False)
    model_final.signal_models['sig_pdf'].dcb2_coeff.setConstant(False)

    model_final.add_constraints({
        'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(10)),
        'dcb1_mean_constraint' : ROOT.RooGaussian('dcb1_mean_constraint', 'dcb1_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb1_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb1_sigma_constraint' : ROOT.RooGaussian('dcb1_sigma_constraint', 'dcb1_sigma_constraint', model_final.signal_models['sig_pdf'].dcb1_sigma, ROOT.RooFit.RooConst(template['dcb1_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb1_coeff_constraint' : ROOT.RooGaussian('dcb1_coeff_constraint', 'dcb1_coeff_constraint', model_final.signal_models['sig_pdf'].dcb1_coeff, ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb1_coeff_sig_pdf']*.05)),
        'dcb2_mean_constraint' : ROOT.RooGaussian('dcb2_mean_constraint', 'dcb2_mean_constraint', model_final.signal_models['sig_pdf'].dcb1_mean, ROOT.RooFit.RooConst(template['dcb2_mean_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        'dcb2_sigma_constraint' : ROOT.RooGaussian('dcb2_sigma_constraint', 'dcb2_sigma_constraint', model_final.signal_models['sig_pdf'].dcb2_sigma, ROOT.RooFit.RooConst(template['dcb2_sigma_sig_pdf']), ROOT.RooFit.RooConst(.01)),
        # 'dcb2_coeff_constraint' : ROOT.RooGaussian('dcb2_coeff_constraint', 'dcb2_coeff_constraint', model_final.signal_models['sig_pdf'].dcb2_coeff, ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']), ROOT.RooFit.RooConst(template['dcb2_coeff_sig_pdf']*.05)),
    })

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        file_label=file_label,
        fit_components = {
            'Signal PDF' : model_final.sig_pdf,
            # model_final.signal_models['sig_pdf'].dcb1_pdf,
            # model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg. PDF' : model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg. PDF' : model_final.part_bkg_pdf,
        },
        fit_result=model_final.fit_result,
        legend=True,
        extra_text='N_{{#psi(2s)}} = {} #pm {}'.format(round(sig_coeff.getValV()), round(sig_coeff.getError(),1)),
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    _dcb1_coeff = model_final.signal_models['sig_pdf'].dcb1_coeff.getVal()
    _dcb2_coeff = model_final.signal_models['sig_pdf'].dcb2_coeff.getVal()
    _norm_sf = 1 / (_dcb1_coeff + _dcb2_coeff)
    model_final.signal_models['sig_pdf'].dcb1_coeff.setVal(_dcb1_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].dcb2_coeff.setVal(_dcb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])

    # Use function to grab yields
    yields = {
        'yield_sig' : round(sig_coeff.getValV(),2),
        'yield_sig_err' : round(sig_coeff.getError(),2),
        'yield_comb_bkg' : round(comb_bkg_coeff.getValV(),2),
        'yield_comb_bkg_err' : round(comb_bkg_coeff.getError(),2),
        'yield_part_bkg' : round(part_bkg_coeff.getValV(),2),
        'yield_part_bkg_err' : round(part_bkg_coeff.getError(),2),
        'yield_part_bkg_kstar' : round(part_bkg_coeff.getValV() * kstar_yield_frac,2),
        'yield_part_bkg_kstar_err' : round(part_bkg_coeff.getError(),2),
        'yield_part_bkg_kstar_pion' : round(part_bkg_coeff.getValV() * kstar_pion_yield_frac,2),
        'yield_part_bkg_kstar_pion_err' : round(part_bkg_coeff.getError() * kstar_pion_yield_frac,2),
        'yield_part_bkg_k0star_kaon' : round(part_bkg_coeff.getValV() * k0star_kaon_yield_frac,2),
        'yield_part_bkg_k0star_kaon_err' : round(part_bkg_coeff.getError() * k0star_kaon_yield_frac,2),
        'yield_part_bkg_k0star_pion' : round(part_bkg_coeff.getValV() * k0star_pion_yield_frac,2),
        'yield_part_bkg_k0star_pion_err' : round(part_bkg_coeff.getError() * k0star_pion_yield_frac,2),
    }

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

    if args.mode=='all':
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

    elif args.mode=='lowq2':
        do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, toy_fit=args.toy_fit)

    elif args.mode=='jpsi':
        do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)

    elif args.mode=='psi2s':
        do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which fit to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    parser.add_argument('-t', '--toy_fit', dest='toy_fit', action='store_true', help='fit toy data in low-q2')
    args = parser.parse_args()

    main(args)
