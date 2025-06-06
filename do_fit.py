import ROOT
import yaml
import argparse
import copy
from pprint import pprint
from utils import *
from fit_models import FitModel

ALLOWED_MODES = ['jpsi', 'psi2s', 'lowq2']

def do_lowq2_signal_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, toy_fit=True, unblinded=False, file_label=None, legend_text=None, param_file_lock=False):
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
        _, dataset_rare = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name='final_wgt')

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
            fit_result=model_sig_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_samesign_data = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch,isData=True, set_file=dataset_params.samesign_data_file, score_cut=0., unblind=True)

        # Build Roofit model for exponential background
        model_comb_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_samesign_data, 'channel_label' : fit_params.channel_label})
        model_comb_template.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_comb_template.fit_model = model_comb_template.comb_bkg_pdf

        # Fit model to data
        model_comb_template.fit(dataset_samesign_data, printlevel=printlevel, fit_range='semilow', fit_norm_range='semilow')
        params = model_comb_template.fit_result.floatParsFinal()

        # Plot fit result
        model_comb_template.plot_fit(
            b_mass_branch,
            dataset_samesign_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_comb_template.pdf'),
            bins=30,
            file_label=file_label,
            fit_range='semilow',
            fit_result=model_comb_template.fit_result,
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)
    
    # Fit jpsi leakage in low-q2 region from MC
    if args.verbose:
        print('\nStarting Fit 3 - J/Psi Leakage Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    _, dataset_jpsi = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.jpsi_file, weight_branch_name='final_wgt')

    # Build Roofit model for exponential background
    model_jpsi_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_jpsi, 'channel_label' : fit_params.channel_label})
    model_jpsi_template.add_background_model('jpsi_bkg_pdf', 'gauss', fit_params.fit_defaults, let_float=True)
    model_jpsi_template.fit_model = model_jpsi_template.jpsi_bkg_pdf

    # Fit model to data
    model_jpsi_template.fit(dataset_jpsi, fit_range='low', fit_norm_range='low', printlevel=printlevel)
    params = model_jpsi_template.fit_result.floatParsFinal()

    # Plot fit result
    model_jpsi_template.plot_fit(
        b_mass_branch,
        dataset_jpsi,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_jpsi_template.pdf'),
        file_label=file_label,
        fit_range='low',
        fit_result=model_jpsi_template.fit_result,
        bins=35,
    )
       
    # Save fit shape parameters
    template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 4 - KStar Partial Template\n{}'.format(50*'~'))
        
    # Import ROOT file dataset
    _, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, set_file=dataset_params.kstar_pion_file, weight_branch_name='final_wgt')
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch,isData=False, set_file=dataset_params.k0star_kaon_file, weight_branch_name='final_wgt')
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch,isData=False, set_file=dataset_params.k0star_pion_file, weight_branch_name='final_wgt')
    dataset_kstar_comb = dataset_kstar_pion.Clone('dataset_kstar_comb'+fit_params.channel_label)
    dataset_kstar_comb.append(dataset_k0star_kaon)
    dataset_kstar_comb.append(dataset_k0star_pion)

    if args.verbose: 
        print('nEvents for K*+ -> piee cand = {}'.format(dataset_kstar_pion.sumEntries()))
        print('nEvents for K*0 -> piee cand = {}'.format(dataset_k0star_pion.sumEntries()))
        print('nEvents for K*0 -> Kee cand = {}'.format(dataset_k0star_kaon.sumEntries()))

    tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
    leg = ROOT.TLegend(.6,.5,.85,.85)
    tmp_frame = b_mass_branch.frame()
    
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
    tmp_c.SaveAs(os.path.join(output_params.output_dir,'fit_'+args.mode+'_kstar_combination_dataset.pdf'))
    tmp_c.Close()

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar_comb, 'channel_label' : fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        b_mass_branch,
        dataset_kstar_comb,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_kstar_partial_template.pdf'),
        file_label=file_label,
        bins=30,
    )

    # Add template for final fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))
    
    comb_bkg_norm = 170
    part_bkg_norm = 23
    jpsi_bkg_norm = 200

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    _, dataset_data = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=True)

    # Use toys to produce expected signal 
    if toy_fit:
        # Fit background-only model to data sidebands
        bkg_only_model = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
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
        jpsi_bkg_coeff.setConstant(False)
        part_bkg_coeff.setConstant(False)
        bkg_only_model.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)

        bkg_only_model.constraints.update({
            'part_bkg_coeff_constraint' : ROOT.RooGaussian('part_bkg_coeff_constraint', 'part_bkg_coeff_constraint', part_bkg_coeff, ROOT.RooFit.RooConst(part_bkg_coeff.getVal()), ROOT.RooFit.RooConst(part_bkg_coeff.getVal()*.2)),
            # 'exp_slope_comb_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_comb_bkg_pdf_constraint', 'exp_slope_comb_bkg_pdf_constraint', bkg_only_model.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(5)),
        })

        bkg_only_model.fit(dataset_data, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)
        params = bkg_only_model.fit_result.floatParsFinal()
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)
        
        bkg_only_model.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_bkg_only.pdf'),
            file_label=file_label,
            fit_components = {
                'Combinatorial Bkg.' : bkg_only_model.comb_bkg_pdf,
                'Part.-Reco. Bkg.' : bkg_only_model.part_bkg_pdf,
                'B #rightarrow J/#psi K Bkg.' : bkg_only_model.jpsi_bkg_pdf,
            },
            fit_range='full',
            fit_norm_range='sb1,sb2',
            fit_result=bkg_only_model.fit_result,
            bins=35,
            legend=True,
            yrange=[0,100],
        )

        # Generate expected background from sideband fit
        expected_bkg, _ = integrate(
            b_mass_branch, 
            bkg_only_model.fit_model, 
            [4.5, 5.7], 
            coeffs=[comb_bkg_coeff,jpsi_bkg_coeff,part_bkg_coeff], 
        )
        toy_background = bkg_only_model.fit_model.generate(ROOT.RooArgSet(b_mass_branch), expected_bkg)

        # Generate expected signal from MC shape and jpsi-extrapolated yield
        if args.cache:
            _, dataset_rare = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name='final_wgt')
            model_sig_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_rare, 'channel_label' : fit_params.channel_label})
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
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'),ROOT.RooFit.LineStyle(ROOT.kSolid),ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name('f_comb'))
        comp = ROOT.RooArgSet(bkg_only_model.jpsi_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid),ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name('f_jpsi'))
        comp = ROOT.RooArgSet(bkg_only_model.part_bkg_pdf)
        bkg_only_model.fit_model.plotOn(tmp_frame, ROOT.RooFit.Range('full'), ROOT.RooFit.Components(comp), ROOT.RooFit.NormRange('sb1,sb2'), ROOT.RooFit.LineStyle(ROOT.kSolid),ROOT.RooFit.LineColor(ROOT.kViolet), ROOT.RooFit.Name('f_part'))

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
        tmp_frame.GetYaxis().SetRangeUser(0,80)
        legend.Draw()
        tmp_c.SaveAs(os.path.join(output_params.output_dir,'fit_'+args.mode+'_toy_dataset.pdf'))
        tmp_c.Close()

        dataset_data = toy_dataset

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})

    if toy_fit:
        model_final.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('jpsi_bkg_pdf', 'gauss', template, let_float=False)
    # model_final.add_background_model('jpsi_bkg_pdf', model_jpsi_template.background_models['jpsi_bkg_pdf'])
    model_final.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])

    if toy_fit:
        sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 101., 0., 5*dataset_data.numEntries())
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)
    else:
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', jpsi_bkg_norm)
    
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 500, 0., 5*dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 23, 0, dataset_data.numEntries())

    model_comps = {
        'Combinatorial Bkg.' : model_final.comb_bkg_pdf, 
        'B #rightarrow J/#psi K Leakage' : model_final.jpsi_bkg_pdf, 
        'Part.-Reco. Bkg.' : model_final.part_bkg_pdf,
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
    template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    if toy_fit:
        sig_yield, sig_yield_err = integrate(b_mass_branch, model_final.sig_pdf, fit_params.fit_range, coeffs=sig_coeff)
        yield_text = f'N_{{B #rightarrow eeK}} = {round(sig_yield)} #pm {round(sig_yield_err,2)}' 
    else:
        bkg_yield, bkg_yield_err = integrate(b_mass_branch, model_final.fit_model, [5.1,5.4], coeffs=model_coeffs)
        yield_text = f'N_{{Bkg}} [5.1-5.4 GeV] = {round(bkg_yield)} #pm {round(bkg_yield_err,2)}'

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
        legend=True,
        yrange=[0,80],
        extra_text=yield_text,
    )

    # Calculate Yields in fit window
    if toy_fit:
        yield_sig, yield_sig_err = integrate(
            b_mass_branch, 
            model_final.sig_pdf, 
            fit_params.fit_range, 
            coeffs=sig_coeff, 
        )
    yield_comb_bkg, yield_comb_bkg_err = integrate(
        b_mass_branch, 
        model_final.comb_bkg_pdf, 
        fit_params.fit_range, 
        coeffs=comb_bkg_coeff, 
    )
    yield_part_bkg, yield_part_bkg_err = integrate(
        b_mass_branch, 
        model_final.part_bkg_pdf, 
        fit_params.fit_range, 
        coeffs=part_bkg_coeff, 
    )
    yield_jpsi_bkg, yield_jpsi_bkg_err = integrate(
        b_mass_branch, 
        model_final.jpsi_bkg_pdf, 
        fit_params.fit_range, 
        coeffs=jpsi_bkg_coeff, 
    )

    # Add normalization terms for Combine
    if toy_fit:
        sig_pdf_norm = ROOT.RooRealVar('sig_pdf'+fit_params.channel_label+'_norm', 'Number of signal events', yield_sig, 0, 999999)
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', yield_comb_bkg, 0, 999999)
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', yield_part_bkg, 0, 999999)
    jpsi_bkg_pdf_norm = ROOT.RooRealVar('jpsi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of jpsi low-q2 background events', yield_jpsi_bkg, 0, 999999)

    # Write final fit to RooWorkspace
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm, jpsi_bkg_pdf_norm]
        if toy_fit:
            extra_objects.append(sig_pdf_norm)
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

    yields = {
        'yield_sig'      : (round(yield_sig,2) if toy_fit else 'N/A', round(yield_sig_err,2) if toy_fit else 'N/A'),
        'yield_comb_bkg' : (round(yield_comb_bkg, 2), round(yield_comb_bkg_err, 2)),
        'yield_part_bkg' : (round(yield_part_bkg, 2), round(yield_part_bkg_err, 2)),
        'yield_jpsi_bkg' : (round(yield_jpsi_bkg, 2), round(yield_jpsi_bkg_err, 2)),
    }
    if get_yields:
        return yields
    else:
        pprint(yields)


def do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False):
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
        _, dataset_mc = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name='sf_combined_mean')

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
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

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
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - Partial Template \n{}'.format(50*'~'))
    
    # Look at partial shape files
    tmp_b_mass_branch, dataset_kstar_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_kaon_file, weight_branch_name='final_wgt')
    _, dataset_kstar_pion   = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_jpsi_pion_file, weight_branch_name='final_wgt')
    _, dataset_k0star_kaon  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_kaon_file, weight_branch_name='final_wgt')
    _, dataset_k0star_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_jpsi_pion_file, weight_branch_name='final_wgt')
    _, dataset_chic1_kaon   = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.chic1_jpsi_kaon_file, weight_branch_name='final_wgt')
    _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='final_wgt')
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
    
    _, dataset_jpsipi_pion  = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.jpsipi_jpsi_kaon_file, weight_branch_name='final_wgt')
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
        fit_result=model_jpsipi_pion_template.fit_result,
    )

    # Save fit shape parameters
    template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Final Composite Fit
    if args.verbose:
        print('\nStarting Fit 5 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
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
        return yields
    else:
        pprint(yields)


def do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, file_label=None, legend_text=None, param_file_lock=False):
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
        _, dataset_mc = prepare_inputs(dataset_params, fit_params, b_mass_branch=b_mass_branch, isData=False, weight_branch_name='sf_combined_mean')

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
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, lock_file=param_file_lock)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))
        
        # Import ROOT file dataset
        _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch, set_file=dataset_params.samesign_data_file, score_cut=0.)

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
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args, update_dict=template, lock_file=param_file_lock)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - KStar Partial Template\n{}'.format(50*'~'))

    # Import ROOT file dataset
    tmp_b_mass_branch, dataset_kstar_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.kstar_psi2s_pion_file, weight_branch_name='final_wgt')
    _, dataset_k0star_kaon = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_psi2s_kaon_file, weight_branch_name='final_wgt')#, extra_weight=.1)
    _, dataset_k0star_pion = prepare_inputs(dataset_params, fit_params, isData=False, b_mass_branch=b_mass_branch, set_file=dataset_params.k0star_psi2s_pion_file, weight_branch_name='final_wgt')#, extra_weight=.1)
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
    _, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, b_mass_branch=b_mass_branch)

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
            'Signal' : model_final.sig_pdf,
            # model_final.signal_models['sig_pdf'].dcb1_pdf,
            # model_final.signal_models['sig_pdf'].dcb2_pdf,
            'Combinatorial Bkg.' : model_final.comb_bkg_pdf,
            'Part.-Reco. Bkg.' : model_final.part_bkg_pdf,
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
    
    if write:
        extra_objects = [comb_bkg_pdf_norm, part_bkg_pdf_norm]
        write_workspace(output_params, args, model_final, extra_objs=extra_objects)

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
