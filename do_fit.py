import ROOT
import yaml
import argparse
from utils import *
from fit_models import FitModel

ALLOWED_MODES = ['jpsi', 'psi2s', 'rare']

def do_signal_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False, toy_fit=True):
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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_samesign_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0., unblind=True)

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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit jpsi leakage in low-q2 region from MC
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 3 - J/Psi Leakage Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_jpsi = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.jpsi_file)

        # Build Roofit model for exponential background
        model_jpsi_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_jpsi, 'channel_label' : fit_params.channel_label})
        model_jpsi_template.add_background_model('jpsi_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_jpsi_template.fit_model = model_jpsi_template.jpsi_bkg_pdf

        # Fit model to data
        model_jpsi_template.fit(dataset_jpsi, printlevel=printlevel)
        params = model_jpsi_template.fit_result.floatParsFinal()

        # Plot fit result
        model_jpsi_template.plot_fit(
            b_mass_branch,
            dataset_jpsi,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_jpsi_template.pdf'),
            bins=30,
        )
           
        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Add template for final fit
    if args.verbose:
        print('\nStarting Fit 4 - Final Model\n{}'.format(50*'~'))

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
        comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 0.15, 0., 100000.)
        jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', 540., 0., 100000.)
        bkg_only_model.fit_model = ROOT.RooAddPdf(
            'bkg_only_pdf',
            'Sum of Background PDFs',
            ROOT.RooArgList(
                bkg_only_model.comb_bkg_pdf,
                bkg_only_model.jpsi_bkg_pdf,
            ),
            ROOT.RooArgList(
                comb_bkg_coeff,
                jpsi_bkg_coeff,
            )
        )
        bkg_only_model.fit(dataset_data, fit_range='sb1,sb2', fit_norm_range='sb1,sb2', printlevel=printlevel)

        # Generate expected background from sideband fit
        bkg_yield = comb_bkg_coeff.getVal() + jpsi_bkg_coeff.getVal()
        toy_background = bkg_only_model.fit_model.generate(ROOT.RooArgSet(b_mass_branch), bkg_yield)
        
        # Generate expected signal from MC shape and jpsi-extrapolated yield
        N_sig_exp = 0.0018 * 66700.91
        toy_signal = model_sig_template.fit_model.generate(ROOT.RooArgSet(b_mass_branch), N_sig_exp)

        # Create toy dataset for final fit
        toy_dataset = dataset_data.emptyClone('dataset_data'+fit_params.channel_label)
        toy_dataset.append(toy_background)
        toy_dataset.append(toy_signal)
        dataset_data = toy_dataset

        # Temporary plot of just toy dataset 
        tmp_c = ROOT.TCanvas('tmp_c', ' ', 800, 600)
        tmp_frame = b_mass_branch.frame()
        toy_background.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.MarkerColor(ROOT.kBlue))
        toy_signal.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.MarkerColor(ROOT.kRed))
        toy_dataset.plotOn(tmp_frame, ROOT.RooFit.Binning(30), ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.MarkerColor(ROOT.kBlack))
        tmp_frame.Draw()
        tmp_c.SaveAs(os.path.join(output_params.output_dir,'tmp_toy_datasets.pdf'))

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'dcb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('jpsi_bkg_pdf', 'exp', template, let_float=False)

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 100., 0., 5*dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 0.15, 0., 5*dataset_data.numEntries())
    jpsi_bkg_coeff = ROOT.RooRealVar('jpsi_bkg_coeff'+fit_params.channel_label, 'J/Psi Leakage Background Coefficient', 540., 0., 5*dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.jpsi_bkg_pdf,
        ),
        ROOT.RooArgList(
            sig_coeff,
            comb_bkg_coeff,
            jpsi_bkg_coeff,
        )
    )

    # Add gaussian contraints to fit parameters
    model_final.constraints.update({
        'exp_slope_comb_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_comb_bkg_pdf_constraint', 'exp_slope_comb_bkg_pdf_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_comb_bkg_pdf']), ROOT.RooFit.RooConst(0.5)),
        'exp_slope_jpsi_bkg_pdf_constraint' : ROOT.RooGaussian('exp_slope_jpsi_bkg_pdf_constraint', 'exp_slope_jpsi_bkg_pdf_constraint', model_final.background_models['jpsi_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope_jpsi_bkg_pdf']), ROOT.RooFit.RooConst(0.5)),
        # 'jpsi_bkg_coeff_jpsi_bkg_pdf_constraint' : ROOT.RooGaussian('jpsi_bkg_coeff_jpsi_bkg_pdf_constraint', 'jpsi_bkg_coeff_jpsi_bkg_pdf_constraint', jpsi_bkg_coeff, ROOT.RooFit.RooConst(template['exp_slope_jpsi_bkg_pdf']), ROOT.RooFit.RooConst(0.5)),
    })

    # Fit model to data
    fit_range = 'sb1,sb2' if fit_params.blinded else 'full'
    fit_norm_range = 'sb1,sb2' if fit_params.blinded else 'full'
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.jpsi_bkg_pdf,
        ],
        bins=30,
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    jpsi_bkg_pdf_norm = ROOT.RooRealVar('jpsi_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of jpsi low-q2 background events', jpsi_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Write final fit to RooWorkspace
    if write:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, jpsi_bkg_pdf_norm])

    # Use function to grab yields
    # if get_yields:
    if True:
        yields = {
            'yield_sig' : sig_coeff.getValV(),
            'yield_comb_bkg' : comb_bkg_coeff.getValV(),
            'yield_jpsi_bkg' : jpsi_bkg_coeff.getValV(),
        }
        print(yields)
        return yields


def do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False):
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
        model_sig_template.add_signal_model('sig_pdf', 'cb+cb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_sig_template.pdf'),
            fit_components = [
                model_sig_template.signal_models['sig_pdf'].cb1_pdf,
                model_sig_template.signal_models['sig_pdf'].cb2_pdf,
            ],
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0.)

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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit partial background shape to kstar MC
    if args.verbose:
        print('\nStarting Fit 3 - KStar Partial Template 1\n{}'.format(50*'~'))

    # Import ROOT file dataset
    b_mass_branch, dataset_kstar = prepare_inputs(dataset_params, fit_params, isData=False, set_file=dataset_params.bu_kstar_jpsi_pion_file, set_tree='Events')

    # Build Roofit model for exponential background
    model_kstar_template = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_kstar, 'channel_label' : fit_params.channel_label})
    model_kstar_template.add_background_model('part_bkg_pdf', 'kde', fit_params.fit_defaults, let_float=True)
    model_kstar_template.fit_model = model_kstar_template.part_bkg_pdf

    # Plot fit result
    model_kstar_template.plot_fit(
        b_mass_branch,
        dataset_kstar,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_kstar_partial_template.pdf'),
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
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'cb+cb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    # model_final.add_background_model('part_bkg_pdf', 'generic', fit_params.fit_defaults, let_float=True)
    model_final.add_background_model('part_bkg_pdf', model_kstar_template.background_models['part_bkg_pdf'])

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 0.8, 0.0, 5*dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 0.15,0.0, 5*dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 0.01,0.0, 5*dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final',
        'Sum of Signal and Background PDFs',
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
    model_final.background_models['comb_bkg_pdf'].exp_slope.setConstant(False)
    model_final.add_constraints({
        'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope']), ROOT.RooFit.RooConst(10))
    })

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf,
        ],
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    _cb1_coeff = model_final.signal_models['sig_pdf'].cb1_coeff.getVal()
    _cb2_coeff = model_final.signal_models['sig_pdf'].cb2_coeff.getVal()
    _norm_sf = 1 / (_cb1_coeff + _cb2_coeff)
    model_final.signal_models['sig_pdf'].cb1_coeff.setVal(_cb1_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].cb2_coeff.setVal(_cb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if get_yields:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])

    # Use function to grab yields
    if True:
        yields = {
            'yield_sig' : sig_coeff.getValV(),
            'yield_comb_bkg' : comb_bkg_coeff.getValV(),
            'yield_part_bkg' : part_bkg_coeff.getValV(),
        }
        return yields


def do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False):
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
        model_sig_template.add_signal_model('sig_pdf', 'cb+cb', fit_params.fit_defaults, let_float=True)
        model_sig_template.fit_model = model_sig_template.sig_pdf

        # Fit model to data
        model_sig_template.fit(dataset_mc, printlevel=printlevel)
        params = model_sig_template.fit_result.floatParsFinal()

        # Plot fit result
        model_sig_template.plot_fit(
            b_mass_branch,
            dataset_mc,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_sig_template.pdf'),
            fit_components = [
                model_sig_template.signal_models['sig_pdf'].cb1_pdf,
                model_sig_template.signal_models['sig_pdf'].cb2_pdf,
            ],
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Combinatorial Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesign_data_file, score_cut=0.)

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
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Add partial background shape to simplified fit
    if args.verbose:
        print('\nStarting Fit 3 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Build final Roofit model
    model_final = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data, 'channel_label' : fit_params.channel_label})
    model_final.add_signal_model('sig_pdf', 'cb+cb', template, let_float=False)
    model_final.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_final.add_background_model('part_bkg_pdf', 'generic', fit_params.fit_defaults, let_float=True)

    sig_coeff = ROOT.RooRealVar('sig_coeff'+fit_params.channel_label, 'Signal PDF Coefficient', 5000, 0.0, 5*dataset_data.numEntries())
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff'+fit_params.channel_label, 'Combinatorial Background Coefficient', 0.15,0.0, 5*dataset_data.numEntries())
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff'+fit_params.channel_label, 'Partially Reconstructed Background Coefficient', 0.01,0.0, 5*dataset_data.numEntries())
    model_final.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final'+fit_params.channel_label,
        'Sum of Signal and Background PDFs',
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
    model_final.constraints.update({
        'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_final.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope']), ROOT.RooFit.RooConst(0.5))
    })

    # Fit model to data
    model_final.fit(dataset_data, printlevel=printlevel)
    params = model_final.fit_result.floatParsFinal()

    # Plot fit result
    model_final.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_final.pdf'),
        fit_components = [
            model_final.sig_pdf,
            model_final.comb_bkg_pdf,
            model_final.part_bkg_pdf,
        ],
    )

    # Add normalization terms for Combine
    # sig_pdf_norm = ROOT.RooRealVar('sig_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', 5000, 0, 1000000)
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of combinatorial background events', comb_bkg_coeff.getVal(), 0, dataset_data.numEntries())
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf'+fit_params.channel_label+'_norm', 'Number of partially reconstructed background events', part_bkg_coeff.getVal(), 0, dataset_data.numEntries())

    # Renormalize signal pdf
    _cb1_coeff = model_final.signal_models['sig_pdf'].cb1_coeff.getVal()
    _cb2_coeff = model_final.signal_models['sig_pdf'].cb2_coeff.getVal()
    _norm_sf = 1 / (_cb1_coeff + _cb2_coeff)
    model_final.signal_models['sig_pdf'].cb1_coeff.setVal(_cb1_coeff * _norm_sf)
    model_final.signal_models['sig_pdf'].cb2_coeff.setVal(_cb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if write:
        write_workspace(output_params, args, model_final, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])

    # Use function to grab yields
    if get_yields:
        yields = {
            'yield_sig' : sig_coeff.getValV(),
            'yield_comb_bkg' : comb_bkg_coeff.getValV(),
            'yield_part_bkg' : part_bkg_coeff.getValV(),
        }

        return yields


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    if args.mode=='all':
        args.mode = 'rare'
        if args.verbose:
            print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
        do_signal_region_fit(dataset_params, output_params, fit_params, args)

        args.mode = 'jpsi'
        if args.verbose:
            print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
        do_jpsi_control_region_fit(dataset_params, output_params, fit_params, args)

        args.mode = 'psi2s'
        if args.verbose:
            print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
        do_psi2s_control_region_fit(dataset_params, output_params, fit_params, args)

    elif args.mode=='rare':
        do_signal_region_fit(dataset_params, output_params, fit_params, args)

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
    args = parser.parse_args()

    main(args)
