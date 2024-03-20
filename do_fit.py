import ROOT
import yaml
import argparse
from utils import *
from fit_models import FitModel

ALLOWED_MODES = ['jpsi', 'psi2s']

def do_fit(dataset_params, output_params, fit_params, args, write=True, get_yields=False):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Part 1 - Fit signal template from MC sample
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 1 - MC Signal Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_mc = prepare_inputs(dataset_params, fit_params, isData=False)

        # Build Roofit model for signal
        model_pt1 = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_mc})
        model_pt1.add_signal_model('sig_pdf', 'cb+cb', fit_params.fit_defaults, let_float=True)
        model_pt1.fit_model = model_pt1.sig_pdf

        # Fit model to data
        model_pt1.fit(dataset_mc, printlevel=printlevel)
        params = model_pt1.fit_result.floatParsFinal()

        # Plot fit result
        model_pt1.plot_fit(
            b_mass_branch,
            dataset_mc,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt1_sig_template.pdf'),
            fit_components = [
                model_pt1.signal_models['sig_pdf'].cb1_pdf,
                model_pt1.signal_models['sig_pdf'].cb2_pdf,
            ],
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Part 2 - Fit combinatorial background to same-sign electron data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Single-Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True, set_file=dataset_params.samesignelectrons_data_file, score_cut=0.)

        # Build Roofit model for exponential background
        model_pt2 = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data})
        model_pt2.add_background_model('comb_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)
        model_pt2.fit_model = model_pt2.comb_bkg_pdf

        # Fit model to data
        model_pt2.fit(dataset_data, printlevel= printlevel)
        params = model_pt2.fit_result.floatParsFinal()

        # Plot fit result
        model_pt2.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt2_exp_template.pdf'),
        )

        # Save fit shape parameters
        template = save_params(params, os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), fit_params, args)

    # Part 3 - Add partial background shape to simplified fit
    if args.verbose:
        print('\nStarting Fit 3 - Final Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

    # Import ROOT file dataset
    b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Build final Roofit model
    model_pt3 = FitModel({'branch' : b_mass_branch, 'dataset' : dataset_data})
    model_pt3.add_signal_model('sig_pdf', 'cb+cb', template, let_float=False)
    model_pt3.add_background_model('comb_bkg_pdf', 'exp', template, let_float=False)
    model_pt3.add_background_model('part_bkg_pdf', 'generic', fit_params.fit_defaults, let_float=True)
    model_pt3.add_background_model('extra_bkg_pdf', 'exp', fit_params.fit_defaults, let_float=True)

    sig_coeff = ROOT.RooRealVar('sig_coeff', 'Signal PDF Coefficient', 0.8, 0.0, 100000.)
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff', 'Combinatorial Background Coefficient', 0.15,0.0, 100000.)
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff', 'Partially Reconstructed Background Coefficient', 0.01,0.0, 100000.)
    extra_bkg_coeff = ROOT.RooRealVar('extra_bkg_coeff', 'New Background Coefficient', 0.01,0.0, 100000.)
    model_pt3.fit_model = ROOT.RooAddPdf(
        'pdf_sum_final', 
        'Sum of Signal and Background PDFs',
        ROOT.RooArgList(
            model_pt3.sig_pdf, 
            model_pt3.comb_bkg_pdf, 
            model_pt3.part_bkg_pdf, 
            model_pt3.extra_bkg_pdf,
        ),
        ROOT.RooArgList(
            sig_coeff, 
            comb_bkg_coeff, 
            part_bkg_coeff, 
            extra_bkg_coeff,
        )
    )

    # Add gaussian contraints to fit parameters
    model_pt3.constraints.update({
        'exp_slope_constraint' : ROOT.RooGaussian('exp_slope_constraint', 'exp_slope_constraint', model_pt3.background_models['comb_bkg_pdf'].exp_slope, ROOT.RooFit.RooConst(template['exp_slope']), ROOT.RooFit.RooConst(0.5))
    })

    # Fit model to data
    model_pt3.fit(dataset_data, printlevel=printlevel)
    params = model_pt3.fit_result.floatParsFinal()

    # Plot fit result
    model_pt3.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt3_final.pdf'),
        fit_components = [
            model_pt3.sig_pdf,
            model_pt3.comb_bkg_pdf,
            model_pt3.part_bkg_pdf,
            model_pt3.extra_bkg_pdf,
        ],
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf_norm', 'Number of combinatorial background events', 0, 0, 1000000)
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf_norm', 'Number of partially reconstructed background events', 0, 0, 1000000)
    extra_bkg_pdf_norm = ROOT.RooRealVar('extra_bkg_pdf_norm', 'Number of additional background events', 0, 0, 1000000)

    # Renormalize signal pdf
    _cb1_coeff = model_pt3.signal_models['sig_pdf'].cb1_coeff.getVal()
    _cb2_coeff = model_pt3.signal_models['sig_pdf'].cb2_coeff.getVal()
    _norm_sf = 1 / (_cb1_coeff + _cb2_coeff)
    model_pt3.signal_models['sig_pdf'].cb1_coeff.setVal(_cb1_coeff * _norm_sf)
    model_pt3.signal_models['sig_pdf'].cb2_coeff.setVal(_cb2_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    if write:
        write_workspace(output_params, args, model_pt3, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])

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
        for mode in ALLOWED_MODES:
            args.mode=mode
            if args.verbose:
                print('\nRunning Fit in {} Mode\n{}'.format(args.mode, 50*'~'))
            do_fit(dataset_params, output_params, fit_params, args)
    else:
        do_fit(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which fit to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    args = parser.parse_args()

    main(args)
