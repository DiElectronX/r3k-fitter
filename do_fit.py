import os
import ROOT
import yaml
import argparse
from utils import *
from fit_models import Model


def do_fit(dataset_params, output_params, fit_params, args):
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
        model_pt1 = Model({'branch' : b_mass_branch, 'dataset' : dataset_mc})
        model_pt1.build_signal_model('cb+gauss', b_mass_branch, fit_params.fit_defaults, let_float=True)
        model_pt1.fit_model = model_pt1.signal_model

        # Fit model to data
        model_pt1.fit_result = model_pt1.signal_model.fitTo(dataset_mc, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
        params = model_pt1.fit_result.floatParsFinal()
        
        # Plot fit result
        model_pt1.plot_fit(
            b_mass_branch,
            dataset_mc,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt1_sig_template.png'),
            fit_components = [
                model_pt1.cb_pdf,
                model_pt1.gauss_pdf,
            ],
        )

        # Save fit shape parameters
        template = save_params(params, fit_params, output_params, args)

    # Part 2 - Use template to fit signal + exponential in data
    if not args.cache:
        if args.verbose:
            print('\nStarting Fit 2 - Single-Background Template\n{}'.format(50*'~'))

        # Import ROOT file dataset
        b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

        # Build Roofit model for signal + exponential background
        model_pt2 = Model({'branch' : b_mass_branch, 'dataset' : dataset_data})
        model_pt2.build_signal_model('cb+gauss', b_mass_branch, template, let_float=False)
        model_pt2.add_background_model('comb_bkg_pdf', 'exp', b_mass_branch, fit_params.fit_defaults, let_float=True)

        sig_coeff = ROOT.RooRealVar('sig_coeff', 'CB+Gaussian Coefficient', 0.5, 0.0, 1.0)
        comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff', 'Exponential Coefficient', 0.5,0.0, 1.0)
        model_pt2.fit_model = ROOT.RooAddPdf('pdf_sum_pt2', 'Sum of Gaussian and Exponential',
                                 ROOT.RooArgList(model_pt2.signal_model, model_pt2.background_models['comb_bkg_pdf']),
                                 ROOT.RooArgList(sig_coeff, comb_bkg_coeff)
        )

        # Fit model to data
        model_pt2.fit_result = model_pt2.fit_model.fitTo(dataset_data, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
        params = model_pt2.fit_result.floatParsFinal()

        # Plot fit result
        model_pt2.plot_fit(
            b_mass_branch,
            dataset_data,
            os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt2_exp_template.png'),
            fit_components = [
                model_pt2.signal_model,
                model_pt2.background_models['comb_bkg_pdf'],
            ],
        )

        # Save fit shape parameters
        template = save_params(params, fit_params, output_params, args)

    # Part 3 - Add partial background shape to simplified fit
    if args.verbose:
        print('\nStarting Fit 3 - Final Double-Background Model\n{}'.format(50*'~'))

    if args.cache:
        # Load fit shape templates from file
        with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_template.yml'), 'r') as file:
            template = yaml.safe_load(file)

        # Import ROOT file dataset
        b_mass_branch, dataset_data = prepare_inputs(dataset_params, fit_params, isData=True)

    # Build Roofit model for signal + 2-shape background
    model_pt3 = Model({'branch' : b_mass_branch, 'dataset' : dataset_data})
    model_pt3.build_signal_model('cb+gauss', b_mass_branch, template, let_float=False)
    model_pt3.add_background_model('comb_bkg_pdf', 'exp', b_mass_branch, template, let_float=False)
    model_pt3.add_background_model('part_bkg_pdf', 'generic', b_mass_branch, fit_params.fit_defaults, let_float=True)

    sig_coeff = ROOT.RooRealVar('sig_coeff', 'Signal PDF Coefficient', 0.8, 0.0, 1.0)
    comb_bkg_coeff = ROOT.RooRealVar('comb_bkg_coeff', 'Combinatorial Background Coefficient', 0.15,0.0, 0.9)
    part_bkg_coeff = ROOT.RooRealVar('part_bkg_coeff', 'Partially Reconstructed Background Coefficient', 0.01,0.0, 0.25)
    model_pt3.fit_model = ROOT.RooAddPdf('pdf_sum_final', 'Sum of Gaussian and Exponential',
                                   ROOT.RooArgList(model_pt3.signal_model, model_pt3.background_models['comb_bkg_pdf'], model_pt3.background_models['part_bkg_pdf']),
                                   ROOT.RooArgList(sig_coeff, comb_bkg_coeff, part_bkg_coeff)
    )

    # Fit model to data
    model_pt3.fit_result = model_pt3.fit_model.fitTo(dataset_data, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
    params = model_pt3.fit_result.floatParsFinal()

    # Plot fit result
    model_pt3.plot_fit(
        b_mass_branch,
        dataset_data,
        os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt3_final.png'),
        fit_components = [
            model_pt3.signal_model,
            model_pt3.background_models['comb_bkg_pdf'],
            model_pt3.background_models['part_bkg_pdf'],
        ],
    )

    # Add normalization terms for Combine
    comb_bkg_pdf_norm = ROOT.RooRealVar('comb_bkg_pdf_norm', 'Number of combinatorial background events', 0, 0, 1000000)
    part_bkg_pdf_norm = ROOT.RooRealVar('part_bkg_pdf_norm', 'Number of partially reconstructed background events', 0, 0, 1000000)

    # Renormalize signal pdf
    _cb_coeff = model_pt3.cb_coeff.getVal()
    _gauss_coeff = model_pt3.gauss_coeff.getVal()
    _norm_sf = 1 / (_cb_coeff + _gauss_coeff)
    model_pt3.cb_coeff.setVal(_cb_coeff * _norm_sf)
    model_pt3.gauss_coeff.setVal(_gauss_coeff * _norm_sf)

    # Write final fit to RooWorkspace
    write_workspace(output_params, args, model_pt3, extra_objs=[comb_bkg_pdf_norm, part_bkg_pdf_norm])


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_params = argparse.Namespace(**cfg['datasets'])
    output_params = argparse.Namespace(**cfg['output'])
    fit_params = argparse.Namespace(**cfg['fit'])

    do_fit(dataset_params, output_params, fit_params, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='fit_cfg.yml', help='fit configuration file (.yml)')
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['jpsi', 'psi2s'], help='which fit to perform')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print fitting procedure to stdout')
    parser.add_argument('-lc', '--loadcache', dest='cache', action='store_true', help='load cached templates if available')
    args, _ = parser.parse_known_args()

    main(args)
