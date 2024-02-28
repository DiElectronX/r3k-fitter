import os
import ROOT
import yaml
import argparse

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def set_verbosity(args):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if args.verbose else ROOT.kWarning
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if args.verbose else ROOT.RooFit.ERROR)
    printlevel = ROOT.RooFit.PrintLevel(1 if args.verbose else -1)

    return printlevel


def set_mode(dataset_params, output_params, fit_params, args):
    mode = args.mode

    valid_file_key = [i for i in vars(dataset_params).keys() if mode in i]
    assert len(valid_file_key)==1
    dataset_params.mc_sig_file = getattr(dataset_params,valid_file_key[0])

    valid_fit_key = [i for i in fit_params.regions.keys() if mode in i]
    assert len(valid_fit_key)==1
    fit_params.region = fit_params.regions[valid_fit_key[0]]
    fit_params.fit_defaults = fit_params.region['defaults']


def do_fit(dataset_params, output_params, fit_params, args):
    printlevel = set_verbosity(args)
    set_mode(dataset_params, output_params, fit_params, args)
    makedirs(output_params.output_dir)

    # Part 1 - Fit signal template from MC sample
    f_in = ROOT.TFile(dataset_params.mc_sig_file, 'READ')
    tree = f_in.Get(dataset_params.tree_name)
    var = dataset_params.b_mass_branch
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'Mass [GeV]', *fit_params.full_mass_range)
    bdt_branch = ROOT.RooRealVar(dataset_params.score_branch, 'Weight', -100., 100.)
    ll_mass_branch = ROOT.RooRealVar(dataset_params.ll_mass_branch, 'Weight', -100., 100.)
    b_mass_branch.setRange('full', *fit_params.full_mass_range)
    variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch)
    dataset_mc = ROOT.RooDataSet('dataset_mc', 'dataset_mc', tree, variables)
    cutstring = '{}>4.0&&{}>{}&&{}<{}'.format(
        dataset_params.score_branch,
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][0],
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][1],
    )
    cuts = ROOT.TCut(cutstring)
    dataset_mc = dataset_mc.reduce(cuts.GetTitle())
    f_in.Close()

    cb_mean = ROOT.RooRealVar(
        'cb_mean',
        'DS-CB: location parameter of the Gaussian component',
        *fit_params.fit_defaults['cb_mean'])
    cb_sigma = ROOT.RooRealVar(
        'cb_sigma',
        'DS-CB: width parameter of the Gaussian component',
        *fit_params.fit_defaults['cb_sigma'])
    cb_alphaL = ROOT.RooRealVar(
        'cb_alphaL',
        'DS-CB: location of transition to a power law on the left, in std devs away from mean',
        *fit_params.fit_defaults['cb_alphaL'])
    cb_nL = ROOT.RooRealVar(
        'cb_nL',
        'DS-CB: exponent of power-law tail on the left',
        *fit_params.fit_defaults['cb_nL'])
    cb_alphaR = ROOT.RooRealVar(
        'cb_alphaR',
        'DS-CB: location of transition to a power law on the right, in std devs away from mean',
        *fit_params.fit_defaults['cb_alphaR'])
    cb_nR = ROOT.RooRealVar(
        'cb_nR',
        'DS-CB: exponent of power-law tail on the right',
        *fit_params.fit_defaults['cb_nR'])
    cb_pdf = ROOT.RooTwoSidedCBShape(
        'cb_pdf',
        'Double-sided crystal-ball pdf',
        b_mass_branch,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

    model = cb_pdf

    result_pt1 = model.fitTo(dataset_mc, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
    params = result_pt1.floatParsFinal()

    frame = b_mass_branch.frame()
    dataset_mc.plotOn(frame)
    model.plotOn(frame)

    canvas = ROOT.TCanvas('canvas', 'Fitting Example', 800, 600)
    frame.Draw()
    canvas.SaveAs(os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt1_sig_template.png'))

    signal_template = {}
    for param in params:
        if param.GetName() in ['cb_alphaL','cb_alphaR','cb_mean','cb_nL','cb_nR','cb_sigma','signal_num']:
            signal_template[param.GetName()] = param.getVal()

    if args.verbose:
        print('Fitted MC Template Parameters:')
        for k, v in signal_template.items():
            print('\t'+k+' = '+str(round(v,2)))

    with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt1_sig_template.yml'), 'w') as file:
        yaml.dump(signal_template, file)

    # Part 2 - Use template to fit signal + exponential in data
    f_in = ROOT.TFile(dataset_params.data_file, 'READ')
    tree = f_in.Get(dataset_params.tree_name)
    var = dataset_params.b_mass_branch
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'Mass [GeV]', *fit_params.full_mass_range)
    bdt_branch = ROOT.RooRealVar(dataset_params.score_branch, 'Weight', -100., 100.)
    ll_mass_branch = ROOT.RooRealVar(dataset_params.ll_mass_branch, 'Weight', -100., 100.)
    b_mass_branch.setRange('full', *fit_params.full_mass_range)
    variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch)
    dataset_data = ROOT.RooDataSet('dataset_data', 'dataset_data', tree, variables)
    cutstring = '{}>4.0&&{}>{}&&{}<{}'.format(
        dataset_params.score_branch,
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][0],
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][1],
    )
    cuts = ROOT.TCut(cutstring)
    dataset_data = dataset_data.reduce(cuts.GetTitle())
    f_in.Close()


    cb_mean = ROOT.RooRealVar(
        'cb_mean',
        'DS-CB: location parameter of the Gaussian component',
        signal_template['cb_mean'])
    cb_sigma = ROOT.RooRealVar(
        'cb_sigma',
        'DS-CB: width parameter of the Gaussian component',
        signal_template['cb_sigma'])
    cb_alphaL = ROOT.RooRealVar(
        'cb_alphaL',
        'DS-CB: location of transition to a power law on the left, in std devs away from mean',
        signal_template['cb_alphaL'])
    cb_nL = ROOT.RooRealVar(
        'cb_nL',
        'DS-CB: exponent of power-law tail on the left',
        signal_template['cb_nL'])
    cb_alphaR = ROOT.RooRealVar(
        'cb_alphaR',
        'DS-CB: location of transition to a power law on the right, in std devs away from mean',
        signal_template['cb_alphaR'])
    cb_nR = ROOT.RooRealVar(
        'cb_nR',
        'DS-CB: exponent of power-law tail on the right',
        signal_template['cb_nR'])
    cb_pdf = ROOT.RooTwoSidedCBShape(
        'cb_pdf',
        'Double-sided crystal-ball pdf',
        b_mass_branch,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

    exp_slope = ROOT.RooRealVar(
        'exp_slope',
        'slope of exponential',
        *fit_params.fit_defaults['exp_slope'])
    expo_pdf = ROOT.RooExponential('expo_pdf', 'Exponential PDF', b_mass_branch, exp_slope)

    cb_coeff = ROOT.RooRealVar('cb_coeff', 'Gaussian Coefficient', 0.8, 0.0, 1.0)
    exp_coeff = ROOT.RooRealVar('exp_coeff', 'Exponential Coefficient', 0.2,0.0, 1.0)

    pdf_sum = ROOT.RooAddPdf('pdf_sum', 'Sum of Gaussian and Exponential',
                             ROOT.RooArgList(cb_pdf, expo_pdf),
                             ROOT.RooArgList(cb_coeff, exp_coeff)
    )

    result_pt2 = pdf_sum.fitTo(dataset_data, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
    params = result_pt2.floatParsFinal()

    frame = b_mass_branch.frame()
    dataset_data.plotOn(frame)
    pdf_sum.plotOn(frame)

    canvas = ROOT.TCanvas('canvas', 'Fitting Example', 800, 600)
    frame.Draw()
    canvas.SaveAs(os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt2_exp_template.png'))

    exp_template = {}
    for param in params:
        if param.GetName() in ['exp_slope']:
            exp_template[param.GetName()] = param.getVal()

    if args.verbose:
        print('Fitted Exponential Template Parameters:')
        for k, v in exp_template.items():
            print('\t'+k+' = '+str(round(v,2)))

    with open(os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt2_exp_template.yml'), 'w') as file:
        yaml.dump(exp_template, file)

    # Part 3 - Add partial background shape to simplified fit
    cb_mean = ROOT.RooRealVar(
        'cb_mean',
        'DS-CB: location parameter of the Gaussian component',
        signal_template['cb_mean'])
    cb_sigma = ROOT.RooRealVar(
        'cb_sigma',
        'DS-CB: width parameter of the Gaussian component',
        signal_template['cb_sigma'])
    cb_alphaL = ROOT.RooRealVar(
        'cb_alphaL',
        'DS-CB: location of transition to a power law on the left, in std devs away from mean',
        signal_template['cb_alphaL'])
    cb_nL = ROOT.RooRealVar(
        'cb_nL',
        'DS-CB: exponent of power-law tail on the left',
        signal_template['cb_nL'])
    cb_alphaR = ROOT.RooRealVar(
        'cb_alphaR',
        'DS-CB: location of transition to a power law on the right, in std devs away from mean',
        signal_template['cb_alphaR'])
    cb_nR = ROOT.RooRealVar(
        'cb_nR',
        'DS-CB: exponent of power-law tail on the right',
        signal_template['cb_nR'])
    cb_pdf = ROOT.RooTwoSidedCBShape(
        'cb_pdf',
        'Double-sided crystal-ball pdf',
        b_mass_branch,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

    exp_slope = ROOT.RooRealVar(
        'exp_slope',
        'slope of exponential',
        exp_template['exp_slope'])
    expo_pdf = ROOT.RooExponential('expo_pdf','Exponential PDF',b_mass_branch,exp_slope)

    part_exp_slope = ROOT.RooRealVar(
        'part_exp_slope',
        'slope of exponential',
        *fit_params.fit_defaults['part_exp_slope'])
    erfc_mean = ROOT.RooRealVar(
        'erfc_mean',
        'mean of the Erfc gaussian',
        *fit_params.fit_defaults['erfc_mean'])
    erfc_sigma = ROOT.RooRealVar(
        'erfc_sigma',
        'width of the Erfc gaussian',
        *fit_params.fit_defaults['erfc_sigma'])
    function = 'TMath::Exp(TMath::Abs(part_exp_slope)*('+dataset_params.b_mass_branch+'-erfc_mean))*TMath::Erfc(('+dataset_params.b_mass_branch+'-erfc_mean)/erfc_sigma)'
    generic_pdf = ROOT.RooGenericPdf(
        'generic_pdf',
        'generic pdf (expo*erfc)',
        function,ROOT.RooArgSet(b_mass_branch,erfc_mean,erfc_sigma,part_exp_slope))


    cb_coeff = ROOT.RooRealVar('cb_coeff', 'Gaussian Coefficient', 0.8, 0.0, 1.0)
    exp_coeff = ROOT.RooRealVar('exp_coeff', 'Exponential Coefficient', 0.15,0.0, 0.9)
    part_coeff = ROOT.RooRealVar('part_coeff', 'part Coefficient', 0.01,0.0, 0.25)
    pdf_sum_final = ROOT.RooAddPdf('pdf_sum_final', 'Sum of Gaussian and Exponential',
                                   ROOT.RooArgList(cb_pdf, expo_pdf, generic_pdf),
                                   ROOT.RooArgList(cb_coeff, exp_coeff, part_coeff)
    )

    result_pt3 = pdf_sum_final.fitTo(dataset_data, ROOT.RooFit.Save(), ROOT.RooFit.Range('full'), printlevel)
    generic_pdf_norm = ROOT.RooRealVar('generic_pdf_norm', 'Number of partially reconstructed background events', 0, 0, 1000000)
    expo_pdf_norm = ROOT.RooRealVar('expo_pdf_norm', 'Number of combinatorial background events', 0, 0, 1000000)
    params = result_pt3.floatParsFinal()

    frame = b_mass_branch.frame()
    dataset_data.plotOn(frame)
    pdf_sum_final.plotOn(frame)

    exp_plot = ROOT.RooArgSet(expo_pdf)
    cb_plot = ROOT.RooArgSet(cb_pdf)
    part_plot = ROOT.RooArgSet(generic_pdf)

    pdf_sum_final.plotOn(frame,ROOT.RooFit.Components(cb_plot),
                 ROOT.RooFit.LineStyle(ROOT.kDotted))
    pdf_sum_final.plotOn(frame,ROOT.RooFit.Components(exp_plot),
                 ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf_sum_final.plotOn(frame,ROOT.RooFit.Components(part_plot),
                 ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(
            ROOT.kOrange))

    canvas = ROOT.TCanvas('canvas', 'Fitting Example', 800, 600)
    frame.Draw()
    canvas.SaveAs(os.path.join(output_params.output_dir,'fit_'+args.mode+'_pt3_final.png'))

    # Write final fit to RooWorkspace
    f_out = ROOT.TFile(os.path.join(output_params.output_dir,'workspace_'+args.mode+'.root'), 'RECREATE')

    workspace = ROOT.RooWorkspace('workspace','workspace')
    getattr(workspace, 'import')(dataset_data)
    getattr(workspace, 'import')(pdf_sum_final)
    getattr(workspace, 'import')(expo_pdf_norm)
    getattr(workspace, 'import')(generic_pdf_norm)

    if args.verbose:
        workspace.Print()

    workspace.Write()
    f_out.Close()


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
    args, _ = parser.parse_known_args()

    main(args)
