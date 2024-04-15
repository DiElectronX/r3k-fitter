import os
import yaml
import ROOT

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

    valid_file_key = [i for i in vars(dataset_params).keys() if mode+'_file' in i]
    assert len(valid_file_key)==1
    dataset_params.mc_sig_file = getattr(dataset_params,valid_file_key[0])

    valid_fit_key = [i for i in fit_params.regions.keys() if mode in i]
    assert len(valid_fit_key)==1
    fit_params.region = fit_params.regions[valid_fit_key[0]]
    fit_params.channel_label = '_'+valid_fit_key[0]+'_region'
    fit_params.fit_defaults = fit_params.region['defaults']
    fit_params.blinded = fit_params.region.get('blinded')


def save_params(params, template_filename, fit_params, args, get_params=False):
    template = {}
    for param in params:
        if param.GetName() in fit_params.fit_defaults.keys():
            template[param.GetName()] = param.getVal()

    if args.verbose:
        print('Fitted Template Parameters:')
        for k, v in template.items():
            print('\t'+k+' = '+str(round(v,2)))

    if os.path.isfile(template_filename):
        with open(template_filename) as f:
            old_template = yaml.safe_load(f)
    else:
        old_template = None

    if old_template:
        template.update(old_template)

    with open(template_filename, 'w') as f:
        yaml.dump(template, f)

    return template

def prepare_inputs(dataset_params, fit_params, isData=True, set_file=None, score_cut=None, binned=False, unblind=False):
    # Read data from config file or manually set input
    if set_file is None:
        f_in = ROOT.TFile(dataset_params.data_file if isData else dataset_params.mc_sig_file, 'READ')
    else:
        f_in = ROOT.TFile(set_file, 'READ')

    # Read branches
    tree = f_in.Get(dataset_params.tree_name)
    b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'Mass [GeV]', *fit_params.full_mass_range)
    bdt_branch = ROOT.RooRealVar(dataset_params.score_branch, 'Weight', -100., 100.)
    ll_mass_branch = ROOT.RooRealVar(dataset_params.ll_mass_branch, 'Weight', -100., 100.)

    # Set fit ranges
    b_mass_branch.setRange('full', *fit_params.full_mass_range)
    blindDataset = (isData and (fit_params.blinded)) and not unblind
    if blindDataset:
        assert len(fit_params.blinded)==2
        sb1_range = (fit_params.full_mass_range[0], fit_params.blinded[0])
        sb2_range = (fit_params.blinded[1], fit_params.full_mass_range[1])
        b_mass_branch.setRange('sb1', *sb1_range)
        b_mass_branch.setRange('sb2', *sb2_range)

    if isData:
        variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch)
        dataset = ROOT.RooDataHist('dataset_data'+fit_params.channel_label, 'Dataset', tree, variables) \
            if binned else ROOT.RooDataSet('dataset_data'+fit_params.channel_label, 'Dataset', tree, variables)
    else:
        weight_branch = ROOT.RooRealVar(dataset_params.mc_weight_branch, 'Weight', -100., 100.)
        variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch, weight_branch)
        dataset = ROOT.RooDataHist('dataset_mc'+fit_params.channel_label, 'Dataset', tree, variables, weight_branch.GetName()) \
            if binned else ROOT.RooDataSet('dataset_mc'+fit_params.channel_label, 'Dataset', tree, variables, weight_branch.GetName())

    cutstring = '{}>{}&&{}>{}&&{}<{}'.format(
        dataset_params.score_branch,
        fit_params.bdt_score_cut if score_cut is None else score_cut,
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][0],
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][1],
    )
    cuts = ROOT.TCut(cutstring)
    dataset = dataset.reduce(cuts.GetTitle())

    if blindDataset:
        dataset = dataset.reduce(ROOT.RooFit.CutRange('sb1,sb2'))

    f_in.Close()

    return b_mass_branch, dataset


def format_params(params, let_float=False):
    params = {k : (v if isinstance(v,list) else (v,)) for k, v in params.items()}
    params = {k : (v if let_float else (v[0],)) for k, v in params.items()}

    return params


def write_workspace(output_params, args, model, extra_objs=[]):
    f_out = ROOT.TFile(os.path.join(output_params.output_dir,'workspace_'+args.mode+'.root'), 'RECREATE')
    workspace = ROOT.RooWorkspace('workspace','workspace')
    getattr(workspace, 'import')(model.dataset)
    getattr(workspace, 'import')(model.fit_model)

    for obj in extra_objs:
        getattr(workspace, 'import')(obj)

    if args.verbose:
        workspace.Print()

    workspace.Write()
    f_out.Close()
