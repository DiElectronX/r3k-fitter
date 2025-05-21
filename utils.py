import os
import sys
import re
import yaml
import ROOT
from io import StringIO 
from tqdm import tqdm
from tqdm.contrib import tzip
import tempfile
import numpy as np

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
    
    file_mode = 'rare' if ('q2' in mode) else mode
    valid_file_key = [i for i in vars(dataset_params).keys() if file_mode+'_file' in i]
    assert len(valid_file_key)==1
    dataset_params.mc_sig_file = getattr(dataset_params,valid_file_key[0])

    valid_fit_key = [i for i in fit_params.regions.keys() if mode in i]
    assert len(valid_fit_key)==1
    fit_params.region = fit_params.regions[valid_fit_key[0]]
    fit_params.channel_label = '_'+valid_fit_key[0]+'_region'
    fit_params.fit_range = fit_params.region['fit_range']
    fit_params.ll_mass_range = fit_params.region['ll_mass_range']
    fit_params.fit_defaults = fit_params.region['defaults']
    fit_params.blinded = fit_params.region.get('blinded')
    fit_params.toy_signal_yield = fit_params.region.get('toy_signal_yield')


def save_params(params, template_filename, fit_params, args, update_dict=None, get_params=False, just_print=False, lock_file=False):
    template = {}
    for param in params:
        for key in fit_params.fit_defaults.keys(): 
            if key in param.GetName():
                template[key] = param.getVal()

    if args.verbose or just_print:
        print('Fitted Template Parameters:')
        for k, v in template.items():
            print('\t'+k+' = '+str(round(v,2)))
        
        if just_print:
            return

    if os.path.isfile(template_filename):
        with open(template_filename, 'r') as f:
            old_template = yaml.safe_load(f)
    else:
        old_template = None 

    if old_template:
        old_template.update(template)
        template = old_template

    if update_dict:
        update_dict.update(template)
        template = update_dict

    if not lock_file: 
        with open(template_filename, 'w') as f:
            yaml.dump(template, f)

    return template


def prepare_inputs(dataset_params, fit_params, b_mass_branch=None, isData=True, set_file=None, set_tree=None, score_cut=None, binned=False, unblind=False, weight_branch_name=None, weight_norm=None, weight_sf=None):
    # Read data from config file or manually set input
    if set_file is None:
        f_in = ROOT.TFile(dataset_params.data_file if isData else dataset_params.mc_sig_file, 'READ')
    else:
        f_in = ROOT.TFile(set_file, 'READ') 
    
    # Read branches
    tree = f_in.Get(dataset_params.tree_name if set_tree is None else set_tree)
    # if b_mass_branch is None:
    #     b_mass_branch = ROOT.RooRealVar(dataset_params.b_mass_branch, 'B Candidate Mass [GeV]', *fit_params.fit_range)
    bdt_branch = ROOT.RooRealVar(dataset_params.score_branch, 'BDT Score', -100., 100.)
    ll_mass_branch = ROOT.RooRealVar(dataset_params.ll_mass_branch, 'Di-Lepton Mass [GeV]', -100., 100.)

    # Take region cuts from cfg file
    cutstring = '{}>{}&&{}>{}&&{}<{}'.format(
        dataset_params.score_branch,
        fit_params.bdt_score_cut if score_cut is None else score_cut,
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][0],
        dataset_params.ll_mass_branch,
        fit_params.region['ll_mass_range'][1],
    )
    cutvar = ROOT.RooFormulaVar('cutvar','cutvar',cutstring,ROOT.RooArgList(bdt_branch, ll_mass_branch))

    # Set fit ranges
    # b_mass_branch.setRange('full', *fit_params.fit_range)
    blindDataset = (isData and (fit_params.blinded)) and not unblind
    # if blindDataset:
    #     assert len(fit_params.blinded)==2
    #     sb1_range = (fit_params.fit_range[0], fit_params.blinded[0])
    #     sb2_range = (fit_params.blinded[1], fit_params.fit_range[1])
    #     b_mass_branch.setRange('sb1', *sb1_range)
    #     b_mass_branch.setRange('sb2', *sb2_range)

    # Generate dataset and scale if specified
    if isData:
        variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch)
        tmp_dataset = ROOT.RooDataHist('tmp_dataset_data'+fit_params.channel_label, 'Dataset', tree, variables, cutvar) \
            if binned else ROOT.RooDataSet('tmp_dataset_data'+fit_params.channel_label, 'Dataset', tree, variables, cutvar)
        
        dataset = tmp_dataset.Clone(('dataset_data' if isData else 'dataset_mc')+fit_params.channel_label)
    else:
        rdf = ROOT.RDataFrame(tree)
        weight_branch_name = dataset_params.mc_weight_branch if weight_branch_name is None else weight_branch_name
        weight_sum = rdf.Filter(cutstring).Sum(weight_branch_name).GetValue()
        weight_branch = ROOT.RooRealVar(weight_branch_name, 'Weight', -100., 100.)
        variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch, weight_branch)

        tmp_dataset = ROOT.RooDataHist('tmp_dataset_mc'+fit_params.channel_label, 'Dataset', tree, variables, cutvar, weight_branch.GetName()) \
            if binned else ROOT.RooDataSet('tmp_dataset_mc'+fit_params.channel_label, 'Dataset', tree, variables, cutvar, weight_branch.GetName())

        if weight_sf or weight_norm: 
            # Calculate renormalization factor
            assert not (weight_norm and weight_sf)
            if weight_norm:
                weight_sf = weight_norm / weight_sum

            new_weight_branch = ROOT.RooRealVar('new_weight', 'new_weight', 1, -100000, 100000)
            new_variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch, new_weight_branch)
            dataset = ROOT.RooDataSet(('dataset_data' if isData else 'dataset_mc')+fit_params.channel_label, 'Dataset', new_variables, new_weight_branch.GetName())
            for i in range(0,tmp_dataset.numEntries()):
                args = tmp_dataset.get(int(i))
                new_b_mass_branch = args.find(dataset_params.b_mass_branch).getVal()
                new_bdt_branch = args.find(dataset_params.score_branch).getVal()
                new_ll_mass_branch = args.find(dataset_params.ll_mass_branch).getVal()
                new_weight = args.find(weight_branch.GetName()).getVal()*weight_sf
                b_mass_branch.setVal(new_b_mass_branch)
                bdt_branch.setVal(new_bdt_branch)
                ll_mass_branch.setVal(new_ll_mass_branch)
                new_weight_branch.setVal(new_weight)
                dataset.add(new_variables,new_weight)
        else:
            dataset = tmp_dataset.Clone(('dataset_data' if isData else 'dataset_mc')+fit_params.channel_label)

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


def integrate(var, model, integral_range, coeffs=None):
    var.setRange('int_range', *integral_range)
    rangeset = ROOT.RooFit.Range('int_range')
    xset = ROOT.RooArgSet(var)
    nset = ROOT.RooFit.NormSet(xset)
    integral_unscaled = model.createIntegral(xset, nset, rangeset)

    if coeffs is None:
        final_yield = model.expectedEvents(xset)
        final_yield_err = 0
    else:
        final_yield = np.sum([y.getVal() for y in coeffs]) if isinstance(coeffs,list) else coeffs.getVal()
        final_yield_err = np.sqrt(np.sum([y.getError()**2 for y in coeffs])) if isinstance(coeffs,list) else coeffs.getError()
   
    integral = integral_unscaled.getVal() * final_yield
    integral_err = final_yield_err*integral_unscaled.getVal()

    return integral, integral_err


def get_roofit_comp_names(frame):
    if not hasattr(ROOT, "capturePrint"):
        ROOT.gInterpreter.Declare("""
        #include <sstream>
        #include <string>
        #include <ostream>

        std::string capturePrint(RooPlot* frame) {
            std::ostringstream oss;
            std::streambuf* old_buf = std::cout.rdbuf(oss.rdbuf());
            frame->Print();
            std::cout.rdbuf(old_buf);
            return oss.str();
        }
        """)

    captured_output = ROOT.capturePrint(frame)
    parentheses_content = re.search(r'\((.*?)\)', captured_output).group(1)
    matches = [match.split("::")[-1] for match in parentheses_content.split(",")]
    
    return matches

def loop_wrapper(iterable, args, unit='working point', title=None):
    if title:
        print(title)
    if isinstance(iterable,zip):
        unzipped = list(iterable)
        return iterable if args.verbose else tqdm(unzipped, total=len(unzipped), unit=unit)
    else:
        return iterable if args.verbose else tqdm(iterable, total=len(iterable), unit=unit)
