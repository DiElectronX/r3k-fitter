import sys
import yaml
import argparse
import numpy as np
import uproot as ur

with open('fit_cfg.yml', 'r') as f:
    cfg = yaml.safe_load(f)

dataset_params = argparse.Namespace(**cfg['datasets'])
fit_params = argparse.Namespace(**cfg['fit'])

all_files = {name : path for name, path in vars(dataset_params).items() if name.endswith('file')}
fit_regions = {reg : fit_params.regions[reg]['ll_mass_range'] for reg in fit_params.regions.keys()}
bdt_cut = fit_params.bdt_score_cut

effs = {reg : {} for reg in fit_regions.keys()}
for filename, filepath in all_files.items():
    with ur.open(filepath) as f:
        mass = f[dataset_params.tree_name]['Mll'].array()
        bdt_score = f[dataset_params.tree_name]['bdt_score'].array()
        
        for regname, regrange in fit_regions.items():
            nevts = mass.size
            eff = np.sum((mass>regrange[0]) & (mass<regrange[1]) & (bdt_score>bdt_cut), dtype=float) / nevts 
            effs[regname][filename] = {}
            effs[regname][filename]['eff'] = eff
            effs[regname][filename]['nevts'] = nevts

print('\nSample Event Selection Efficiency:')
for reg, reg_dict in effs.items():
    print('  - In the "{}" region ({}<m(ll)<{}):'.format(reg,fit_regions[reg][0],fit_regions[reg][1]))
    for filename, vals in reg_dict.items():
        eff = vals['eff']
        print('    -  {} = {}'.format(filename, round(eff,4) if eff > .0001 else '{:.2E}'.format(eff)))

print('\nMC Pre-Selection Acceptence x Efficiency:')
ntoys = 500000000
for filename, vals in next(iter(effs.values())).items():
    if 'data' in filename:
        continue
    acc = float(vals['nevts']) / float(ntoys)
    print('  -  {} = {}'.format(filename, '{:.2E}'.format(acc)))
