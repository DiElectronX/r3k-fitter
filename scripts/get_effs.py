import sys
import yaml
import argparse
import numpy as np
import uproot as ur

with open('fit_cfg.yml', 'r') as f:
    cfg = yaml.safe_load(f)

dataset_params = argparse.Namespace(**cfg['datasets'])
output_params = argparse.Namespace(**cfg['output'])
fit_params = argparse.Namespace(**cfg['fit'])

mc_files = {
    'Rare MC'    : dataset_params.rare_file,
    'J/Psi MC'   : dataset_params.jpsi_file,
    'Psi(2s) MC' : dataset_params.psi2s_file,
}

fit_regions = {reg : fit_params.regions[reg]['ll_mass_range'] for reg in fit_params.regions.keys()}

for filename, filepath in mc_files.items():
    with ur.open(filepath) as f:
        mass = f[dataset_params.tree_name]['Mll'].array()
        
        for regname, regrange in fit_regions.items():
            eff = np.sum((mass>regrange[0]) & (mass<regrange[1]), dtype=float) / mass.size
            print('{} efficiency in {} region = {}'.format(filename, regname, round(eff,5)))
