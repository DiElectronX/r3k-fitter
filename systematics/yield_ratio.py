from uncertainties import ufloat
from uncertainties.umath import *
import csv
import pandas as pd
import os

class YieldData():
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.yields = dict(zip(self.data['Yield'], self.data['Value']))

    def get_yield(self, yield_name):
        return self.yields.get(yield_name, 'Yield not found.')

    def get_all_yields(self):
        return self.yields

def save_yields_err(yields, output_dir, filename):
    # os.makedirs(output_params.output_dir, exist_ok=True)
    # filepath = os.path.join(output_params.output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ratio", "Value"])
        for key, value in yields.items():
            writer.writerow([key, value])

# --------------------------------------------------------------------------------------------------------
def mc_yield_ratio(mc_file_nominal, mc_file_relaxed, dir_name):
    ### MC yields and ratios 
    '''
        The following are ratios calculation from MC yields 
        (referenced to both with or without dcb+dcb constraints on data)
        KEYS:
            ['yield_jpsi', 
            'yield_kstar_jpsi_kaon', 
            'yield_kstar_jpsi_pion', 
            'yield_k0star_jpsi_kaon', 
            'yield_k0star_jpsi_pion', 
            'yield_chic1_jpsi_kaon', 
            'yield_jpsipi_jpsi_pion', 
            'yield_part_bkg']
    '''
    # MC nominal yield
    mc_yield_nominal = YieldData(mc_file_nominal)
    mc_yield_nominal_uncer = ufloat(mc_yield_nominal.get_yield('yield_jpsi'), 0) # no error

    # MC yields with different cuts
    mc_ratio_dict = {}
    for file in [mc_file_nominal] + mc_file_relaxed:
        print(file)
        yields = YieldData(file)
        yields_uncer = ufloat(yields.get_yield('yield_jpsi'), 0)
        ratio_yield = yields_uncer/mc_yield_nominal_uncer
        lower_cut = (file.split('/')[-2].split('_')[-1]).replace('.','_')
        # if lower_cut == "nominal":
        #     lower_cut = "2_95"
        print(f'{lower_cut} : {ratio_yield}')
        mc_ratio_dict[f'mc_ratio_nominal_vs_{lower_cut}'] = ratio_yield

    save_yields_err(mc_ratio_dict, output_dir=dir_name, filename='mc_yield_ratio.csv')

# --------------------------------------------------------------------------------------------------------
# DATA yields and ratios
def data_yield_ratio(data_file_nominal, data_file_relaxed, dir_name):
    '''
        The following are ratios calculation from DATA yields.
        KEYS:
            ['yield_sig',
            'yield_sig_err',
            'yield_comb_bkg',
            'yield_comb_bkg_err',
            'yield_part_bkg',
            'yield_part_bkg_err',
            'yield_part_bkg_jpsipi_pion',
            'yield_part_bkg_jpsipi_pion_err']
    '''
    # DATA nominal yield
    yield_nominal = YieldData(data_file_nominal)
    yield_nominal_uncer = ufloat(yield_nominal.get_yield('yield_sig'), yield_nominal.get_yield('yield_sig_err'))

    # DATA yields with different cuts
    ratio_dict ={}
    for file in [data_file_nominal] + data_file_relaxed:
        print(file)
        yields = YieldData(file)
        yields_uncer = ufloat(yields.get_yield('yield_sig'), yields.get_yield('yield_sig_err'))
        ratio_yield = yields_uncer/yield_nominal_uncer

        # lower_cut = (file.split('/')[-2].split('_')[-1]).replace('.','_')
        lower_cut = file.split('/')[0].replace('.','_')
        # if lower_cut == "nominal":
        #     lower_cut = "2_95"
        print(f'{lower_cut} : {ratio_yield}')
        ratio_dict[f'data_ratio_nominal_vs_{lower_cut}'] = ratio_yield

    save_yields_err(ratio_dict, output_dir=dir_name, filename='data_yield_ratio.csv')

def nominal_yield_error(nominal_pdf0_file, nominal_pdf1_file, output_dir, output_file):
    '''
    Calculating the error between two nominal yields (with different pdf, etc)
    This is intended used in fitting parameterization
        comparing between alternative (CB+Gaussian) and baseline (dcb+dcb)
    '''

    yield_pdf0 = YieldData(nominal_pdf0_file)
    yield_pdf0_uncer = ufloat(yield_pdf0.get_yield('yield_sig'), yield_pdf0.get_yield('yield_sig_err'))
    yield_pdf1 = YieldData(nominal_pdf1_file)
    yield_pdf1_uncer = ufloat(yield_pdf1.get_yield('yield_sig'), yield_pdf1.get_yield('yield_sig_err'))

    # calculating the error w.r.t. the default (pdf0 = dcb+dcb)
    error = (yield_pdf1_uncer - yield_pdf0_uncer)/yield_pdf0_uncer
    error_dict = {'CB+Gauss vs dcb+dcb':error}
    save_yields_err(error_dict, output_dir=output_dir, filename=output_file)


def main():
# #===========================================================================================================
# Fit Parameterization systematics: yield-ratios with relaxed windows
# #===========================================================================================================
    mc_nominal = '2.9/mc_yield.csv'
    mc_relaxed = ['2.8/mc_yield.csv',
                  '2.7/mc_yield.csv',
                  '2.6/mc_yield.csv',
                  '2.5/mc_yield.csv',]

    data_monimal = '2.9_wExpCon_v8/fitter_outputs_minos/yield.csv'
    data_relaxed = ['2.8_wExpCon_v8/fitter_outputs_minos/yield.csv',
                    '2.7_wExpCon_v8/fitter_outputs_minos/yield.csv',
                    '2.6_wExpCon_v8/fitter_outputs_minos/yield.csv',
                    '2.5_wExpCon_v8/fitter_outputs_minos/yield.csv',]

    output_dir = 'yield_systematics_wExpCon_minos_v8'

    # computing the ratio
    mc_yield_ratio(mc_nominal, mc_relaxed, output_dir)
    data_yield_ratio(data_monimal, data_relaxed, output_dir)

if __name__ == '__main__':
    main()
