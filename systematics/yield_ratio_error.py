import ROOT

from uncertainties import ufloat
from uncertainties.umath import *
import csv
import pandas as pd
import os

import matplotlib.pyplot as plt
import argparse
import mplhep as hep

# Parse the central value and uncertainty from "Value" column
def parse_value(value_str):
    central, uncertainty = map(float, value_str.split("+/-"))
    return ufloat(central, uncertainty)

def save_yields_err(yields, output_dir, filename):
    # os.makedirs(output_params.output_dir, exist_ok=True)
    # filepath = os.path.join(output_params.output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cut", "Percent Difference"])
        for key, value in yields.items():
            writer.writerow([key, value])

def plot_compared_yields(file_list, labels_list, dir_name, output_name):
    '''
    taking multiple yield_file.csv, extract yields and errors and plot them on the same graph
    '''
    all_df = []

    for i, file in enumerate(file_list):
        # loading data
        df  = pd.read_csv(file)
        # extracting cut number 
        df['q2_cut'] = df['Ratio'].str.extract(r'_vs_([0-9]+_[0-9]+)')[0]
        df['q2_cut'] = df['q2_cut'].str.replace('_', '.', regex=False)
        # extract yield and error from data
        df[['yield', 'error']] = df['Value'].str.split(r'\+/-', expand=True)
        df['yield'] = df['yield'].astype(float)
        df['error'] = df['error'].astype(float)
        # labels
        df['label'] = labels_list[i]
        # append data
        all_df.append(df)
    # combined df
    combined_df = pd.concat(all_df, ignore_index=True)
    # bmasss_cut sorting
    combined_df = combined_df.sort_values(by='q2_cut').reset_index(drop=True)

    # template
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10,10))
    # fig, ax = plt.subplots(figsize=(10,7))

    # extract plots
    for label in combined_df['label'].unique():
        sub_df = combined_df[combined_df['label'] == label]
        plt.errorbar(sub_df['q2_cut'], sub_df['yield'], yerr=sub_df['error'],
                     fmt='o', label=label, capsize=5)

    # Set x-ticks labels
    tick_labels = [f'[{x_label}, 3.2]' for x_label in combined_df['q2_cut'].unique()]
    ax.set_xticks(combined_df['q2_cut'].unique())
    ax.set_xticklabels(tick_labels)
    # Labels and grid
    ax.set_xlabel('Lower dilepton mass window w.r.t. nominal', fontsize=18, loc='right')
    ax.set_ylabel('Yield Ratio [loosened q2/nominal]', fontsize=18, loc='top')
    ax.grid(True)
    # ax.legend(loc='upper right')
    ax.legend(loc='lower left')

    # Add CMS Label after data is plotted
    # hep.cms.label('Preliminary', data=True, lumi=None, year=2023, com=13.7, ax=ax)
    # hep.cms.label(data=True, lumi=None, year=2023, com=13.7, ax=ax)
    hep.cms.label(data=True, lumi=None, year=2023, ax=ax)

    # Save the plot
    plt.savefig(f'{dir_name}/{output_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{dir_name}/{output_name}.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

def yield_error(mc_ratio_file, data_ratio_file, output_dir, filename):
    mc_df  = pd.read_csv(mc_ratio_file)
    data_df  = pd.read_csv(data_ratio_file)

    # Apply the parsing function to both Data and MC files
    data_df['Value_with_uncertainty'] = data_df['Value'].apply(parse_value)
    mc_df['Value_with_uncertainty'] = mc_df['Value'].apply(parse_value)

    yield_diff_dict ={}
    for i in range(len(data_df)):
        cut = str(mc_df['Ratio'][i]).split('_')[-2:]
        cut = '.'.join(cut)
        # print(cut)
        sys_error = abs(data_df['Value_with_uncertainty'][i] - mc_df['Value_with_uncertainty'][i])/mc_df['Value_with_uncertainty'][i]
        yield_diff_dict[cut] = sys_error
        # print(sys_error)

    print(yield_diff_dict)
    save_yields_err(yield_diff_dict, output_dir, filename)

def main():

# #===========================================================================================================
# Fit Parameterization systematics: Relaxed Window Scan 
# #===========================================================================================================
    # Load Data and Monte Carlo CSV files
    data_file = 'yield_systematics_wExpCon_minos_v8/data_yield_ratio.csv'
    mc_file = 'yield_systematics_wExpCon_minos_v8/mc_yield_ratio.csv'

    output_dir = 'yield_systematics_wExpCon_minos_v8'

    # yield_difference between the MC vs Data CBGauss
    yield_error(mc_file, data_file, output_dir, filename='yield_difference.csv')

    # yield_difference between the Data CBGauss vs Data dcb+dcb
    # NOTE: mc_ratio_file = dcb+dcb baseline and data_ratio_file = CBGauss
    # data_file_dcbdcb = 'jpsi_radiative_tail/vDefault_wPartConFalse_wdcbCon/vDefault_wPartConFalse_wdcbCon_s1_minos/data_yield_ratio.csv'

    # yield_error(mc_ratio_file=mc_file,
    #             data_ratio_file=data_file,
    #             output_dir=output_dir,
    #             filename='yield_difference.csv')

    # plotting
    # data_file_dcbdcb = 'jpsi_radiative_tail/vDefault_wPartConFalse_wdcbCon/vDefault_wPartConFalse_wdcbCon_s1_minos/data_yield_ratio.csv'
    
    yield_files = [mc_file,
                   data_file,
                #    data_file_dcbdcb,
                   ]
    labels = ['MC Yield',
              'Data Yield',
            #   'Data Yield (dcb+dcb)',
              ]
    plot_compared_yields(file_list=yield_files, labels_list=labels, dir_name=output_dir, output_name='compared_yields')

if __name__ == '__main__':
    main()