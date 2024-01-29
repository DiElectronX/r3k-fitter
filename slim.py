import uproot
import numpy as np
import pandas as pd
import awkward as ak
import bisect
import time
import json
import os
from tqdm import tqdm
import timeit
import random
import yaml

np.seterr(divide='ignore', invalid='ignore')
print("Start")

with open('slimconfig.yml', 'r') as f:
    config = yaml.safe_load(f)


def region_cuts(region):
    if region == "KEE": mll_cut = (batch['BToKEE_mll_fullfit']>1.05) & (batch['BToKEE_mll_fullfit']<2.45) 
    if region == "JPSI": mll_cut = (batch['BToKEE_mll_fullfit']>2.95) & (batch['BToKEE_mll_fullfit']<3.2) 
    if region == "PSI2S": mll_cut = (batch['BToKEE_mll_fullfit']>3.55) & (batch['BToKEE_mll_fullfit']<3.8) 
    return mll_cut

def Branch_extendor(branch_imports):
    branches = []
    for branch_list in branch_imports:
        branches.extend(branch_imports[branch_list])
    return branches
    
def find_key_for_value(run, ls, data):
    for key, val in data.items():
        if str(run) in val:
            for range_ in val[str(run)]:
                if range_[0] <= ls <= range_[1]:
                    return key
    return None

def make_trigger_excl(array,trigger):
    mask = np.array(array) == trigger
    return mask
v_find_key_for_value = np.vectorize(find_key_for_value)

def load_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def output_booker(region,data_type):
    name_output = "./output/slimmed/slimmed_"+region+"_"+data_type+".root"
    file_output = uproot.recreate(name_output)
    file_output.mktree("tree",{
        "BToKEE_fit_mass":("float",()),
        "Noah_BDT_baseline":("float",()),
        'L1_4p5_HLT_4p0':("int",()),
        'L1_5p0_HLT_4p0':("int",()),
        'L1_5p5_HLT_4p0':("int",()),
        'L1_5p5_HLT_6p0':("int",()),
        'L1_6p0_HLT_4p0':("int",()),
        'L1_6p5_HLT_4p5':("int",()),
        'L1_7p0_HLT_5p0':("int",()),
        'L1_7p5_HLT_5p0':("int",()),
        'L1_8p0_HLT_5p0':("int",()),
        'L1_8p5_HLT_5p0':("int",()),
        'L1_8p5_HLT_5p5':("int",()),
        'L1_9p0_HLT_6p0':("int",()),
        'L1_10p5_HLT_5p0':("int",()),
        'L1_10p5_HLT_6p5':("int",()),
        'L1_11p0_HLT_6p5':("int",()),
        'trigger_OR':("int",()),    
        'trig_wgt':("float",()),
    })
    return file_output

def output_filer(entries,outputfile):
    for start in range(0, len(batch["trigger_OR"]),entries):
        outputfile["tree"].extend({
            "BToKEE_fit_mass": batch["BToKEE_fit_mass",cuts][start: start + entries],
            "Noah_BDT_baseline": batch["Noah_BDT_baseline",cuts][start: start + entries],
            'L1_4p5_HLT_4p0': batch["L1_4p5_HLT_4p0",cuts][start: start + entries],
            'L1_5p0_HLT_4p0': batch["L1_5p0_HLT_4p0",cuts][start: start + entries],
            'L1_5p5_HLT_4p0': batch["L1_5p5_HLT_4p0",cuts][start: start + entries],
            'L1_5p5_HLT_6p0': batch["L1_5p5_HLT_6p0",cuts][start: start + entries],
            'L1_6p0_HLT_4p0': batch["L1_6p0_HLT_4p0",cuts][start: start + entries],
            'L1_6p5_HLT_4p5': batch["L1_6p5_HLT_4p5",cuts][start: start + entries],
            'L1_7p0_HLT_5p0': batch["L1_7p0_HLT_5p0",cuts][start: start + entries],
            'L1_7p5_HLT_5p0': batch["L1_7p5_HLT_5p0",cuts][start: start + entries],
            'L1_8p0_HLT_5p0': batch["L1_8p0_HLT_5p0",cuts][start: start + entries],
            'L1_8p5_HLT_5p0': batch["L1_8p5_HLT_5p0",cuts][start: start + entries],
            'L1_8p5_HLT_5p5': batch["L1_8p5_HLT_5p5",cuts][start: start + entries],
            'L1_9p0_HLT_6p0': batch["L1_9p0_HLT_6p0",cuts][start: start + entries],
            'L1_10p5_HLT_5p0': batch["L1_10p5_HLT_5p0",cuts][start: start + entries],
            'L1_10p5_HLT_6p5': batch["L1_10p5_HLT_6p5",cuts][start: start + entries],
            'L1_11p0_HLT_6p5': batch["L1_11p0_HLT_6p5",cuts][start: start + entries],
            'trigger_OR': batch["trigger_OR",cuts][start: start + entries],
            'trig_wgt': batch["trig_wgt",cuts][start: start + entries],
        })   

data = load_json('/vols/cms/jo3717/FINAL2022PRELIMINARYANALYSIS_01_2024/PrelimRKanalysis/merged.json')


print("Start")
starttime = time.time()

for key, value in config["File"].items():
    for region, region_file in value.items():
        print(region)
        branches = Branch_extendor(config["Import_Branches"])
        Output_file = output_booker(region,key)
        data_type = key
        print(region_file)
        if region_file == None: continue
        name_input = region_file+":Events"
        with uproot.open(name_input,
                        file_handler=uproot.MultithreadedFileSource,
                        num_workers=100) as file_input:
            print(file_input.num_entries)
            pbar = tqdm(total=file_input.num_entries)
            print(file_input.keys())
            for batch, report in file_input.iterate(step_size="100 Mb",
                                            library="ak",
                                            filter_name=branches,
                                            report=True):
                #print(report.tree_entry_stop)
                pbar.update(report.tree_entry_stop-report.tree_entry_start)
                mll_cut = region_cuts(region)
                cuts_noah =( 
                            (batch['Noah_BDT_baseline'] > config["Cuts"]["BDT_CUT"])
                            &(batch["BToKEE_D0_mass_LepToPi_KToK"]>2.)&(batch["BToKEE_D0_mass_LepToK_KToPi"]>2.)
                            &(batch["BToKEE_fit_mass"]>4.7)&(batch["BToKEE_fit_mass"]<5.7)
                            &mll_cut
                )
                cuts = cuts_noah
                start = timeit.default_timer()
                biglist = v_find_key_for_value(batch['run'],batch['luminosityBlock'],data)
                if data_type == "DATA":
                    batch["L1_4p5_HLT_4p0"] = make_trigger_excl(biglist,"L1_4p5_HLT_4p0")
                    batch["L1_5p0_HLT_4p0"] = make_trigger_excl(biglist,"L1_5p0_HLT_4p0")
                    batch["L1_5p5_HLT_4p0"] = make_trigger_excl(biglist,"L1_5p5_HLT_4p0")
                    batch["L1_5p5_HLT_6p0"] = make_trigger_excl(biglist,"L1_5p5_HLT_6p0")
                    batch["L1_6p0_HLT_4p0"] = make_trigger_excl(biglist,"L1_6p0_HLT_4p0")
                    batch["L1_6p5_HLT_4p5"] = make_trigger_excl(biglist,"L1_6p5_HLT_4p5")
                    batch["L1_7p0_HLT_5p0"] = make_trigger_excl(biglist,"L1_7p0_HLT_5p0")
                    batch["L1_7p5_HLT_5p0"] = make_trigger_excl(biglist,"L1_7p5_HLT_5p0")
                    batch["L1_8p0_HLT_5p0"] = make_trigger_excl(biglist,"L1_8p0_HLT_5p0")
                    batch["L1_8p5_HLT_5p0"] = make_trigger_excl(biglist,"L1_8p5_HLT_5p0")
                    batch["L1_8p5_HLT_5p5"] = make_trigger_excl(biglist,"L1_8p5_HLT_5p5")
                    batch["L1_9p0_HLT_6p0"] = make_trigger_excl(biglist,"L1_9p0_HLT_6p0")
                    batch["L1_10p5_HLT_5p0"] = make_trigger_excl(biglist,"L1_10p5_HLT_5p0")
                    batch["L1_10p5_HLT_6p5"] = make_trigger_excl(biglist,"L1_10p5_HLT_6p5")
                    batch["L1_11p0_HLT_6p5"] = make_trigger_excl(biglist,"L1_11p0_HLT_6p5")
                    
                    batch['trigger_OR'] = (batch['L1_4p5_HLT_4p0'] | batch['L1_5p0_HLT_4p0'] | 
                    batch['L1_5p5_HLT_4p0'] | batch['L1_5p5_HLT_6p0'] | batch['L1_6p0_HLT_4p0'] | 
                    batch['L1_6p5_HLT_4p5'] | batch['L1_7p0_HLT_5p0'] | batch['L1_7p5_HLT_5p0'] | 
                    batch['L1_8p0_HLT_5p0'] | batch['L1_8p5_HLT_5p0'] | batch['L1_8p5_HLT_5p5'] | 
                    batch['L1_9p0_HLT_6p0'] | batch['L1_10p5_HLT_5p0'] | batch['L1_10p5_HLT_6p5'] | 
                    batch['L1_11p0_HLT_6p5'])
                    
                    batch['trig_wgt'] = 1.0
                    
                elif data_type == "MC":
                    batch["L1_4p5_HLT_4p0"] = batch["L1_DoubleEG4p5_er1p2_dR_Max0p9"]*batch["HLT_DoubleEle4_eta1p22_mMax6"]
                    batch["L1_5p0_HLT_4p0"] = batch["L1_DoubleEG5_er1p2_dR_Max0p9"]*batch["HLT_DoubleEle4_eta1p22_mMax6"]
                    batch["L1_5p5_HLT_4p0"] = batch["L1_DoubleEG5p5_er1p2_dR_Max0p8"]*batch["HLT_DoubleEle4_eta1p22_mMax6"]
                    batch["L1_5p5_HLT_6p0"] = batch["L1_DoubleEG5p5_er1p2_dR_Max0p8"]*batch["HLT_DoubleEle6_eta1p22_mMax6"]
                    batch["L1_6p0_HLT_4p0"] = batch["L1_DoubleEG6_er1p2_dR_Max0p8"]*batch["HLT_DoubleEle4_eta1p22_mMax6"]
                    batch["L1_6p5_HLT_4p5"] = batch["L1_DoubleEG6p5_er1p2_dR_Max0p8"]*batch["HLT_DoubleEle4p5_eta1p22_mMax6"]
                    batch["L1_7p0_HLT_5p0"] = batch["L1_DoubleEG7_er1p2_dR_Max0p8"]*batch["HLT_DoubleEle5_eta1p22_mMax6"]
                    batch["L1_7p5_HLT_5p0"] = batch["L1_DoubleEG7p5_er1p2_dR_Max0p7"]*batch["HLT_DoubleEle5_eta1p22_mMax6"]
                    batch["L1_8p0_HLT_5p0"] = batch["L1_DoubleEG8_er1p2_dR_Max0p7"]*batch["HLT_DoubleEle5_eta1p22_mMax6"]
                    batch["L1_8p5_HLT_5p0"] = batch["L1_DoubleEG8p5_er1p2_dR_Max0p7"]*batch["HLT_DoubleEle5_eta1p22_mMax6"]
                    batch["L1_8p5_HLT_5p5"] = batch["L1_DoubleEG8p5_er1p2_dR_Max0p7"]*batch["HLT_DoubleEle5p5_eta1p22_mMax6"]
                    batch["L1_9p0_HLT_6p0"] = batch["L1_DoubleEG9_er1p2_dR_Max0p7"]*batch["HLT_DoubleEle6_eta1p22_mMax6"]
                    batch["L1_10p5_HLT_5p0"] = batch["L1_DoubleEG10p5_er1p2_dR_Max0p6"]*batch["HLT_DoubleEle5_eta1p22_mMax6"]
                    batch["L1_10p5_HLT_6p5"] = batch["L1_DoubleEG10p5_er1p2_dR_Max0p6"]*batch["HLT_DoubleEle6p5_eta1p22_mMax6"]
                    batch["L1_11p0_HLT_6p5"] = batch["L1_DoubleEG11_er1p2_dR_Max0p6"]*batch["HLT_DoubleEle6p5_eta1p22_mMax6"]
                    

                    batch['trigger_OR'] = (batch['L1_4p5_HLT_4p0'] | batch['L1_5p0_HLT_4p0'] | 
                                           batch["L1_5p5_HLT_4p0"] | batch['L1_5p5_HLT_6p0'] | 
                                           batch['L1_6p0_HLT_4p0'] | batch['L1_6p5_HLT_4p5'] | 
                                           batch['L1_7p0_HLT_5p0'] | batch['L1_7p5_HLT_5p0'] | 
                                           batch['L1_8p0_HLT_5p0'] | batch['L1_8p5_HLT_5p0'] | 
                                           batch['L1_8p5_HLT_5p5'] | batch['L1_9p0_HLT_6p0'] | 
                                           batch['L1_10p5_HLT_5p0'] | batch['L1_10p5_HLT_6p5'] | 
                                           batch['L1_11p0_HLT_6p5'])
                         
                entries = 4000000
                
                output_filer(entries,Output_file)
                
                end = timeit.default_timer()
             
                
                
                
            pbar.close()