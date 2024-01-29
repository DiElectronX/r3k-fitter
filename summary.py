import json
import math
import numpy as np

################################################################################
# Methods ...

#for value in ['signal_num','cb_mean','cb_sigma','cb_alphaL','cb_alphaR','cb_nL','cb_nR',]:

def val_err(dct,string):
    return (
        dct[string][0], # central value
        dct[string][0]-dct[string][1] # central value - low value = error
        )

def print_signal_header():
    print(
        "Trigger"+
        "                    Counts"+
        "            Mass"+
        "           Width"+
        "     cb_alphaL"+
        "            cb_nL"+
        "     cb_alphaR"+
        "            cb_nR"
        )

def print_signal_row(dct,trigger,lumi,isMC='mc',region='JPSI',blind=False):
    sub_dct = dct[isMC][region][trigger]
    print(f'{trigger:16s}, ',end='')
    val,err = val_err(sub_dct,"signal_num")
    count = (val,err)
    #if isMC=='mc': count = expectation(val,lumi,err,region)
    if blind==False: print(f'{val:7.1f}+/-{err:5.1f} , ',end='')
    else: print('        blinded , ',end='')
    val,err = val_err(sub_dct,"cb_mean")
    print(f'{val:5.3f}+/-{err:5.3f} , ',end='')
    val,err = val_err(sub_dct,"cb_sigma")
    print(f'{val:5.3f}+/-{err:5.3f} , ',end='')
    val,err = val_err(sub_dct,"cb_alphaL")
    print(f'{val:4.2f}+/-{err:4.2f} , ',end='')
    val,err = val_err(sub_dct,"cb_nL")
    print(f'{val:6.2f}+/-{err:5.2f} , ',end='')
    val,err = val_err(sub_dct,"cb_alphaR")
    print(f'{val:4.2f}+/-{err:4.2f} , ',end='')
    val,err = val_err(sub_dct,"cb_nR")
    print(f'{val:6.2f}+/-{err:5.2f} , ',end='')
    print()
    return count

def print_signal_table(dct,triggers,isMC="mc",region="JPSI",blind=False):
    print("Table of signal parameter values for","isMC=",isMC,"region=",region)
    print_signal_header()
    counts = []
    for trigger,lumi in triggers:
        count,err = print_signal_row(dct,trigger,lumi,isMC=isMC,region=region,blind=blind)
        counts.append((count,err))
    print()
    return counts

def print_bkgd_header():
    print(
        "Trigger"+
        "                  comb_num"+
        "      exp_slope"
        "           part_num"+
        "     expo_slope"+
        #"    expo_offset"+
        "      erfc_mean"+
        "      erfc_sigma"
        )

def print_bkgd_row(dct,trigger,isMC='data',region='JPSI',partial=True):
    sub_dct = dct[isMC][region][trigger]
    print(f'{trigger:16s}, ',end='')
    val,err = val_err(sub_dct,"comb_num")
    print(f'{val:7.1f}+/-{err:5.1f} , ',end='')
    val,err = val_err(sub_dct,"exp_slope")
    print(f'{val:5.2f}+/-{err:4.2f} , ',end='')
    if partial==True:
        val,err = val_err(sub_dct,"part_num")
        print(f'{val:7.1f}+/-{err:6.1f} , ',end='')
        val,err = val_err(sub_dct,"expo_slope")
        #print(f'{val:6.1f}+/-{err:5.1f} , ',end='')
        #val,err = val_err(sub_dct,"expo_offset")
        print(f'{val:5.2f}+/-{err:4.2f} , ',end='')
        val,err = val_err(sub_dct,"erfc_mean")
        print(f'{val:5.2f}+/-{err:4.2f} , ',end='')
        val,err = val_err(sub_dct,"erfc_sigma")
        print(f'{val:5.3f}+/-{err:4.3f} , ',end='')
    print()
                
def print_bkgd_table(dct,triggers,region="JPSI",partial=True):
    isMC='data'
    print("Table of background parameter values for","isMC=",isMC,"region=",region)
    print_bkgd_header()
    for trigger,lumi in triggers:
        print_bkgd_row(dct,trigger,isMC=isMC,region=region,partial=partial)
    print()

def print_comparison_header():
    print(
        "Trigger"+
        "      Lint [fb]"+
        "   AxE [1e-4]"
        "      Exp. counts"+
        "      Obs. counts"+
        "         Ratio"
        )

def ntoys(): return 500.e6

def expectation(val,lumi,err=None,region="JPSI"):
    bf = {
        "JPSI":0.001*0.06,
        "PSI2S":6.2e-4*7.9e-3,
        "KEE":4.5e-7,
    }.get(region)
    eff     = val / ntoys()
    eff_err = err / ntoys()
    if region=="PSI2S":
        eff     = val / 50.e6
        eff_err = err / 50.e6    
    exp = lumi * 4.7e11 * 0.4 * bf * eff * 2
    exp_err = exp * (eff_err/eff)
    return (exp,exp_err)


def print_comparison_table(dct,triggers,region="JPSI",blind=False):
    print("Comparison for observed (data) and expected (MC) in region",region)
    print_comparison_header()
    ratios = []
    for trigger,lumi in triggers:
        data = dct["data"][region][trigger]
        mc   = dct["mc"][region][trigger]
        print(f"{trigger:16s}, ",end="")
        print(f"{lumi:4.2f}, ",end="")
        val,err = val_err(mc,"signal_num")
        eff     = val / ntoys()
        eff_err = err / ntoys()
        print(f"{eff/1.e-4:4.2f}+/-{eff_err/1.e-4:4.2f}, " , end="")
        exp,exp_err = expectation(val,lumi,err,region)
        print(f"{exp:7.1f}+/-{exp_err:5.1f}, ",end="")
        obs,obs_err = val_err(data,"signal_num")
        if blind==False: print(f'{obs:7.1f}+/-{obs_err:5.1f} ',end='')
        else: print('        blinded ',end='')
        ratio = obs/exp if exp > 0. else 0.
        ratio_err = math.sqrt(exp)/exp * ratio if exp > 0. else 0.
        if blind==False: print(f'{ratio:5.2f}+/-{ratio_err:5.2f} ',end='')
        else: print('      blinded ',end='')
        print()
        ratios.append((ratio,ratio_err))
    return ratios

def summary(filename,triggers=None,blind=True) :

    print("Parsing json file ...")

    # Open file and parse json
    dct = {}
    try:
      with open(filename,'r') as f:
        try:
          dct = json.load(f)
        except json.decoder.JSONDecodeError:
          print("Problem parsing json contained in file:",filename)
    except FileNotFoundError:
      print("Problem opening file:",filename)

    # Check if MC content is there
    if "mc" not in dct:
        print("Incorrect json format...")
        return

    # Tables ...
    _partial=True
    mc_JPSI = print_signal_table(dct,triggers,isMC="mc",region="JPSI")
    data_JPSI = print_signal_table(dct,triggers,isMC="data",region="JPSI")
    print_bkgd_table(dct,triggers,region="JPSI",partial=_partial)
    mc_PSI2S = print_signal_table(dct,triggers,isMC="mc",region="PSI2S")
    data_PSI2S = print_signal_table(dct,triggers,isMC="data",region="PSI2S")
    print_bkgd_table(dct,triggers,region="PSI2S",partial=_partial)
    mc_KEE = print_signal_table(dct,triggers,isMC="mc",region="KEE")
    data_KEE = print_signal_table(dct,triggers,isMC="data",region="KEE",blind=blind)
    print_bkgd_table(dct,triggers,region="KEE",partial=_partial)

    # Comparison
    ratios_JPSI = print_comparison_table(dct,triggers,region="JPSI")
    ratios_PSI2S = print_comparison_table(dct,triggers,region="PSI2S")
    ratios_KEE = print_comparison_table(dct,triggers,region="KEE",blind=blind)

    # Compare 
    print()
    print("Obs and exp: JPSI and PSI2S")
    for (trg,lumi),d_JPSI,m_JPSI,d_PSI2S,m_PSI2S in zip(triggers,data_JPSI,mc_JPSI,data_PSI2S,mc_PSI2S):
        m_JPSI = expectation(m_JPSI[0],lumi,m_JPSI[1],"JPSI")
        m_PSI2S = expectation(m_PSI2S[0],lumi,m_PSI2S[1],"PSI2S")
        print(
            f'Trigger: {trg:16s}',
            f'  (J/psi) Exp: {m_JPSI[0]:7.1f}, Obs: {d_JPSI[0]:7.1f}',
            f'  (PSI2S) Exp: {m_PSI2S[0]:6.1f}, Obs: {d_PSI2S[0]:6.1f}',
            )

    # Double ratio
    print()
    print("Double ratio: [obs/exp]_PSI2S / [obs/exp]_JPSI")
    for trg,JPSI,PSI2S in zip(triggers,ratios_JPSI,ratios_PSI2S):
        ratio = PSI2S[0]/JPSI[0] if JPSI[0]>0. else 0.
        ratio_err  = (JPSI[1]/ JPSI[0])**2. if  JPSI[0]>0. else 0.
        ratio_err += (PSI2S[1]/PSI2S[0])**2. if PSI2S[0]>0. else 0.
        ratio_err = ratio * np.sqrt(ratio_err)
        #print(trg,ratio,ratio_err)
        print(f'Trigger: {trg[0]:16s}, Lumi: {trg[1]:4.2f}, Ratio: {ratio:4.2f} +/- {ratio_err:4.2f}')

    # Ratio of AxE
    print()
    print("AxE [x1E-4]")
    for trg,m_JPSI,m_PSI2S,m_KEE in zip(triggers,mc_JPSI,mc_PSI2S,mc_KEE):
        print(
            f'Trigger: {trg[0]:16s}',
            f'  (AxE) J/psi: {m_JPSI[0]*1.e4/ntoys():4.2f}',
            f'PSI2S: {m_PSI2S[0]*1.e4/ntoys():4.2f}',
            f'KEE: {m_KEE[0]*1.e4/ntoys():5.3f}',
            f'  (Ratios) KEE/JPSI: {m_KEE[0]/m_JPSI[0] if m_JPSI[0]>0. else 0.:5.3f}',
            f'PSI2S/JPSI: {m_PSI2S[0]/m_JPSI[0] if m_JPSI[0]>0. else 0.:5.3f}',
            )

    # Predict PSI2S
    print()
    print("Predict @ Psi(2S)")
    for (trg,lumi),r_JPSI,m_PSI2S,d_PSI2S in zip(triggers,ratios_JPSI,mc_PSI2S,data_PSI2S):
        m_PSI2S = expectation(m_PSI2S[0],lumi,m_PSI2S[1],"PSI2S")
        print(
            f'Trigger: {trg:16s}',
            f'  Obs/Exp @ JPSI: {r_JPSI[0]:4.2f}',
            f'  Exp @ PSI2S: {m_PSI2S[0]:5.1f}',
            f'  Pred @ PSI2S: {m_PSI2S[0]*r_JPSI[0]:5.1f}',
            f'  Number per fb :{m_PSI2S[0]*r_JPSI[0]/lumi:4.2f}',
            f'  Obs @ PSI2S: {d_PSI2S[0]:5.1f}',
            )

    # Predict KEE (blinded)
    print()
    print("Predict @ low q2")
    for (trg,lumi),r_JPSI,m_KEE,d_KEE in zip(triggers,ratios_JPSI,mc_KEE,data_KEE):
        m_KEE = expectation(m_KEE[0],lumi,m_KEE[1],"KEE")
        print(
            f'Trigger: {trg:16s}',
            f'  Lint :{lumi:4.2f}',
            f'  Obs/Exp @ JPSI: {r_JPSI[0]:4.2f}',
            f'  Exp @ KEE: {m_KEE[0]:5.1f}',
            f'  Pred @ KEE: {m_KEE[0]*r_JPSI[0]:5.1f}',
            f'  Number per fb :{m_KEE[0]*r_JPSI[0]/lumi:4.2f}',

            '  Obs @ KEE:',
            f'{d_KEE[0]:5.1f}' if blind==False else 'blinded',
            )
    
################################################################################
# Main ...

if __name__ == "__main__":

    # Production tag
    
    filename = 'output/params/parameters.json'
    triggers = [
        #("trigger_none",7.36),
        ("trigger_OR",33.8),
        ("L1_11p0_HLT_6p5", 1.577),
        ("L1_10p5_HLT_6p5", 1.136),
        ("L1_10p5_HLT_5p0", 0.103),
        ("L1_9p0_HLT_6p0",  8.844),
        ("L1_8p5_HLT_5p5",  3.339),
        ("L1_8p5_HLT_5p0",  0.675),
        ("L1_8p0_HLT_5p0",  6.890),
        ("L1_7p5_HLT_5p0",  1.635),
        ("L1_7p0_HLT_5p0",  2.662),
        ("L1_6p5_HLT_4p5",  3.611),
        ("L1_6p0_HLT_4p0",  2.511),
        ("L1_5p5_HLT_6p0",  0.150),
        ("L1_5p5_HLT_4p0",  0.650),
        ("L1_5p0_HLT_4p0",  0.041),
        ("L1_4p5_HLT_4p0",  0.030)
    ]
    
#    triggers = [
#        #("trigger_none",7.36),
#        ("trigger_OR",22.60827538),
#        ("L1_11p0_HLT_6p5", 2.159205064),
#        ("L1_10p5_HLT_6p5",1.804763109),
#        ("L1_9p0_HLT_6p0",  1.959824817),
#        ("L1_8p5_HLT_5p5",  0.972094098),
#        ("L1_8p0_HLT_5p0",  1.884784617 ),
#        ("L1_7p5_HLT_5p0",  2.267806566),
#        ("L1_7p0_HLT_5p0", 2.886079815),
#        ("L1_6p5_HLT_4p5",  5.011907215),
#        ("L1_6p0_HLT_4p0",  1.720140186),
#        ("L1_5p5_HLT_4p0",  1.859816871),
#        ("L1_5p0_HLT_4p0",  0.081853027),
#    ]
    summary(filename,triggers=triggers,blind=True)