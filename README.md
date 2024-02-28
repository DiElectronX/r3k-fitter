# Fit Code for CMS Run 3 RK Analysis

## Setting Up
Based on recommendations from Combine documentation, use slc7 Singularity/Apptainer along with CMSSW_11_3_X and Combine v9

note: to use additional fit shapes, we merge changes from an open PR to HiggsCombine, this will be deprecated when the PR is merged into a recommended release

### Building Environment (First-Time Setup)
```
cmssw-el7
cmsrel CMSSW_11_3_4
cd CMSSW_11_3_4/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit

cd $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v9.0.0

git fetch origin pull/843/head
git checkout -b r3k_modified FETCH_HEAD

cd $CMSSW_BASE/src/
bash <(curl -s https://raw.githubusercontent.com/cms-analysis/CombineHarvester/main/CombineTools/scripts/sparse-checkout-https.sh)

scramv1 b clean; scramv1 b
cmsenv

git clone git@github.com:DiElectronX/r3k-fitter.git
cd $CMSSW_BASE/src/r3k-fitter
```

### Loading Environment (When Logging In)
```
cd .../r3k-fitter
cmssw-el7
cmsenv
```
## Initial Fitting Code

Use following script to perform simplified fits in RooFit 
```
python do_plot.py -m <mode> [-v] [-lc]
```
The mode flag is currently configured for unblinded fits in the jpsi and psi2s regions, so the only accepted arguments are `jpsi` or `psi2s`. The optional `-v` flag handles verbosity if output in the terminal from Root and RooFit is desired. The optional `-lc` flag directs the script to load fit templates from step 1 & 2 of the script directly from cached files, jumping directly to the final step 3 fit if such templates exist. All other parameters are stored in the `fit_cfg.yml` configuration file.

## Running Initial Fits Through Combine

### Generate .txt Datacard
No real instruction here yet, just ensure that you are loading in the specified shapes from the previous step's RooWorkspace. `datacard_psi2s_simple.txt` is a functional template.

### Process .root Datacard
```
text2workspace.py datacard.txt
```

### Run Combine MultiDimFit
```
combine -M MultiDimFit datacard.root --saveWorkspace -n <fit output file label>
```

## Plotting

Use following script to show results for combine ML fit to data
```
combine -M FitDiagnostics --plots --signalPdfNames='*sig*' --backgroundPdfNames='*bkg*' <combine MultiDimFit fit output>
```
note: must have used `--saveWorkspace` flag in previous step to save objects for this script
