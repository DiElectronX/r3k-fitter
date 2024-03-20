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
## Initial RooFit Fitting Code

Use following script to perform simplified fits in RooFit 
```
python do_fit.py -c <config> -m <mode> [-v] [-lc]
```
- `-c` : Configuration file for handling inputs, outputs, and fit parameters. Default argument is `fit_cfg.yml`
- `-m` : Mode argument is currently set up for unblinded fits in the jpsi and psi2s control regions. The only accepted values are `jpsi`, `psi2s`, or `all` for processing the full list.
- `-v` : Optional flag for verbosity. Use if output in the terminal from Root and RooFit is desired
- `-lc` : Optional flag directs the script to load cached fit templates from step 1 & 2 of the script, skips to final fit
- All other parameters are stored and easily editable in the `fit_cfg.yml` configuration file

## Running Fits Through Combine

### Workflow Script

To run all the following steps in a single go, the following script is included for ease-of-use. Make sure that the given datacard is correctly linked to the relevant RooWorkspace and PDF objects from the RooFit output.

```
./runCombineFit.sh <.txt datacard file> <label>
```

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

### Plotting Results

Use following script to show results for combine ML fit to data. All PDF shapes are saved in `fitDiagnostics` file.
```
combine -M FitDiagnostics --plots --saveShapes --signalPdfNames='*sig*' --backgroundPdfNames='*comb_bkg*,*part_bkg*' <combine MultiDimFit fit output>
```
