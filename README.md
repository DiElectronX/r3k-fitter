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

## Plotting

Use following script to show results pre- and post- combine ML fit to data
```
python plotBeforeAndAfterMLFit.py <combine MultiDimFit fit output>
```
note 1: may have to edit bin/channel name in script
note 2: must `--saveWorkspace` to save objects for this plot script
