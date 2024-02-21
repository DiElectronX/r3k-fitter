# Fit Code for CMS Run 3 RK Analysis

## Basic Combine templates for working ML fit code

### Setting Up
Based on recommendations from Combine documentation, use slc7 Singularity/Apptainer along with CMSSW_11_3_X and Combine v9

[note: to use additional fit shapes, we merge changes from an open PR to HiggsCombine, this will be deprecated when the PR is merged into a recommended release]
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

git clone -b combine_template git@github.com:DiElectronX/r3k-fitter.git
cd $CMSSW_BASE/src/r3k-fitter
```

### Steps for Simple ML Fits

(same steps can be followed for "_binned" version)

1. Generate Fit to data with Roofit & save PDFs + RooDataSet to RooWorkspace in ROOT file
```
python simple_fit.py
```
2. Using Combine datacard with shapes linked to RooWorkspace objects, convert `.txt` datacard into Combine Workspace
```
text2workspace.py datacard_shapes_simple.txt
```
3. Run Combine maximum likelihood fit with generated workspace datacard
```
combine -M MultiDimFit datacard_shapes_simple.root --saveWorkspace
```

### Plotting

Use following script to show results pre- and post- combine ML fit to data
```
python plotBeforeAndAfterMLFit.py <combine ML fit output>
```
note: must `--saveWorkspace` to save objects for this plot script
