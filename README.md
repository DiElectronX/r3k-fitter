# Fit Code for CMS Run 3 RK Analysis

This repository houses the core fitting code for the Run 3 RK analysis. Also contains various other scripts for scanning BDT cut significant, sanity checks, and systematic uncertainty estimations.

## Setting Up

The new working point for this repo uses an updated Combine release and therefore a updated corresponding CMSSW version. Here are steps to install the necessary tools for running the core fitting procedures.

### CMSSW Set-Up (Simplest Version)

These instructions work on LXPLUS, but have not been tested on other CMS networks (FNAL LPC, etc.). This environment is easiest to set up as it leverages the included packages available for CMSSW software releases, the Combine integration via CMSSW, and easy access to RK samples on the network.

#### Initial Set-Up

```
cmsrel CMSSW_14_1_0
cd CMSSW_14_1_0_pre4/src
cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v10.4.1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b -j$(nproc --ignore=2)

cd $CMSSW_BASE/src
git clone git@github.com:DiElectronX/r3k-fitter.git
cd r3k-fitter
```

#### Loading Environment
```
cd .../r3k-fitter
cmsenv
```

### Standalone Set-Up

This set-up is slightly more involved due to the fragility of Conda installs with ROOT (you may have to do some Conda troubleshooting based on your particular computer) and the manual compilation of Combine, but offers the user the ability to avoid unnecessary CMSSW installations and work on a local machine such as one's laptop.

#### Initial Set-Up
```
git clone git@github.com:DiElectronX/r3k-fitter.git
cd r3k-fitter

# Make sure conda is updated
conda activate base
conda update -n base conda
conda config --set solver libmamba

# Create r3k-fitter env from YAML file
conda env create -f environment.yml

# Try "conda env create -f environment.yml --no-channel-priority" 
# as a first troubleshooting step if initial command fails

conda activate r3k-fitter

# Clone Combine for integration into Conda environment
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit

# Configure with CMake
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCMAKE_INSTALL_PYTHONDIR=lib/python3.12/site-packages \
  -DUSE_VDT=OFF

# Build and Install
cmake --build build -j$(nproc --ignore=2)
cmake --install build

# Check that custom packages are available
python3 -c "import numpy; import ROOT; print('ROOT and Numpy loaded successfully!')"

# Check Combine is installed
combine --help
```
#### Loading Environment
```
cd .../r3k-fitter
conda activate r3k-fitter
```

---

## Core Fitting Framework

The primary fitting logic is contained within `do_fit.py`. This script handles the Maximum Likelihood Estimation (MLE) using RooFit for the J/$\psi$ control region, $\psi(2S)$ control region, and low-$q^2$ (rare) signal regions. It manages data loading, PDF construction, fitting, and plotting.

### Configuration

All input file paths, branch names, fit ranges, and initial parameter values are defined in `config/fit_cfg.yml`. You should modify this file to change datasets or fit starting points without altering the Python code.

### Running the Fitter

Run the script from the root directory of the repository:

```bash
python3 do_fit.py -c <config_file> -m <mode> [options]

```

**Arguments:**

* `-c, --config`: Path to the configuration YAML file (default: `fit_cfg.yml`).
* `-m, --mode`: The fit region to process. Options:
    * `jpsi`: Fits the  control region.
    * `psi2s`: Fits the  control region.
    * `lowq2`: Fits the rare signal region (blinded by default).
    * `all`: Runs fits for all three regions sequentially.


* `-v, --verbose`: Enables verbose output from ROOT/RooFit.
* `-lc, --loadcache`: Skips the initial template fitting steps (steps 1-4) and loads cached parameters from previous runs. Useful for tweaking the final composite fit without re-fitting MC templates.
* `-minos`: Runs the MINOS minimizer to calculate asymmetric error bars (slower but more accurate).

**Specific Mode Options:**

* `-t, --toy_fit`: **(lowq2 mode only)** Generates and fits a toy dataset (Expected Signal + Background) instead of blinded data.
* `-cf, --constrained_fit`: **(jpsi mode only)** Performs the fit using Gaussian constraints on specific background normalizations.

### Examples

**1. Standard fit of the J/psi control region:**

```bash
python3 do_fit.py -c config/example_cfg.yml -m jpsi -v

```

**2. Toy fit of the low-q2 signal region (unblinding study):**

```bash
python3 do_fit.py -c config/example_cfg.yml -m lowq2 -t

```

**3. Run all regions using a specific config file:**

```bash
python3 do_fit.py -c config/fit_cfg_12_9_24.yml -m all

```

**Outputs:**
Plots (`.pdf`), workspaces (`.root`), and fit parameter logs (`.yml`) are saved to the directory specified in the `output` section of your config file (usually `fitter_outputs/`).

---

## Significance Scans

These scripts are designed to optimize the BDT score cut by scanning through a range of CUT values, performing fits at each step, and calculating the signal significance ($S/\sqrt{S+B}$) in the extrapolated low-$q^2$ region.

**⚠️ Important:** You must navigate into the `significance_scan/` directory before running these scripts to ensure relative imports work correctly.

```bash
cd significance_scan/

```

### 1. Running the Scan

The main scanning logic is in `significance_scan_v2.py`. This script iterates over BDT cuts defined in the code, runs the fits via `do_fit.py` imports, and saves the results to a pickle file.

```bash
python3 significance_scan_v2.py -c ../config/fit_cfg.yml [-v]

```

* **Output:** Generates `significance_scan_data.pkl` containing yields, efficiencies, and significances for every step.
* **Intermediate Plots:** Individual fit plots for every step are generated in `scan_fits/`.

### 2. Plotting Results

Once the pickle file is generated, use the plotting scripts to visualize the optimization.

* **Plot Significance vs BDT Score:**
Fits a parabola to the significance curve to find the optimal working point.
```bash
python3 plot_significance_scan.py -f significance_scan_data.pkl -fit

```


* **Plot Signal/Background Yields:**
```bash
python3 plot_yield_scan.py -f significance_scan_data.pkl

```


* **Plot Efficiency Trends:**
```bash
python3 plot_efficiency_scan.py -f significance_scan_data.pkl

```



---

## Systematics & R-Ratio Calculations

Scripts for calculating systematic uncertainties, efficiency ratios, and  measurements are located in the `systematics/` directory.

**⚠️ Important:** You must navigate into the `systematics/` directory before running these scripts.

```bash
cd systematics/

```

### BDT Data/MC Discrepancy

Calculates systematics associated with Data/MC agreement by varying BDT cuts and comparing efficiency ratios.

```bash
python3 produce_bdt_data_mc_systematic.py -c ../config/fit_cfg.yml

```

* **Output:** Generates `bdt_data_mc_systematic_result.csv`.

###  Calculation

Calculates the double ratio for the  control channel as a cross-check.

```bash
python3 calculate_r_psi2s.py -c ../config/fit_cfg.yml

```

### K* Leakage Scans

Quantifies the leakage of  events into the  signal window.

```bash
python3 kstar_k_scan.py -m [jpsi|psi2s|all]
python3 plot_kstar_k_scan.py -m [jpsi|psi2s|all]

```

---


## Running Fits Through Combine

**⚠️ Note:** These Combine scripts are old and may need to be updated.

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


