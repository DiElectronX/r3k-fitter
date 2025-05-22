import subprocess
from pathlib import Path

# === CONSTANT arguments ===
common_args = [
    "-c", "fit_cfg_5_22_25.yml",
    "-o", "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_30_04_25_reweighted/",
    "-m", "central",
    "-t", "mytree",
    "-w", "trig_wgt",
    "-mc",  # flag (no value)
]

# === RUN VARIANTS ===
runs = [
        {"-m": "jpsi", "-i": "data/sample1.root", "-v": 0.1, "-l": "reweighted"},
        {"-m": "jpsi", "-i": "data/sample2.root", "-v": 0.2, "-l": "reweighted"},
        {"-m": "jpsi", "-i": "data/sample3.root", "-v": 0.3, "-l": "reweighted"},
]

# === Loop over each configuration and run the script ===
for run_args in runs:
    cmd = ["python3", "data_sampler.py"] + common_args
    for k, v in run_args.items():
        cmd.extend([k, str(v)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

