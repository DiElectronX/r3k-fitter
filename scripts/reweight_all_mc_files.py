import subprocess
from pathlib import Path

# === CONSTANT arguments ===
common_args = [
    "-c", "../fit_cfg_5_22_25.yml",
    "-o", "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_30_04_25_reweighted/",
    "-t", "mytree",
    "-w", "sf_combined_mean",
    "-mc",
]

# === RUN VARIANTS ===
runs = [
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_kstar_jpsi_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 4224., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_kstar_jpsi_pion_TrigSfs_bdt_weight_pu_weight.root", "-v": 1165., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_jpsi_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 15226., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_jpsi_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 8031.., "-l": "reweighted_fixed"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_jpsi_pion_TrigSfs_bdt_weight_pu_weight.root", "-v": 2215., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_chic1_jpsi_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 2028., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_jpsipi_jpsi_pion_TrigSfs_bdt_weight_pu_weight.root", "-v": 4214., "-l": "reweighted"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_kstar_psi2s_pion_TrigSfs_bdt_weight_pu_weight.root", "-v": 57., "-l": "reweighted"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_psi2s_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 18., "-l": "reweighted"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_psi2s_kaon_TrigSfs_bdt_weight_pu_weight.root", "-v": 205., "-l": "reweighted_fixed"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/AllSF_final_ntuples_23_05_25/measurement_k0star_psi2s_pion_TrigSfs_bdt_weight_pu_weight.root", "-v": 5., "-l": "reweighted"},
]

# === Loop over each configuration and run the script ===
for run_args in runs:
    cmd = ["python3", "data_sampler.py"] + common_args
    for k, v in run_args.items():
        cmd.extend([k, str(v)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

