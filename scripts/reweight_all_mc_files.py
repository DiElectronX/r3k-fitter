import subprocess
from pathlib import Path

# === CONSTANT arguments ===
common_args = [
    "-c", "../new_trigger_cfg.yml",
    "-o", "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/reweighted/",
    "-t", "mytree",
    "-w", "trigger_sf_value",
    "-mc",
]

upsample_rate= 1.5
downsample_rate= .5

# === RUN VARIANTS ===
runs = [
        # jpsi
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_kaon.root", "-v": 2741., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_kaon.root", "-v": 2741.*upsample_rate, "-l": "reweighted_upsampled"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_kaon.root", "-v": 2741.*downsample_rate, "-l": "reweighted_downsampled"},
        
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_pion.root", "-v": 727., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_pion.root", "-v": 727.*upsample_rate, "-l": "reweighted_upsampled"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_jpsi_pion.root", "-v": 727.*downsample_rate, "-l": "reweighted_downsampled"},

        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 10328., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 10328.*upsample_rate, "-l": "reweighted_upsampled"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 10328.*downsample_rate, "-l": "reweighted_downsampled"},
        
        # using yield ratio from kstar (kstar_jpsi_kaon / kstar_jpsi_pion * k0star_jpsi_pion)
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 5478., "-l": "reweighted_fixed"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 5478.*upsample_rate, "-l": "reweighted_fixed_upsampled"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_kaon.root", "-v": 5478.*downsample_rate, "-l": "reweighted_fixed_downsampled"},

        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_pion.root", "-v": 1453., "-l": "reweighted"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_pion.root", "-v": 1453.*upsample_rate, "-l": "reweighted_upsampled"},
        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_jpsi_pion.root", "-v": 1453.*downsample_rate, "-l": "reweighted_downsampled"},

        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_chic1_jpsi_kaon.root", "-v": 1153., "-l": "reweighted"},

        {"-m": "jpsi", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_jpsipi_jpsi_pion.root", "-v": 2950., "-l": "reweighted"},
        
        #psi2s
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_psi2s_pion.root", "-v": 25., "-l": "reweighted"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_psi2s_kaon.root", "-v": 10., "-l": "reweighted"},
        # account for k0star_kaon + kstar_kaon in k0star_kaon sample (k0star_kaon + (k0star_kaon / k0star_pion * kstar_pion))
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_psi2s_kaon.root", "-v": 135., "-l": "reweighted_fixed"},
        {"-m": "psi2s", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_psi2s_pion.root", "-v": 2., "-l": "reweighted"},

        # lowq2
        {"-m": "lowq2", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_kstar_pion.root", "-v": 1., "-l": "reweighted"},
        {"-m": "lowq2", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_kaon.root", "-v": 8., "-l": "reweighted"},
        # account for k0star_kaon + kstar_kaon in k0star_kaon sample (k0star_kaon + (k0star_kaon / k0star_pion * kstar_pion))
        {"-m": "lowq2", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_kaon.root", "-v": 12., "-l": "reweighted_fixed"},
        {"-m": "lowq2", "-i": "/eos/cms/store/group/phys_bphys/DiElectronX/File_location_20_11_24/NewMethod_29_08_25/newmethod_trigger_sf_values_bdt/measurement_k0star_pion.root", "-v": 2., "-l": "reweighted"},
]

# === Loop over each configuration and run the script ===
for run_args in runs:
    cmd = ["python3", "data_sampler.py"] + common_args
    for k, v in run_args.items():
        cmd.extend([k, str(v)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
