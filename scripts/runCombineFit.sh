#!/usr/bin/bash
datacard=$1
label=$2

if [[ -z "$label" ]]; then
    label="test"
fi

text2workspace.py $datacard
combine -M MultiDimFit --saveWorkspace -n ".${label}" ${datacard/txt/root}
combine -M FitDiagnostics "higgsCombine.${label}.MultiDimFit.mH120.root" -n ".${label}" --plots --saveShapes --signalPdfNames='*sig*' --backgroundPdfNames='*comb_bkg*,*part_bkg*' 

mv "higgsCombine.${label}.MultiDimFit.mH120.root" ./combine_outputs
mv "higgsCombine.${label}.FitDiagnostics.mH120.root" ./combine_outputs
mv "fitDiagnostics.${label}.root" ./combine_outputs
