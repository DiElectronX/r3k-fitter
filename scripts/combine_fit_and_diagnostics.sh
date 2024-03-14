#!/usr/bin/bash
datacard=$1
label=$2
combine -M MultiDimFit $datacard --saveWorkspace -n ".${label}"
combine -M FitDiagnostics --plots --saveShapes --signalPdfNames='*sig*' --backgroundPdfNames='*comb_bkg*,*part_bkg*' "higgsCombine.${label}.MultiDimFit.mH120.root"
mv fitDiagnosticsTest.root "fitDiagnostics_${label}.root"
