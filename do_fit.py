import ROOT
import json

defaults = {
      "JPSI":{
          "cb_mean":(5.279,5.26,5.30),
          "cb_sigma":(0.057,0.04,0.10),
          "cb_alphaL":(2.,0.5,3.),
          "cb_nL":(10.,1.,100.),
          "cb_alphaR":(2.,0.5,3.),
          "cb_nR":(10.,1.,100.),
          "exp_slope":(-0.5,-100.,0.), # combinatorial
          "expo_slope":(3.8,3.0,5.0),       # (2.25,1.,10.)
          #"expo_offset":(5.13, 5.11, 5.15), # (5.13, 5.1, 5.15)
          "erfc_mean":(5.16,5.15,5.17),     # (5.13,5.1,5.15)
          "erfc_sigma":(0.06,0.05,0.20),    # (0.03,0.015,0.05)
          },
      "PSI2S":{
          "cb_mean":(5.279,5.26,5.30),
          "cb_sigma":(0.063,0.04,0.07),
          "cb_alphaL":(2.,0.5,3.),
          "cb_nL":(10.,1.,100.),
          "cb_alphaR":(2.,0.5,3.),
          "cb_nR":(10.,1.,100.),
          "exp_slope":(-0.5,-100.,0.),
          "expo_slope":(3.8,3.0,5.0),       # (2.25,1.,10.)
          #"expo_offset":(5.13, 5.11, 5.15), # (5.13, 5.1, 5.15)
          "erfc_mean":(5.16,5.15,5.17),     # (5.13,5.1,5.15)
          "erfc_sigma":(0.06,0.05,0.10),    # (0.03,0.015,0.05)
          },
      "RARE":{
          "cb_mean":(5.279,5.26,5.30),
          "cb_sigma":(0.05,0.04,0.07),
          "cb_alphaL":(2.,0.5,3.),
          "cb_nL":(10.,1.,100.),
          "cb_alphaR":(2.,0.5,3.),
          "cb_nR":(10.,1.,100.),
          "exp_slope":(-0.5,-100.,0.),
          "expo_slope":(3.8,3.0,5.0),       # (2.25,1.,10.)
          #"expo_offset":(5.13, 5.11, 5.15), # (5.13, 5.1, 5.15)
          "erfc_mean":(5.16,5.15,5.17),     # (5.13,5.1,5.15)
          "erfc_sigma":(0.06,0.05,0.10),    # (0.03,0.015,0.05)
          },
  }.get("PSI2S")



input_mc_file = "/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/tmp/xval_data_files_updated/measurement_baseline_v6__psi2s_xval.root"

f_in = ROOT.TFile(input_mc_file,"READ")
tree = f_in.Get("mytreefit")
var = "Bmass"
mass = ROOT.RooRealVar("Bmass","Mass [GeV]",4.7,5.7)
xgbBDT = ROOT.RooRealVar("xgb","Weight",-100.,100.)
MLL = ROOT.RooRealVar("Mll","Weight",-100.,100.)
mass.setRange("full",4.7,5.7)
variables = ROOT.RooArgSet(mass,xgbBDT,MLL)
dataset = ROOT.RooDataSet("dataset","dataset",tree,variables)
cuts = ROOT.TCut("xgb>4.0&&Mll>3.55&&Mll<3.8")
dataset = dataset.reduce(cuts.GetTitle())

(idx0,idx1,idx2) = (0,1,2)

cb_mean = ROOT.RooRealVar(
    "cb_mean",
    "DS-CB: location parameter of the Gaussian component",
    defaults["cb_mean"][idx0],defaults["cb_mean"][idx1],defaults["cb_mean"][idx2])
cb_sigma = ROOT.RooRealVar(
    "cb_sigma",
    "DS-CB: width parameter of the Gaussian component",
    defaults["cb_sigma"][idx0],defaults["cb_sigma"][idx1],defaults["cb_sigma"][idx2])
cb_alphaL = ROOT.RooRealVar(
    "cb_alphaL",
    "DS-CB: location of transition to a power law on the left, in std devs away from mean",
    defaults["cb_alphaL"][idx0],defaults["cb_alphaL"][idx1],defaults["cb_alphaL"][idx2])
cb_nL = ROOT.RooRealVar(
    "cb_nL",
    "DS-CB: exponent of power-law tail on the left",
    defaults["cb_nL"][idx0],defaults["cb_nL"][idx1],defaults["cb_nL"][idx2])
cb_alphaR = ROOT.RooRealVar(
    "cb_alphaR",
    "DS-CB: location of transition to a power law on the right, in std devs away from mean",
    defaults["cb_alphaR"][idx0],defaults["cb_alphaR"][idx1],defaults["cb_alphaR"][idx2])
cb_nR = ROOT.RooRealVar(
    "cb_nR",
    "DS-CB: exponent of power-law tail on the right",
    defaults["cb_nR"][idx0],defaults["cb_nR"][idx1],defaults["cb_nR"][idx2])
cb_pdf = ROOT.RooTwoSidedCBShape(
    "cb_pdf",
    "Double-sided crystal-ball pdf",
    mass,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

model = cb_pdf

result = model.fitTo(dataset, ROOT.RooFit.Save(), ROOT.RooFit.Range("full"))
params = result.floatParsFinal()
params.Print("v")

frame = mass.frame()
dataset.plotOn(frame)
model.plotOn(frame)

canvas = ROOT.TCanvas("canvas", "Fitting Example", 800, 600)
frame.Draw()
canvas.SaveAs("MC.png")


# Write the data to the JSON file
cb_dct = {}
for param in params:
    if param.GetName() in ["cb_alphaL","cb_alphaR","cb_mean","cb_nL","cb_nR","cb_sigma","signal_num"]:
        print("PARAM:",param.GetName(),param.getVal(),param.getError())
        cb_dct[param.GetName()] = param.getVal()
print(cb_dct)
SignalParamsfile = 'cb.json'
with open(SignalParamsfile, 'w') as file:
    json.dump(cb_dct, file, indent=4)
f_in.Close()

#input_data_file = "/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/tmp/xval_data_files_updated/measurement_baseline_v6__xval.root"
input_data_file = "JPSI_DATA.root"
f_in = ROOT.TFile(input_data_file,"READ")
tree = f_in.Get("mytreefit")
var = "Bmass"
mass = ROOT.RooRealVar("Bmass","Mass [GeV]",4.7,5.7)
xgbBDT = ROOT.RooRealVar("xgb","Weight",-100.,100.)
MLL = ROOT.RooRealVar("Mll","Weight",-100.,100.)
mass.setRange("full",4.7,5.7)
variables = ROOT.RooArgSet(mass,xgbBDT,MLL)
datasetData = ROOT.RooDataSet("datasetData","datasetData",tree,variables)
cuts = ROOT.TCut("xgb>4.0&&Mll>3.55&&Mll<3.8")
datasetData = datasetData.reduce(cuts.GetTitle())
cb_mean = ROOT.RooRealVar(
    "cb_mean",
    "DS-CB: location parameter of the Gaussian component",
    cb_dct["cb_mean"],cb_dct["cb_mean"],cb_dct["cb_mean"])
cb_sigma = ROOT.RooRealVar(
    "cb_sigma",
    "DS-CB: width parameter of the Gaussian component",
    cb_dct["cb_sigma"],cb_dct["cb_sigma"],cb_dct["cb_sigma"])
cb_alphaL = ROOT.RooRealVar(
    "cb_alphaL",
    "DS-CB: location of transition to a power law on the left, in std devs away from mean",
    cb_dct["cb_alphaL"],cb_dct["cb_alphaL"],cb_dct["cb_alphaL"])
cb_nL = ROOT.RooRealVar(
    "cb_nL",
    "DS-CB: exponent of power-law tail on the left",
    cb_dct["cb_nL"],cb_dct["cb_nL"],cb_dct["cb_nL"])
cb_alphaR = ROOT.RooRealVar(
    "cb_alphaR",
    "DS-CB: location of transition to a power law on the right, in std devs away from mean",
    cb_dct["cb_alphaR"],cb_dct["cb_alphaR"],cb_dct["cb_alphaR"])
cb_nR = ROOT.RooRealVar(
    "cb_nR",
    "DS-CB: exponent of power-law tail on the right",
    cb_dct["cb_nR"],cb_dct["cb_nR"],cb_dct["cb_nR"])
cb_pdf = ROOT.RooTwoSidedCBShape(
    "cb_pdf",
    "Double-sided crystal-ball pdf",
    mass,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

exp_slope = ROOT.RooRealVar(
    "exp_slope",
    "slope of exponential",
    defaults["exp_slope"][idx0],defaults["exp_slope"][idx1],defaults["exp_slope"][idx2])

expo_pdf = ROOT.RooExponential("expo_pdf","Exponential PDF",mass,exp_slope)


cb_coeff = ROOT.RooRealVar("cb_coeff", "Gaussian Coefficient", 0.8, 0.0, 1.0)
exp_coeff = ROOT.RooRealVar("exp_coeff", "Exponential Coefficient", 0.2,0.0, 1.0)

pdf_sum = ROOT.RooAddPdf("pdf_sum", "Sum of Gaussian and Exponential",
                                                 ROOT.RooArgList(cb_pdf, expo_pdf),
                         ROOT.RooArgList(cb_coeff, exp_coeff))

result2 = pdf_sum.fitTo(datasetData, ROOT.RooFit.Save(), ROOT.RooFit.Range("full"))
params = result2.floatParsFinal()
params.Print("v")




expo_dct = {}
for param in params:
    if param.GetName() in ["exp_slope"]:
        print("PARAM:",param.GetName(),param.getVal(),param.getError())
        expo_dct[param.GetName()] = param.getVal()
print(expo_dct)
SignalParamsfile = 'exp.json'
with open(SignalParamsfile, 'w') as file:
    json.dump(expo_dct, file, indent=4)

frame = mass.frame()
datasetData.plotOn(frame)
pdf_sum.plotOn(frame)

canvas = ROOT.TCanvas("canvas", "Fitting Example", 800, 600)
frame.Draw()
canvas.SaveAs("NEWTHINGY.png")


f_in.Close()

input_data_file = "JPSI_DATA.root"
f_in = ROOT.TFile(input_data_file,"READ")
tree = f_in.Get("mytreefit")
var = "Bmass"
mass = ROOT.RooRealVar("Bmass","Mass [GeV]",4.7,5.7)
xgbBDT = ROOT.RooRealVar("xgb","Weight",-100.,100.)
MLL = ROOT.RooRealVar("Mll","Weight",-100.,100.)
mass.setRange("full",4.7,5.7)
variables = ROOT.RooArgSet(mass,xgbBDT,MLL)
datasetData2 = ROOT.RooDataSet("datasetData2","datasetData2",tree,variables)
cuts = ROOT.TCut("xgb>4.0&&Mll>3.55&&Mll<3.8")
datasetData2 = datasetData2.reduce(cuts.GetTitle())

cb_mean = ROOT.RooRealVar(
    "cb_mean",
    "DS-CB: location parameter of the Gaussian component",
    cb_dct["cb_mean"],cb_dct["cb_mean"],cb_dct["cb_mean"])
cb_sigma = ROOT.RooRealVar(
    "cb_sigma",
    "DS-CB: width parameter of the Gaussian component",
    cb_dct["cb_sigma"],cb_dct["cb_sigma"],cb_dct["cb_sigma"])
cb_alphaL = ROOT.RooRealVar(
    "cb_alphaL",
    "DS-CB: location of transition to a power law on the left, in std devs away from mean",
    cb_dct["cb_alphaL"],cb_dct["cb_alphaL"],cb_dct["cb_alphaL"])
cb_nL = ROOT.RooRealVar(
    "cb_nL",
    "DS-CB: exponent of power-law tail on the left",
    cb_dct["cb_nL"],cb_dct["cb_nL"],cb_dct["cb_nL"])
cb_alphaR = ROOT.RooRealVar(
    "cb_alphaR",
    "DS-CB: location of transition to a power law on the right, in std devs away from mean",
    cb_dct["cb_alphaR"],cb_dct["cb_alphaR"],cb_dct["cb_alphaR"])
cb_nR = ROOT.RooRealVar(
    "cb_nR",
    "DS-CB: exponent of power-law tail on the right",
    cb_dct["cb_nR"],cb_dct["cb_nR"],cb_dct["cb_nR"])
cb_pdf = ROOT.RooTwoSidedCBShape(
    "cb_pdf",
    "Double-sided crystal-ball pdf",
    mass,cb_mean,cb_sigma,cb_alphaL,cb_nL,cb_alphaR,cb_nR)

exp_slope = ROOT.RooRealVar(
    "exp_slope",
    "slope of exponential",
    expo_dct["exp_slope"],expo_dct["exp_slope"],expo_dct["exp_slope"])

expo_pdf = ROOT.RooExponential("expo_pdf","Exponential PDF",mass,exp_slope)

expo_slope = ROOT.RooRealVar(
    "expo_slope",
    "slope of exponential",
    defaults["expo_slope"][idx0],defaults["expo_slope"][idx1],defaults["expo_slope"][idx2])
#  expo_offset = ROOT.RooRealVar(
#      "expo_offset",
#      "offset of exponential",
#      defaults["expo_offset"][idx0],defaults["expo_offset"][idx1],defaults["expo_offset"][idx2])
erfc_mean = ROOT.RooRealVar(
    "erfc_mean",
    "mean of the Erfc gaussian",
    defaults["erfc_mean"][idx0],defaults["erfc_mean"][idx1],defaults["erfc_mean"][idx2])
erfc_sigma = ROOT.RooRealVar(
    "erfc_sigma",
    "width of the Erfc gaussian",
    defaults["erfc_sigma"][idx0],defaults["erfc_sigma"][idx1],defaults["erfc_sigma"][idx2])
#function = "TMath::Exp(TMath::Abs(expo_slope)*("+var+"-expo_offset))*TMath::Erfc(("+var+"-erfc_mean)/erfc_sigma)"
function = "TMath::Exp(TMath::Abs(expo_slope)*("+var+"-erfc_mean))*TMath::Erfc(("+var+"-erfc_mean)/erfc_sigma)"
generic_pdf = ROOT.RooGenericPdf(
    "generic_pdf",
    "generic pdf (expo*erfc)",
    function,ROOT.RooArgSet(mass,erfc_mean,erfc_sigma,expo_slope))


cb_coeff = ROOT.RooRealVar("cb_coeff", "Gaussian Coefficient", 0.8, 0.0, 1.0)
exp_coeff = ROOT.RooRealVar("exp_coeff", "Exponential Coefficient", 0.15,0.0, 0.9)
part_coeff = ROOT.RooRealVar("part_coeff", "part Coefficient", 0.01,0.0, 0.25)
pdf_sum1 = ROOT.RooAddPdf("pdf_sum1", "Sum of Gaussian and Exponential",
                                                 ROOT.RooArgList(cb_pdf, expo_pdf,generic_pdf),
                         ROOT.RooArgList(cb_coeff, exp_coeff,part_coeff)
)

result3 = pdf_sum1.fitTo(datasetData2, ROOT.RooFit.Save(), ROOT.RooFit.Range("full"))
generic_pdf_norm = ROOT.RooRealVar("generic_pdf_norm", "Number of background events in Tag0", 0, 0, 100000)
expo_pdf_norm = ROOT.RooRealVar("expo_pdf_norm", "Number of background events in Tag0", 0, 0, 100000)
params = result3.floatParsFinal()



frame = mass.frame()
datasetData2.plotOn(frame)
pdf_sum1.plotOn(frame)

exp_plot = ROOT.RooArgSet(expo_pdf)
cb_plot = ROOT.RooArgSet(cb_pdf)
part_plot = ROOT.RooArgSet(generic_pdf)

pdf_sum1.plotOn(frame,ROOT.RooFit.Components(cb_plot),
             ROOT.RooFit.LineStyle(ROOT.kDotted))
pdf_sum1.plotOn(frame,ROOT.RooFit.Components(exp_plot),
             ROOT.RooFit.LineStyle(ROOT.kDashed))
pdf_sum1.plotOn(frame,ROOT.RooFit.Components(part_plot),
             ROOT.RooFit.LineStyle(ROOT.kDashed),ROOT.RooFit.LineColor(
        ROOT.kYellow))
# Draw the frame on canvas
canvas = ROOT.TCanvas("canvas", "Fitting Example", 800, 600)
frame.Draw()
canvas.SaveAs("psi2s.png")


f_out = ROOT.TFile("workspace_simple.root", "RECREATE")
workspace = ROOT.RooWorkspace("workspace_simple","workspace_simple")
getattr(workspace, "import")(datasetData2)
getattr(workspace, "import")(pdf_sum1)
getattr(workspace, "import")(expo_pdf_norm)
getattr(workspace, "import")(generic_pdf_norm)

workspace.Print()
workspace.Write()

f_out.Close()
f_in.Close()
