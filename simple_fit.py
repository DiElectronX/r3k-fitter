import ROOT

input_path = "/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/tmp/xval_data_files_updated/measurement_baseline_v6__xval.root"
f_in = ROOT.TFile(input_path,"READ")
tree = f_in.Get("mytreefit")
mass = ROOT.RooRealVar("Bmass","Mass [GeV]",4.7,5.7)
xgbBDT = ROOT.RooRealVar("xgb","Weight",-100.,100.)

mass.setRange("full",4.7,5.7)
variables = ROOT.RooArgSet(mass,xgbBDT)
dataset = ROOT.RooDataSet("dataset","dataset",tree,variables)
cuts = ROOT.TCut("xgb>4.0")
dataset = dataset.reduce(cuts.GetTitle())

mean = ROOT.RooRealVar("mean", "Mean of Gaussian", 5.3, 4.8, 5.5)
sigma = ROOT.RooRealVar("sigma", "Width of Gaussian", 0.1, 0.01, 1.0)
gaussian = ROOT.RooGaussian("gaussian", "Gaussian PDF", mass, mean, sigma)

tau = ROOT.RooRealVar("tau", "Exponential decay parameter", -0.1, -1.0, 0.0)
exponential = ROOT.RooExponential("exponential", "Exponential PDF", mass, tau)

gauss_coeff = ROOT.RooRealVar("gauss_coeff", "Gaussian Coefficient", 0.8, 0.0, 1.0)
exp_coeff = ROOT.RooRealVar("exp_coeff", "Exponential Coefficient", 0.2, 0.0, 1.0)

pdf_sum = ROOT.RooAddPdf("pdf_sum", "Sum of Gaussian and Exponential", 
						 ROOT.RooArgList(gaussian, exponential),
                         ROOT.RooArgList(gauss_coeff, exp_coeff)
)

# Perform the fit
result = pdf_sum.fitTo(dataset, ROOT.RooFit.Save(), ROOT.RooFit.Range("full"))

# Print fit results
result.Print()

# Plot the fit result
frame = mass.frame()
dataset.plotOn(frame)
pdf_sum.plotOn(frame)

# Draw the frame on canvas
canvas = ROOT.TCanvas("canvas", "Fitting Example", 800, 600)
frame.Draw()
canvas.SaveAs("simple_fit_result.png")

f_out = ROOT.TFile("simple_fit_result.root", "RECREATE")
workspace = ROOT.RooWorkspace("workspace","workspace")
getattr(workspace, "import")(dataset)
getattr(workspace, "import")(pdf_sum)
workspace.Print()
workspace.Write()

f_out.Close()
f_in.Close()
