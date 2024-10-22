import ROOT
 
f_in = ROOT.TFile('/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/fit_input_files/6_19_24/measurement_k0star_psi2s_kaon.root', 'READ')
tree = f_in.Get('mytree')
b_mass_branch = ROOT.RooRealVar('Bmass', 'B Candidate Mass [GeV]', 4.7,5.7)
bdt_branch = ROOT.RooRealVar('bdt_score', 'BDT Score', -100., 100.)
ll_mass_branch = ROOT.RooRealVar('Mll', 'Di-Lepton Mass [GeV]', -100., 100.)
weight_branch = ROOT.RooRealVar('trig_wgt', 'Weight', -100., 100.)
variables = ROOT.RooArgSet(b_mass_branch, bdt_branch, ll_mass_branch, weight_branch)
dataset = ROOT.RooDataSet('dataset_mc', 'Dataset', tree, variables, weight_branch.GetName())
cuts = ROOT.TCut('Mll>3.55&&Mll<3.8&&bdt_score>4.5')
dataset = dataset.reduce(cuts.GetTitle())

keys_pdf = ROOT.RooKeysPdf('keys','keys',b_mass_branch,dataset, mirror=ROOT.RooKeysPdf.MirrorLeft, rho=2)
xset = ROOT.RooArgSet(b_mass_branch)
nset = ROOT.RooFit.NormSet(xset)
b_mass_branch.setRange('signal', 5.1, 5.4)
rangeset = ROOT.RooFit.Range('signal')

print(keys_pdf)
print(b_mass_branch)
print(xset)
print(nset)
print(rangeset)

integral = keys_pdf.createIntegral(xset,nset)
integral_sig = keys_pdf.createIntegral(xset, nset, rangeset)
print(f'{integral.getVal()} ; {integral_sig.getVal()}')
