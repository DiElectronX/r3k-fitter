import sys
import ROOT

filename = sys.argv[1]
f = ROOT.TFile(filename)
w = f.Get("w")

n_bins = 50
binning = ROOT.RooFit.Binning(n_bins,4.7,5.7)

can = ROOT.TCanvas()
plot = w.var("Bmass").frame()
w.data("data_obs").plotOn( plot, binning )

# Load the S+B model
sb_model = w.pdf("model_s").getPdf("jpsi_region")

# Prefit
sb_model.plotOn( plot, ROOT.RooFit.LineColor(2), ROOT.RooFit.Name("prefit") )

# Postfit
w.loadSnapshot("MultiDimFit")
sb_model.plotOn( plot, ROOT.RooFit.LineColor(4), ROOT.RooFit.Name("postfit") )
r_bestfit = w.var("r").getVal()

plot.Draw()

leg = ROOT.TLegend(0.55,0.6,0.85,0.85)
leg.AddEntry("prefit", "Prefit S+B model (r=1.00)", "L")
leg.AddEntry("postfit", "Postfit S+B model (r=%.2f)"%r_bestfit, "L")
leg.Draw("Same")

can.Update()
can.SaveAs(filename.rstrip(".root")+".beforeAndAfter.png")

f.Close()