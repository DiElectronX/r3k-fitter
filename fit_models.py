import ROOT
from utils import *

class PDFDict():
    def __init__(self, name, shape, xvar, parameters, dataset=None, let_float=False):
        allowed_shapes = ['dcb', 'cb+gauss', 'cb+cb', 'exp', 'generic', 'kde']
        assert shape in allowed_shapes, "Choose a PDF shape from list:{}".format(allowed_shapes)

        self.name = name
        self.parameters = parameters
        self.build_model(shape, xvar, self.parameters, let_float, dataset)


    def build_model(self, shape, xvar, parameters, let_float, dataset):
        if shape=='dcb':
            self.cb_mean = ROOT.RooRealVar(
                'cb_mean',
                'DS-CB: location parameter of the Gaussian component',
                *parameters['cb_mean'])
            self.cb_sigma = ROOT.RooRealVar(
                'cb_sigma',
                'DS-CB: width parameter of the Gaussian component',
                *parameters['cb_sigma'])
            self.cb_alphaL = ROOT.RooRealVar(
                'cb_alphaL',
                'DS-CB: location of transition to a power law on the left, in std devs away from mean',
                *parameters['cb_alphaL'])
            self.cb_nL = ROOT.RooRealVar(
                'cb_nL',
                'DS-CB: exponent of power-law tail on the left',
                *parameters['cb_nL'])
            self.cb_alphaR = ROOT.RooRealVar(
                'cb_alphaR',
                'DS-CB: location of transition to a power law on the right, in std devs away from mean',
                *parameters['cb_alphaR'])
            self.cb_nR = ROOT.RooRealVar(
                'cb_nR',
                'DS-CB: exponent of power-law tail on the right',
                *parameters['cb_nR'])

            self.model = ROOT.RooTwoSidedCBShape(
                self.name,
                'Double-sided crystal-ball pdf',
                xvar, self.cb_mean, self.cb_sigma, self.cb_alphaL, self.cb_nL, self.cb_alphaR, self.cb_nR)

        if shape=='cb+gauss':
            self.gauss_mean = ROOT.RooRealVar(
                'gauss_mean',
                'CB+Gauss: Mean of gaussian component',
                *parameters['gauss_mean'])
            self.gauss_sigma = ROOT.RooRealVar(
                'gauss_sigma',
                'CB+Gauss: Width of gaussian component',
                *parameters['gauss_sigma'])
            self.gauss_pdf = ROOT.RooGaussian(
                'gauss_pdf',
                'CB+Gauss: Gaussian component',
                xvar, self.gauss_mean, self.gauss_sigma)
            self.cb_mean = ROOT.RooRealVar(
                'cb_mean',
                'CB+Gauss: Mean of CB component',
                *parameters['cb_mean'])
            self.cb_sigma = ROOT.RooRealVar(
                'cb_sigma',
                'CB+Gauss: Width of CB component',
                *parameters['cb_sigma'])
            self.cb_alpha = ROOT.RooRealVar(
                'cb_alpha',
                'CB+Gauss: Location of transition to a power law of CB component',
                *parameters['cb_alpha'])
            self.cb_n = ROOT.RooRealVar(
                'cb_n',
                'CB+Gauss: Exponent of power-law tail of CB component',
                *parameters['cb_n'])
            self.cb_pdf = ROOT.RooCBShape(
                'cb_pdf',
                'CB+Gauss: CB component',
                xvar, self.cb_mean, self.cb_sigma, self.cb_alpha, self.cb_n)

            self.cb_coeff = ROOT.RooRealVar('cb_coeff', 'CB Coefficient', 0.8, 0.0, 1.0)
            self.gauss_coeff = ROOT.RooRealVar('gauss_coeff', 'Gaussian Coefficient', 0.2,0.0, 1.0)
            self.model = ROOT.RooAddPdf(
                 self.name,
                'CB+Gauss',
                 ROOT.RooArgList(self.cb_pdf, self.gauss_pdf),
                 ROOT.RooArgList(self.cb_coeff, self.gauss_coeff)
            )

        if shape=='cb+cb':
            self.cb1_mean = ROOT.RooRealVar(
                'cb1_mean',
                'CB+CB: Mean of CB1 component',
                *parameters['cb1_mean'])
            self.cb1_sigma = ROOT.RooRealVar(
                'cb1_sigma',
                'CB+CB: Width of CB1 component',
                *parameters['cb1_sigma'])
            self.cb1_alpha = ROOT.RooRealVar(
                'cb1_alpha',
                'CB+CB: Location of transition to a power law of CB1 component',
                *parameters['cb1_alpha'])
            self.cb1_n = ROOT.RooRealVar(
                'cb1_n',
                'CB+CB: Exponent of power-law tail of CB1 component',
                *parameters['cb1_n'])
            self.cb1_pdf = ROOT.RooCBShape(
                'cb1_pdf',
                'CB+CB: CB1 component',
                xvar, self.cb1_mean, self.cb1_sigma, self.cb1_alpha, self.cb1_n)
            self.cb2_mean = ROOT.RooRealVar(
                'cb2_mean',
                'CB+CB: Mean of CB2 component',
                *parameters['cb2_mean'])
            self.cb2_sigma = ROOT.RooRealVar(
                'cb2_sigma',
                'CB+CB: Width of CB2 component',
                *parameters['cb2_sigma'])
            self.cb2_alpha = ROOT.RooRealVar(
                'cb2_alpha',
                'CB+CB: Location of transition to a power law of CB2 component',
                *parameters['cb2_alpha'])
            self.cb2_n = ROOT.RooRealVar(
                'cb2_n',
                'CB+CB: Exponent of power-law tail of CB2 component',
                *parameters['cb2_n'])
            self.cb2_pdf = ROOT.RooCBShape(
                'cb2_pdf',
                'CB+CB: CB2 component',
                xvar, self.cb2_mean, self.cb2_sigma, self.cb2_alpha, self.cb2_n)

            self.cb1_coeff = ROOT.RooRealVar('cb1_coeff', 'CB1 Coefficient', 1., 0.0, 1000000.)
            self.cb2_coeff = ROOT.RooRealVar('cb2_coeff', 'CB2 Coefficient', 1., 0.0, 1000000.)
            self.model = ROOT.RooAddPdf(
                self.name,
                'CB+CB',
                 ROOT.RooArgList(self.cb1_pdf, self.cb2_pdf),
                 ROOT.RooArgList(self.cb1_coeff, self.cb2_coeff)
            )


        if shape=='exp':
            self.exp_slope = ROOT.RooRealVar(
                'exp_slope',
                'slope of exponential',
                *parameters['exp_slope'])

            self.model = ROOT.RooExponential(self.name, 'Exponential PDF', xvar, self.exp_slope)

        if shape=='generic':
            self.part_exp_slope = ROOT.RooRealVar(
                'part_exp_slope',
                'slope of exponential',
                *parameters['part_exp_slope'])
            self.erfc_mean = ROOT.RooRealVar(
                'erfc_mean',
                'mean of the Erfc gaussian',
                *parameters['erfc_mean'])
            self.erfc_sigma = ROOT.RooRealVar(
                'erfc_sigma',
                'width of the Erfc gaussian',
                *parameters['erfc_sigma'])

            function = 'TMath::Exp(TMath::Abs(part_exp_slope)*('+xvar.GetName()+'-erfc_mean))*TMath::Erfc(('+xvar.GetName()+'-erfc_mean)/erfc_sigma)'
            self.model = ROOT.RooGenericPdf(
                self.name,
                'Generic PDF (exp*erfc)',
                function,ROOT.RooArgSet(xvar, self.erfc_mean, self.erfc_sigma, self.part_exp_slope)
            )

        if shape=='kde':
            assert dataset is not None, 'Dataset required for KDE initilization'
            self.model = ROOT.RooKeysPdf(self.name, 'Kernel Density Estimate PDF', xvar, dataset, ROOT.RooKeysPdf.NoMirror)


class FitModel:
    def __init__(self, dictionary={}):
        self.branch = None
        self.dataset = None
        self.signal_models = {}
        self.background_models = {}
        self.constraints = {}
        self.fit_model = None
        self.fit_result = None

        for k, v in dictionary.items():
            setattr(self, k, v)

        assert self.branch is not None, "Must define variable for fit"


    def add_signal_model(self, name, shape, parameters, let_float=True):
        fit_params = format_params(parameters, let_float)
        sig_model = PDFDict(name, shape, self.branch, fit_params, self.dataset, let_float)
        self.signal_models[name] = sig_model
        setattr(self, sig_model.name, sig_model.model)


    def add_background_model(self, name, shape, parameters, let_float=True):
        fit_params = format_params(parameters, let_float)
        bkg_model = PDFDict(name, shape, self.branch, fit_params, self.dataset, let_float)
        self.background_models[name] = bkg_model
        setattr(self, bkg_model.name, bkg_model.model)


    def fit(self, dataset, fit_range=ROOT.RooFit.Range('full'), printlevel=ROOT.RooFit.PrintLevel(-1)):
        if self.constraints:
            self.fit_result = self.fit_model.fitTo(dataset, ROOT.RooFit.Save(), ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*self.constraints.values())), fit_range, printlevel)
        else:
            self.fit_result = self.fit_model.fitTo(dataset, ROOT.RooFit.Save(), fit_range, printlevel)

    def plot_fit(self, branch, dataset, output_filepath, fit_components=[]):
        assert self.fit_model is not None, "Must assign 'fit_model'"
        plot_model = self.fit_model

        get_color = (col for col in [ROOT.kBlue, ROOT.kGreen+3, ROOT.kRed+2, ROOT.kOrange-3, ROOT.kMagenta+1, ROOT.kCyan+1])
        frame = branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        dataset.plotOn(frame, ROOT.RooFit.Name(dataset.GetName()))
        plot_model.plotOn(
            frame,
            ROOT.RooFit.Range('full'),
            ROOT.RooFit.NormRange(''),
            ROOT.RooFit.Name(plot_model.GetName()),
            ROOT.RooFit.LineStyle(ROOT.kSolid),
            ROOT.RooFit.LineColor(next(get_color)),
        )

        h_pull = frame.pullHist()
        frame_pull = branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        frame_pull.addPlotable(h_pull, 'P')

        for comp in fit_components:
            plot_argset = ROOT.RooArgSet(comp)
            plot_model.plotOn(
                frame,
                ROOT.RooFit.Components(plot_argset),
                ROOT.RooFit.Range('full'),
                ROOT.RooFit.NormRange(''),
                ROOT.RooFit.LineStyle(ROOT.kDashed),
                ROOT.RooFit.LineColor(next(get_color))
            )

        chi2 = frame.chiSquare(
            plot_model.GetName(),
            dataset.GetName(),
            len(plot_model.getParameters(dataset))
        )

        c = ROOT.TCanvas('c', ' ', 800, 600)
        pad1 = ROOT.TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.02)  # joins upper and lower plot
        pad1.SetGridx()
        pad1.Draw()
        c.cd()
        pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.2)
        pad2.SetGridx()
        pad2.Draw()

        pad1.cd()
        frame.Draw()
        ax_y_main = frame.GetYaxis()
        ax_x_main = frame.GetXaxis()
        ax_x_main.SetLabelOffset(3.)

        label = ROOT.TLatex(0.65, 0.8, '#chi^{{2}}/ndf = {}'.format(round(chi2,1)))
        label.SetTextSize(0.08)
        label.SetNDC(ROOT.kTRUE)
        label.Draw()

        pad2.cd()
        frame_pull.Draw()

        ax_y_pull = frame_pull.GetYaxis()
        ax_x_pull = frame_pull.GetXaxis()

        line = ROOT.TLine(ax_x_pull.GetXmin(), 0, ax_x_pull.GetXmax(), 0)
        line.SetLineStyle(7)
        line.Draw()

        ax_y_pull.SetTitle('#frac{y - y_{fit}}{#sigma_{y}}')
        ax_y_pull.SetTitleOffset(.35)
        ax_y_pull.SetNdivisions(8)

        ax_y_pull.SetTitleSize(2.8*ax_y_main.GetTitleSize())
        ax_y_pull.SetLabelSize(2.8*ax_y_main.GetLabelSize())
        ax_x_pull.SetTitleSize(2.8*ax_x_main.GetTitleSize())
        ax_x_pull.SetLabelSize(2.8*ax_x_main.GetLabelSize())

        c.SaveAs(output_filepath)
        c.SaveAs(output_filepath.replace('pdf','png'))
        c.Close()
