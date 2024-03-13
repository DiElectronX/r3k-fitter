import ROOT

class Model:
    def __init__(self, dictionary={}):
        self.branch = None
        self.dataset = None
        self.signal_model = None
        self.background_models = {}
        self.constraints = {}
        self.fit_model = None
        self.fit_result = None

        for k, v in dictionary.items():
            setattr(self, k, v)

    def format_params(self, params, let_float=False):
        params = {k : (v if isinstance(v,list) else (v,)) for k, v in params.items()}
        params = {k : (v if let_float else (v[0],)) for k, v in params.items()}

        return params

    def build_signal_model(self, shape, mass, parameters, let_float=True):
        assert shape in ['dcb', 'cb+gauss']
        if self.branch:
            assert self.branch == mass
        parameters = self.format_params(parameters, let_float)

        if 'dcb' in shape:
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

            self.signal_model = ROOT.RooTwoSidedCBShape(
                'sig_pdf',
                'Double-sided crystal-ball pdf',
                mass, self.cb_mean, self.cb_sigma, self.cb_alphaL, self.cb_nL, self.cb_alphaR, self.cb_nR)

        if 'cb+gauss':
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
                mass, self.gauss_mean, self.gauss_sigma)

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
                mass, self.cb_mean, self.cb_sigma, self.cb_alpha, self.cb_n)

            self.cb_coeff = ROOT.RooRealVar('cb_coeff', 'CB Coefficient', 0.8, 0.0, 1.0)
            self.gauss_coeff = ROOT.RooRealVar('gauss_coeff', 'Gaussian Coefficient', 0.2,0.0, 1.0)
            self.signal_model = ROOT.RooAddPdf(
                'sig_pdf',
                'CB+Gauss',
                 ROOT.RooArgList(self.cb_pdf, self.gauss_pdf),
                 ROOT.RooArgList(self.cb_coeff, self.gauss_coeff)
            )


    def add_background_model(self, name, shape, mass, parameters, let_float=True):
        assert shape in ['exp', 'generic']
        if self.branch:
            assert self.branch == mass
        self.branch = mass
        parameters = self.format_params(parameters, let_float)

        if 'exp' in shape:
            self.exp_slope = ROOT.RooRealVar(
                'exp_slope',
                'slope of exponential',
                *parameters['exp_slope'] if let_float else parameters['exp_slope'])
            self.background_models[name] = ROOT.RooExponential(name, 'Exponential PDF', mass, self.exp_slope)

        if 'generic' in shape:
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

            function = 'TMath::Exp(TMath::Abs(part_exp_slope)*('+mass.GetName()+'-erfc_mean))*TMath::Erfc(('+mass.GetName()+'-erfc_mean)/erfc_sigma)'
            self.background_models[name] = ROOT.RooGenericPdf(
                name,
                'Generic PDF (exp*erfc)',
                function,ROOT.RooArgSet(mass, self.erfc_mean, self.erfc_sigma, self.part_exp_slope)
            )

    def plot_fit(self, branch, dataset, output_filepath, fit_components=[]):
        assert self.fit_model is not None

        get_color = (col for col in [ROOT.kBlue, ROOT.kGreen+3, ROOT.kRed+2, ROOT.kOrange-3])
        frame = branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        dataset.plotOn(frame, ROOT.RooFit.Name(dataset.GetName()))
        self.fit_model.plotOn(
            frame,
            ROOT.RooFit.Range('full'),
            ROOT.RooFit.NormRange(''),
            ROOT.RooFit.Name(self.fit_model.GetName()),
            ROOT.RooFit.LineStyle(ROOT.kSolid),
            ROOT.RooFit.LineColor(next(get_color)),
        )

        h_pull = frame.pullHist()
        frame_pull = branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        frame_pull.addPlotable(h_pull, 'P')

        for comp in fit_components:
            plot_argset = ROOT.RooArgSet(comp)
            self.fit_model.plotOn(
                frame,
                ROOT.RooFit.Components(plot_argset),
                ROOT.RooFit.Range('full'),
                ROOT.RooFit.NormRange(''),
                ROOT.RooFit.LineStyle(ROOT.kDashed),
                ROOT.RooFit.LineColor(next(get_color))
            )

        chi2 = frame.chiSquare(
            self.fit_model.GetName(),
            dataset.GetName(),
            len(self.fit_model.getParameters(dataset))
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

