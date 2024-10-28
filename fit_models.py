import os
from pathlib import Path
import ROOT
from utils import *

class PDFDict():
    def __init__(self, name, shape, xvar, parameters, dataset=None, let_float=False, channel=None):
        allowed_shapes = ['dcb', 'cb+gauss', 'dcb+dcb', 'cb+cb', 'exp', 'poly', 'generic', 'kde']
        assert shape in allowed_shapes, "Choose a PDF shape from list:{}".format(allowed_shapes)

        self.name = name
        self.parameters = parameters
        self.channel = channel
        self.channel_label = channel if channel else ''
        self.build_model(shape, xvar, self.parameters, let_float, dataset, label=self.name)


    def build_model(self, shape, xvar, parameters, let_float, dataset, label=''):
        if shape=='dcb':
            shape_dict = {
                'dcb_mean' :   'DS-CB: location parameter of the Gaussian component',
                'dcb_sigma' :  'DS-CB: width parameter of the Gaussian component',
                'dcb_alphaL' : 'DS-CB: location of transition to a power law on the left, in std devs away from mean',
                'dcb_nL' :     'DS-CB: exponent of power-law tail on the left',
                'dcb_alphaR' : 'DS-CB: location of transition to a power law on the right, in std devs away from mean',
                'dcb_nR' :     'DS-CB: exponent of power-law tail on the right',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.model = ROOT.RooTwoSidedCBShape(
                self.name+self.channel_label,
                'Double-sided crystal-ball pdf',
                xvar, self.dcb_mean, self.dcb_sigma, self.dcb_alphaL, self.dcb_nL, self.dcb_alphaR, self.dcb_nR)

        if shape=='cb+gauss':
            shape_dict = {
                'gauss_mean' :  'CB+Gauss: Mean of gaussian component',
                'gauss_sigma' : 'CB+Gauss: Width of gaussian component',
                'cb_mean' :     'CB+Gauss: Mean of CB component',
                'cb_sigma' :    'CB+Gauss: Width of CB component',
                'cb_alpha' :    'CB+Gauss: Location of transition to a power law of CB component',
                'cb_n' :        'CB+Gauss: Exponent of power-law tail of CB component',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.gauss_pdf = ROOT.RooGaussian(
                'gauss_pdf'+self.channel_label,
                'CB+Gauss: Gaussian component',
                xvar, self.gauss_mean, self.gauss_sigma)

            self.cb_pdf = ROOT.RooCBShape(
                'cb_pdf'+self.channel_label,
                'CB+Gauss: CB component',
                xvar, self.cb_mean, self.cb_sigma, self.cb_alpha, self.cb_n)

            self.cb_coeff = ROOT.RooRealVar('cb_coeff'+self.channel_label, 'CB Coefficient', 0.8, 0.0, 1.0+channel_label)
            self.gauss_coeff = ROOT.RooRealVar('gauss_coeff'+self.channel_label, 'Gaussian Coefficient', 0.2,0.0, 1.0+channel_label)
            self.model = ROOT.RooAddPdf(
                 self.name+self.channel_label,
                'CB+Gauss',
                 ROOT.RooArgList(self.cb_pdf, self.gauss_pdf),
                 ROOT.RooArgList(self.cb_coeff, self.gauss_coeff)
            )

        if shape=='dcb+dcb':
            shape_dict = {
                'dcb1_coeff'  : 'DCB+DCB: DCB1 Coefficient', 
                'dcb1_mean'   : 'DCB+DCB: Mean of DCB1 component', 
                'dcb1_sigma'  : 'DCB+DCB: Width of DCB1 component', 
                'dcb1_alpha1' : 'DCB+DCB: Location of left transition to a power law of DCB1 component', 
                'dcb1_n1'     : 'DCB+DCB: Exponent of left power-law tail of DCB1 component', 
                'dcb1_alpha2' : 'DCB+DCB: Location of right transition to a power law of DCB1 component', 
                'dcb1_n2'     : 'DCB+DCB: Exponent of right power-law tail of DCB1 component', 
                'dcb2_coeff'  : 'DCB+DCB: DCB2 Coefficient', 
                'dcb2_mean'   : 'DCB+DCB: Mean of DCB2 component', 
                'dcb2_sigma'  : 'DCB+DCB: Width of DCB2 component', 
                'dcb2_alpha1' : 'DCB+DCB: Location of left transition to a power law of DCB2 component', 
                'dcb2_n1'     : 'DCB+DCB: Exponent of left power-law tail of DCB2 component', 
                'dcb2_alpha2' : 'DCB+DCB: Location of right transition to a power law of DCB2 component', 
                'dcb2_n2'     : 'DCB+DCB: Exponent of right power-law tail of DCB2 component', 
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.dcb1_pdf = ROOT.RooTwoSidedCBShape(
                'dcb1_pdf'+self.channel_label,
                'DCB+DCB: DCB1 component',
                xvar, self.dcb1_mean, self.dcb1_sigma, self.dcb1_alpha1, self.dcb1_n1, self.dcb1_alpha2, self.dcb1_n2)

            self.dcb2_pdf = ROOT.RooTwoSidedCBShape(
                'dcb2_pdf'+self.channel_label,
                'DCB+DCB: DCB2 component',
                xvar, self.dcb2_mean, self.dcb2_sigma, self.dcb2_alpha1, self.dcb2_n1, self.dcb2_alpha2, self.dcb2_n2)

            #self.dcb1_coeff = ROOT.RooRealVar('dcb1_coeff'+self.channel_label, 'DCB1 Coefficient',1., 0.0, 1000000.)
            #self.dcb2_coeff = ROOT.RooRealVar('dcb2_coeff'+self.channel_label, 'DCB2 Coefficient',1., 0.0, 1000000.)
            self.model = ROOT.RooAddPdf(
                self.name+self.channel_label,
                'DCB+DCB',
                 ROOT.RooArgList(self.dcb1_pdf, self.dcb2_pdf),
                 ROOT.RooArgList(self.dcb1_coeff, self.dcb2_coeff)
            )

        if shape=='cb+cb':
            shape_dict = {
                'cb1_mean'  : 'CB+CB: Mean of CB1 component', 
                'cb1_sigma' : 'CB+CB: Width of CB1 component', 
                'cb1_alpha' : 'CB+CB: Location of transition to a power law of CB1 component', 
                'cb1_n'     : 'CB+CB: Exponent of power-law tail of CB1 component', 
                'cb2_mean'  : 'CB+CB: Mean of CB2 component', 
                'cb2_sigma' : 'CB+CB: Width of CB2 component', 
                'cb2_alpha' : 'CB+CB: Location of transition to a power law of CB2 component', 
                'cb2_n'     : 'CB+CB: Exponent of power-law tail of CB2 component', 
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.cb1_pdf = ROOT.RooCBShape(
                'cb1_pdf'+self.channel_label,
                'CB+CB: CB1 component',
                xvar, self.cb1_mean, self.cb1_sigma, self.cb1_alpha, self.cb1_n)

            self.cb2_pdf = ROOT.RooCBShape(
                'cb2_pdf'+self.channel_label,
                'CB+CB: CB2 component',
                xvar, self.cb2_mean, self.cb2_sigma, self.cb2_alpha, self.cb2_n)

            self.cb1_coeff = ROOT.RooRealVar('cb1_coeff'+self.channel_label, 'CB1 Coefficient',1., 0.0, 1000000.)
            self.cb2_coeff = ROOT.RooRealVar('cb2_coeff'+self.channel_label, 'CB2 Coefficient',1., 0.0, 1000000.)
            self.model = ROOT.RooAddPdf(
                self.name+self.channel_label,
                'CB+CB',
                 ROOT.RooArgList(self.cb1_pdf, self.cb2_pdf),
                 ROOT.RooArgList(self.cb1_coeff, self.cb2_coeff)
            )

        if shape=='exp':
            shape_dict = {
                'exp_slope' :   'Exp: slope of exponential',
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            self.model = ROOT.RooExponential(self.name+self.channel_label, 'Exponential PDF', xvar, self.exp_slope)

        if shape=='poly':
            n_polypars = sum('poly_a' in s for s in parameters.keys())
            shape_dict = {'poly_a{}'.format(i) : 'Poly: {}th coeff.'.format(i) for i in range(n_polypars)}
            shape_dict.update({'poly_offset' : 'Poly: x-axis offset'})
            
            model_pars = []
            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                roovar = ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt]
                )
                setattr(self, par, roovar)
                model_pars.append(roovar)

            diff = ROOT.RooFormulaVar('diff','{}-{}'.format(xvar.GetName(), self.poly_offset.GetName()), ROOT.RooArgList(xvar, self.poly_offset))

            self.model = ROOT.RooPolynomial(self.name+self.channel_label, 'Exponential PDF', xvar, ROOT.RooArgList(*model_pars))

        if shape=='generic':
            shape_dict = {
                'exp_slope'   : 'Generic (exp*erfc): slope of exponential', 
                'erfc_mean'   : 'Generic (exp*erfc): mean of error function', 
                'erfc_sigma'  : 'Generic (exp*erfc): width of error function', 
            }

            for par, desc in shape_dict.items():
                name_fmt = par+'_'+label if label else par
                setattr(self, par, ROOT.RooRealVar(
                    name_fmt+self.channel_label,
                    desc,
                    *parameters[name_fmt])
                )

            function = 'TMath::Exp(TMath::Abs({})*({}-{}))*TMath::Erfc(({}-{})/{})'.format(self.exp_slope.GetName(), xvar.GetName(), self.erfc_mean.GetName(), xvar.GetName(), self.erfc_mean.GetName(), self.erfc_sigma.GetName())
            self.model = ROOT.RooGenericPdf(
                self.name+self.channel_label,
                'Generic PDF (exp*erfc)',
                function,ROOT.RooArgSet(xvar, self.erfc_mean, self.erfc_sigma, self.exp_slope)
            )

        if shape=='kde':
            assert dataset is not None, 'Dataset required for KDE initilization'
            name_fmt = 'kde_mirror_'+label if label else 'kde_mirror'
            kde_mirror = getattr(ROOT.RooKeysPdf, *parameters[name_fmt])
            name_fmt = 'kde_rho_'+label if label else 'kde_rho'
            self.model = ROOT.RooKeysPdf(self.name+self.channel_label, 'Kernel Density Estimate PDF', xvar, dataset, kde_mirror,*parameters[name_fmt])


class FitModel:
    def __init__(self, dictionary={}):
        self.branch = None
        self.dataset = None
        self.channel_label = None
        self.signal_models = {}
        self.background_models = {}
        self.constraints = {}
        self.fit_model = None
        self.fit_result = None

        for k, v in dictionary.items():
            setattr(self, k, v)

        assert self.branch is not None, "Must define variable for fit"


    def add_signal_model(self, *args, **kwds):
        if len(args)==2 and isinstance(args[0], str) and isinstance(args[1], PDFDict):
            self.add_signal_model_from_object(*args, **kwds)
        else:
            self.add_signal_model_from_scratch(*args, **kwds)


    def add_signal_model_from_scratch(self, name, shape, parameters, let_float=True):
        fit_params = format_params(parameters, let_float)
        sig_model = PDFDict(name, shape, self.branch, fit_params, self.dataset, let_float, self.channel_label)
        self.signal_models[name] = sig_model
        setattr(self, sig_model.name, sig_model.model)


    def add_signal_model_from_object(self, name, model_dict):
        self.signal_models[name] = model_dict
        setattr(self, name, model_dict.model)


    def add_background_model(self, *args, **kwds):
        if len(args)==2 and isinstance(args[0], str) and isinstance(args[1], PDFDict):
            self.add_background_model_from_object(*args, **kwds)
        else:
            self.add_background_model_from_scratch(*args, **kwds)


    def add_background_model_from_scratch(self, name, shape, parameters, let_float=True):
        fit_params = format_params(parameters, let_float)
        bkg_model = PDFDict(name, shape, self.branch, fit_params, self.dataset, let_float, self.channel_label)
        self.background_models[name] = bkg_model
        setattr(self, bkg_model.name, bkg_model.model)


    def add_background_model_from_object(self, name, model_dict):
        self.background_models[name] = model_dict
        setattr(self, name, model_dict.model)


    def add_constraints(self, constraint_dict):
        self.constraints.update(constraint_dict)


    def fit(self, dataset, fit_range='full', fit_norm_range='full', printlevel=ROOT.RooFit.PrintLevel(-1)):
        fit_args = [
            dataset,
            ROOT.RooFit.Save(),
            ROOT.RooFit.Range(fit_range),
            #ROOT.RooFit.NormRange(fit_norm_range),
            printlevel,
        ]

        if self.constraints:
            constraintset = ROOT.RooArgSet()
            for c in self.constraints.values():
                constraintset.add(c)

            fit_args.append(ROOT.RooFit.ExternalConstraints(constraintset))
            # fit_args.append(ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(*self.constraints.values())))

        self.fit_result = self.fit_model.fitTo(*fit_args)


    def plot_fit(self, branch, dataset, output_filepath, fit_components=[], bins=None, fit_range='full', fit_norm_range='full', file_formats=['pdf', 'png'], fit_result=None, legend=False, extra_text=None):
        assert self.fit_model is not None, "Must assign 'fit_model'"
        plot_model = self.fit_model

        hex_colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        root_colors = [ROOT.TColor.GetColor(col) for col in hex_colors]
        get_color = (col for col in root_colors)

        frame = branch.frame(
            ROOT.RooFit.Title(' '),
            #ROOT.RooFit.Range('full'),
            #fit_range,
            #fit_norm_range,
        )

        leg = ROOT.TLegend(.1, .6, .4, .9)

        if fit_range!='full':
            dataset = dataset.reduce(ROOT.RooFit.CutRange(fit_range))

        fit_range = ROOT.RooFit.Range(fit_range)
        fit_norm_range = ROOT.RooFit.NormRange(fit_norm_range)

        if bins is not None:
            if isinstance(bins,int):
                bins = [bins, branch.getMin(), branch.getMax()]
            bins = ROOT.RooBinning(*bins)
            dataset.plotOn(frame, ROOT.RooFit.Name(dataset.GetName()), ROOT.RooFit.Binning(bins))
        else:
            dataset.plotOn(frame,ROOT.RooFit.Name(dataset.GetName()))

        leg.AddEntry(frame.findObject(dataset.GetName()), dataset.GetTitle(), 'PE')

        plot_model.plotOn(
            frame,
            ROOT.RooFit.Range('full'),
            fit_norm_range,
            ROOT.RooFit.Name(plot_model.GetName()),
            ROOT.RooFit.LineStyle(ROOT.kSolid),
            ROOT.RooFit.LineColor(next(get_color)),
        )
        leg.AddEntry(frame.findObject(plot_model.GetName()), plot_model.GetTitle(), 'L')

        h_pull = frame.pullHist()
        frame_pull = branch.frame(ROOT.RooFit.Title(' '), ROOT.RooFit.Range('full'))
        frame_pull.addPlotable(h_pull, 'P')
        
        for comp in fit_components:
            plot_argset = ROOT.RooArgSet(comp)
            plot_comp = ROOT.RooFit.Components(plot_argset)
            plot_model.plotOn(
                frame,
                plot_comp,
                fit_range,
                fit_norm_range,
                ROOT.RooFit.LineStyle(ROOT.kDashed),
                ROOT.RooFit.LineColor(next(get_color))
            )
            leg.AddEntry(frame.findObject(plot_argset.GetName()), comp.GetTitle(), 'L')

        chi2 = frame.chiSquare(
            plot_model.GetName(),
            dataset.GetName(),
            len(plot_model.getParameters(dataset)),
        )

        pvalue = ROOT.Math.chisquared_cdf_c(frame.chiSquare(plot_model.GetName(),dataset.GetName()), len(plot_model.getParameters(dataset)))

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
        
        if legend:
            leg.Draw()

        chi2_text = ROOT.TLatex(0.65, 0.8, '#chi^{{2}}/ndf = {}'.format(round(chi2,1)))
        chi2_text.SetTextSize(0.07)
        chi2_text.SetNDC(ROOT.kTRUE)
        chi2_text.Draw()
        '''
        pvalue_text = ROOT.TLatex(0.65, 0.7, 'p = {}'.format(pvalue // 0.0001 / 10000))
        pvalue_text.SetTextSize(0.07)
        pvalue_text.SetNDC(ROOT.kTRUE)
        pvalue_text.Draw()
        '''
        if extra_text:
            text = ROOT.TLatex(0.65, 0.6, extra_text)
            text.SetTextSize(0.07)
            text.SetNDC(ROOT.kTRUE)
            text.Draw()

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
        
        if isinstance(output_filepath,Path):
            output_filepath = str(output_filepath)  
        path_stem, path_ext = output_filepath.rsplit('.',1)
        for fmt in file_formats:
            c.SaveAs(path_stem+'.'+fmt)
        c.Close()
