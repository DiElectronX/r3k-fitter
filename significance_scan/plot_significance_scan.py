import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, approx_fprime
from scipy.stats import t


def confidence_band(model, xdata, ydata, confidence_level=0.6827, **kwargs):
    # perform the fit
    popt, pcov = curve_fit(model, xdata, ydata, **kwargs)

    # some stuff needed below
    ndata = len(xdata)
    npars = len(popt)
    p0 = np.ones( npars )
    p0 = kwargs.get('p0', p0)
    sigma = np.ones( ndata )
    sigma = kwargs.get('sigma', sigma)
    absolute_sigma = kwargs.get('absolute_sigma', False)

    if not absolute_sigma :
        SSE = ydata - model(xdata, *popt)
        SSE = np.sum( ( SSE / sigma )**2 )
        MSE = SSE / (ndata - npars)

    x = np.asarray(xdata)

    # mean predicted response
    pr_mean = model(x, *popt)

    # compute jacobian around popt
    def model_p(p, z):
        return model(z, *p)

    npoints = len(x)
    jac = np.array([])
    jac_shape = (npoints, npars)

    for z in x :
        dp = approx_fprime(popt, model_p, 10e-6, z)
        jac = np.append(jac, dp)

    jac = np.reshape(jac, jac_shape)
    jac_transposed = np.transpose(jac)

    # compute predicted response variance
    # optimized way to do equivalently
    # np.diag(np.dot(jac, np.dot(pcov, jac_tranposed) )
    pr_var = np.dot(pcov, jac_transposed)
    pr_var = pr_var * jac_transposed
    pr_var = np.sum(pr_var, axis=0)

    # estimate variance with MSE
    pr_var = MSE * pr_var

    # stat factors
    rtail = .5 + confidence_level / 2.
    dof = ndata - npars
    score = t.ppf(rtail, dof)

    pr_band = score * np.sqrt( pr_var )
    central = pr_mean
    upper = pr_mean + pr_band
    lower = pr_mean - pr_band

    return upper, lower


def lorentzian(x, x0, gamma, A, B):
    return B + (2 * A / (np.pi * gamma)) / (1 + ((x - x0) / gamma) ** 2)

def poly(x, x0, A, B):
    return A * (x-x0)**2 + B

def significance_plotter(data, path, add_fitline=False, add_maxline=True, datarange=None, fitrange=None, show=False):
    scores = data['score']

    data_range_mask = np.logical_and(scores>datarange[0], scores<datarange[1]) if datarange else np.ones_like(scores, dtype=np.bool)
    scores = scores[data_range_mask]
    sigs = data['significance'][data_range_mask]
    sig_errs = data['significance_err'][data_range_mask]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.errorbar(scores, sigs, yerr=sig_errs, label='Scan Working Points', ls='none', marker='o')
    
    # Fit
    if add_fitline:
        x_fit = np.linspace(min(scores), max(scores), 1000)
        initial_guess = [5.0, -0.7, 6]
       
        fit_range_mask =  np.logical_and(scores>fitrange[0], scores<fitrange[1]) if fitrange else np.ones_like(scores, dtype=int)
        _scores = scores[fit_range_mask]
        _sigs = sigs[fit_range_mask]
        _sig_errs = sig_errs[fit_range_mask]

        try:
            fit_fcn = poly
            fcn_params, fcn_cov = curve_fit(fit_fcn, _scores, _sigs, sigma=_sig_errs, p0=initial_guess, maxfev=100000)
            fcn_fit_up, fcn_fit_down = confidence_band(fit_fcn, _scores, _sigs, sigma=_sig_errs, confidence_level=0.6827)
            fcn_fit = fit_fcn(x_fit, *fcn_params)
            ax.plot(x_fit, fcn_fit, color='red', label='Fit')
            ax.fill_between(_scores, fcn_fit_down, fcn_fit_up, color='red', alpha=0.5, edgecolor='none', label='$\pm 1$ Std Dev')
        except RuntimeError:
           print('Fit issue, dropping fit line') 

    if add_maxline:
        if add_fitline:
            pass
        else:
            max_idx = np.argmax(sigs)
            ax.axvline(scores[np.argmax(sigs)], color='red', linestyle='--', label=f'Optimized BDT Cut ({round(scores[np.argmax(sigs)],2)})')
            ax.axhline(sigs[max_idx], color='red', linestyle='--', label=f'Optimized Significance ({round(sigs[max_idx],2)} $\pm$ {round(sig_errs[max_idx],2)})')

    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'Signal Significance $(\frac{N_{Sig}}{\sqrt{N_{Sig} + N_{Bkg}}})$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right',fontsize=16)
    
    if show:
        fig.show()

    fig.savefig(path, bbox_inches='tight')

def main(args):
    if args.input_file:
        data_file = Path(args.input_file)
        assert data_file.is_file(), 'Cannot find data file'
    else:
        data_file = Path('.') / 'significance_scan_data.pkl'
        assert data_file.is_file(), 'Cannot find data file'

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('.') / 'significance_scan.pdf'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    with open(data_file, 'rb') as f:
        score_data = pickle.load(f)

    significance_plotter(score_data, output_file, datarange=args.datarange, fitrange=args.fitrange)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='input_file', 
        type=str, help='pickle data file')
    parser.add_argument('-o', '--output', dest='output', 
        type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    parser.add_argument('-dr', '--datarange', nargs='+', dest='datarange', 
        type=float, help='range for data')
    parser.add_argument('-fr', '--fitrange', nargs='+', dest='fitrange', 
        type=float, help='range for fitting function')
    args, _ = parser.parse_known_args()

    main(args)
