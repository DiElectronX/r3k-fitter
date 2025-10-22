import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, approx_fprime
from scipy.stats import t


def confidence_band(model, xdata, ydata, popt, pcov, x_eval, confidence_level=0.6827):
    """
    Compute confidence bands for fitted model.
    - xdata, ydata: data actually used in the fit
    - popt, pcov: fit results
    - x_eval: points where the band should be evaluated
    """
    ndata = len(xdata)
    npars = len(popt)

    # residuals on fit data
    residuals = ydata - model(xdata, *popt)
    mse = np.sum(residuals**2) / (ndata - npars)

    # Jacobian on evaluation points
    def model_p(p, z):
        return model(z, *p)

    jac = []
    for z in x_eval:
        dp = approx_fprime(popt, model_p, 1e-6, z)
        jac.append(dp)
    jac = np.array(jac)

    # variance of predictions
    pr_var = np.einsum("ij,jk,ik->i", jac, pcov, jac)

    # t-distribution score
    rtail = 0.5 + confidence_level / 2.0
    score = t.ppf(rtail, ndata - npars)

    delta = score * np.sqrt(pr_var * mse)

    y_pred = model(x_eval, *popt)
    return y_pred, y_pred - delta, y_pred + delta



def poly(x, x0, A, B):
    return A * (x - x0) ** 2 + B


def significance_plotter(data, path, add_fitline=False, add_maxline=True,
                        datarange=None, fitrange=None, show=False):
    scores = data['score']
    sigs = data['significance']
    sig_errs = data['significance_err']

    # mask by datarange
    data_range_mask = (scores > datarange[0]) & (scores < datarange[1]) if datarange else np.ones_like(scores, dtype=bool)
    scores = scores[data_range_mask]
    sigs = sigs[data_range_mask]
    sig_errs = sig_errs[data_range_mask]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(scores, sigs, yerr=sig_errs, label='Scan Working Points',
                ls='none', marker='o')

    # --- Fit section ---
    if add_fitline:
        # restrict to fit range
        fit_range_mask = (scores > fitrange[0]) & (scores < fitrange[1]) if fitrange else np.ones_like(scores, dtype=bool)
        _scores = scores[fit_range_mask]
        _sigs = sigs[fit_range_mask]
        _sig_errs = sig_errs[fit_range_mask]

        # drop NaN or inf values
        finite_mask = np.isfinite(_scores) & np.isfinite(_sigs) & np.isfinite(_sig_errs)
        _scores = _scores[finite_mask]
        _sigs = _sigs[finite_mask]
        _sig_errs = _sig_errs[finite_mask]


        # initial guess: center at mean, parabola opening upward, baseline at mean y
        p0 = [np.mean(_scores), -1.0, np.mean(_sigs)]

        try:
            popt, pcov = curve_fit(poly, _scores, _sigs, sigma=_sig_errs,
                                   p0=p0, maxfev=20000)
            x_fit = np.linspace(min(_scores), max(_scores), 500)
            y_fit, y_low, y_up = confidence_band(poly, _scores, _sigs, popt, pcov, x_fit)

            ax.plot(x_fit, y_fit, color='red', label='Parabolic Fit')
            ax.fill_between(x_fit, y_low, y_up, color='red', alpha=0.3, label=r'$\pm 1\sigma$ band')

            # draw maximum line from fit
            if add_maxline:
                x0_fit = popt[0]
                y0_fit = poly(x0_fit, *popt)
                ax.axvline(x0_fit, color='red', linestyle='--',
                           label=f'Optimized BDT Cut ({x0_fit:.2f})')
                ax.axhline(y0_fit, color='red', linestyle='--',
                           label=f'Optimized Significance ({y0_fit:.2f})')
        except RuntimeError:
            print("Fit failed, falling back to max from data points")
            add_fitline = False  # fallback below

    # --- Maxline if no fit ---
    if add_maxline and not add_fitline:
        max_idx = np.argmax(sigs)
        ax.axvline(scores[max_idx], color='red', linestyle='--',
                   label=f'Optimized BDT Cut ({scores[max_idx]:.2f})')
        ax.axhline(sigs[max_idx], color='red', linestyle='--',
                   label=f'Optimized Significance ({sigs[max_idx]:.2f} ± {sig_errs[max_idx]:.2f})')

    ax.set_xlabel('BDT Score', loc='right', fontsize=18)
    ax.set_ylabel(r'Signal Significance $(\frac{N_{Sig}}{\sqrt{N_{Sig} + N_{Bkg}}})$',
                  loc='top', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right', fontsize=14)

    if show:
        plt.show()

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

    significance_plotter(score_data, output_file,
                         add_fitline=args.add_fitline,
                         datarange=args.datarange,
                         fitrange=args.fitrange)


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
    parser.add_argument('-fit', '--add-fitline', action='store_true',
        help='Fit a parabola to the significance scan')
    parser.add_argument('-fr', '--fitrange', nargs='+', dest='fitrange', 
        type=float, help='range for fitting function')
    args, _ = parser.parse_known_args()

    main(args)
