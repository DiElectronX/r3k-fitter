import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from pathlib import Path
from uncertainties import unumpy
from pprint import pprint

ALLOWED_MODES = ['jpsi', 'psi2s']

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title

def jpsi_kstar_k_scan_plotter(data, path, show=False):
    metadata = data['metadata']
    score = data['score']
    r_jpsi_data = data['r_jpsi_data']
    r_jpsi_data_err = data['r_jpsi_data_err']
    r_jpsi_mc = data['r_jpsi_mc']
    r_jpsi_mc_err = data['r_jpsi_mc_err']

    fig, ax = plt.subplots(figsize=(8,8))
    pdata = ax.errorbar(score, r_jpsi_data, yerr=r_jpsi_data_err, ls='none', marker='.', color='blue')
    pmc = ax.errorbar(score, r_jpsi_mc, yerr=None, ls='--', marker='', color='orange')
    pmc_err = ax.fill_between(score, r_jpsi_mc-r_jpsi_mc_err, r_jpsi_mc+r_jpsi_mc_err, alpha=0.5, color='orange', edgecolor=None) 
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{B \rightarrow J/\psi (e^{+}e^{-}) K^{*}} \ / \ N_{B \rightarrow J/\psi (e^{+}e^{-}) K^{+}}$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    if metadata["mass_window"]:
        leg_title = f'${metadata["mass_window"][0]} < m_{{B}} < {metadata["mass_window"][1]}$ GeV' 
    else:
        leg_title = '$4.75 < m_{B} < 5.7$ GeV' 
    ax.legend([pdata,(pmc, pmc_err)], ['Data', 'MC $\pm 1 \sigma$'], title=leg_title, fontsize=16, title_fontsize=16)

    if show:
        fig.show()

    outpath = Path(path)
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def jpsi_resonant_scan_plotter(data, path, show=False):
    metadata = data['metadata']
    score = data['score']
    
    _n_jpsi_data_sig = unumpy.uarray(data['n_jpsi_data_sig'], data['n_jpsi_data_sig_err']) / data['n_jpsi_data_sig'][0]
    _n_jpsi_mc_sig = unumpy.uarray(data['n_jpsi_mc_sig'], data['n_jpsi_mc_sig_err']) / data['n_jpsi_mc_sig'][0]

    fig, ax = plt.subplots(figsize=(8,8))

    pdata = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_data_sig), yerr=unumpy.std_devs(_n_jpsi_data_sig), ls='none', marker='.', color='blue')
    pmc = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_mc_sig), yerr=unumpy.std_devs(_n_jpsi_mc_sig), ls='--', marker='', color='orange')
    pmc_err = ax.fill_between(score, unumpy.nominal_values(_n_jpsi_mc_sig)-unumpy.std_devs(_n_jpsi_mc_sig), unumpy.nominal_values(_n_jpsi_mc_sig)+unumpy.std_devs(_n_jpsi_mc_sig), alpha=0.5, color='orange', edgecolor=None) 
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{B \rightarrow J/\psi (e^{+}e^{-}) K^{+}}$ [A.U.]',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    if metadata["mass_window"]:
        leg_title = f'${metadata["mass_window"][0]} < m_{{B}} < {metadata["mass_window"][1]}$ GeV' 
    else:
        leg_title = '$4.75 < m_{B} < 5.7$ GeV' 
    ax.legend([pdata,(pmc, pmc_err)], ['Data', 'MC $\pm 1 \sigma$'], title=leg_title, fontsize=16, title_fontsize=16)

    if show:
        fig.show()

    outpath = Path(path)
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def jpsi_comp_scan_plotter(data, path, show=False, norm=True):
    metadata = data['metadata']
    score = data['score']
    _n_jpsi_data_sig = unumpy.uarray(data['n_jpsi_data_sig'],data['n_jpsi_data_sig_err'])
    _n_jpsi_data_comb = unumpy.uarray(data['n_jpsi_data_comb'],data['n_jpsi_data_comb_err'])
    _n_jpsi_data_kstar = unumpy.uarray(data['n_jpsi_data_kstar'],data['n_jpsi_data_kstar_err'])
    _n_jpsi_mc_sig = unumpy.uarray(data['n_jpsi_mc_sig'],1/np.sqrt(data['n_jpsi_mc_sig']))
    _n_jpsi_mc_kstar = unumpy.uarray(data['n_jpsi_mc_kstar'],1/np.sqrt(data['n_jpsi_mc_kstar']))

    if norm:
        _n_jpsi_data_sig = _n_jpsi_data_sig / _n_jpsi_data_sig.max()
        _n_jpsi_data_comb = _n_jpsi_data_comb / _n_jpsi_data_comb.max()
        _n_jpsi_data_kstar = _n_jpsi_data_kstar / _n_jpsi_data_kstar.max()
        _n_jpsi_mc_sig = _n_jpsi_mc_sig / _n_jpsi_mc_sig.max()
        _n_jpsi_mc_kstar = _n_jpsi_mc_kstar / _n_jpsi_mc_kstar.max()

    fig, ax = plt.subplots(figsize=(8,8))
    psig = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_data_sig), yerr=unumpy.std_devs(_n_jpsi_data_sig), ls='none', marker='.', color='blue')
    pcomb = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_data_comb), yerr=unumpy.std_devs(_n_jpsi_data_comb), ls='none', marker='.', color='orange')
    pkstar = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_data_kstar), yerr=unumpy.std_devs(_n_jpsi_data_kstar), ls='none', marker='.', color='green')
    psig_mc = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_mc_sig), yerr=unumpy.std_devs(_n_jpsi_mc_sig), ls='--', marker='', color='blue')
    pkstar_mc = ax.errorbar(score, unumpy.nominal_values(_n_jpsi_mc_kstar), yerr=unumpy.std_devs(_n_jpsi_mc_kstar), ls='--', marker='', color='green')
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{Events}$ [A.U.]' if norm else r'$N_{Events}$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    if metadata["mass_window"]:
        leg_title = f'${metadata["mass_window"][0]} < m_{{B}} < {metadata["mass_window"][1]}$ GeV' 
    else:
        leg_title = '$4.75 < m_{B} < 5.7$ GeV' 
    ax.legend(
        [
            'Data',
            psig,
            pcomb,
            pkstar,
            'MC',
            psig_mc,
            pkstar_mc
        ], [
            '',
            r'$N_{Signal}$', 
            '$N_{Comb. Bkg.}$', 
            '$N_{K^{*} \ Bkg.}$',
            '',
            r'$N_{B^{+} \rightarrow J/\psi K^{+}}$', 
            r'$N_{B \rightarrow J/\psi K^{*}}$',
        ], 
        title=leg_title, fontsize=16, title_fontsize=16,handler_map={str: LegendTitle({'fontsize': 16})}
    )

    if show:
        fig.show()

    outpath = Path(path)
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def psi2s_kstar_k_scan_plotter(data, path, show=False):
    metadata = data['metadata']
    score = data['score']
    r_psi2s_data = data['r_psi2s_data']
    r_psi2s_data_err = data['r_psi2s_data_err']
    r_psi2s_mc = data['r_psi2s_mc']
    r_psi2s_mc_err = data['r_psi2s_mc_err']

    fig, ax = plt.subplots(figsize=(8,8))
    pdata = ax.errorbar(score, r_psi2s_data, yerr=r_psi2s_data_err, ls='none', marker='.', color='blue')
    pmc = ax.errorbar(score, r_psi2s_mc, yerr=None, ls='--', marker='', color='orange')
    pmc_err = ax.fill_between(score, r_psi2s_mc-r_psi2s_mc_err, r_psi2s_mc+r_psi2s_mc_err, alpha=0.5, color='orange', edgecolor=None) 
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{B \rightarrow \psi(2s) (e^{+}e^{-}) K^{*}} \ / \ N_{B \rightarrow \psi(2s) (e^{+}e^{-}) K^{+}}$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    if metadata["mass_window"]:
        leg_title = f'${metadata["mass_window"][0]} < m_{{B}} < {metadata["mass_window"][1]}$ GeV' 
    else:
        leg_title = '$4.75 < m_{B} < 5.7$ GeV' 
    ax.legend([pdata,(pmc, pmc_err)], ['Data', 'MC $\pm 1 \sigma$'], title=leg_title, fontsize=16, title_fontsize=16)

    if show:
        fig.show()

    outpath = Path(path)
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def psi2s_comp_scan_plotter(data, path, show=False, norm=True):
    metadata = data['metadata']
    score = data['score']
    _n_psi2s_data_sig = unumpy.uarray(data['n_psi2s_data_sig'],data['n_psi2s_data_sig_err'])
    _n_psi2s_data_comb = unumpy.uarray(data['n_psi2s_data_comb'],data['n_psi2s_data_comb_err'])
    _n_psi2s_data_kstar = unumpy.uarray(data['n_psi2s_data_kstar'],data['n_psi2s_data_kstar_err'])
    _n_psi2s_mc_sig = unumpy.uarray(data['n_psi2s_mc_sig'],np.divide(1, np.sqrt(data['n_psi2s_mc_sig']), out=np.zeros_like(data['n_psi2s_mc_sig']), dtype=float, where=data['n_psi2s_mc_sig']!=0))
    _n_psi2s_mc_kstar = unumpy.uarray(data['n_psi2s_mc_kstar'],np.divide(1, np.sqrt(data['n_psi2s_mc_kstar']), out=np.zeros_like(data['n_psi2s_mc_kstar']), dtype=float, where=data['n_psi2s_mc_kstar']!=0))

    if norm:
        _n_psi2s_data_sig = _n_psi2s_data_sig / _n_psi2s_data_sig.max()
        _n_psi2s_data_comb = _n_psi2s_data_comb / _n_psi2s_data_comb.max()
        _n_psi2s_data_kstar = _n_psi2s_data_kstar / _n_psi2s_data_kstar.max()
        _n_psi2s_mc_sig = _n_psi2s_mc_sig / _n_psi2s_mc_sig.max()
        _n_psi2s_mc_kstar = _n_psi2s_mc_kstar / _n_psi2s_mc_kstar.max()

    fig, ax = plt.subplots(figsize=(8,8))
    psig = ax.errorbar(score, unumpy.nominal_values(_n_psi2s_data_sig), yerr=unumpy.std_devs(_n_psi2s_data_sig), ls='none', marker='.', color='blue')
    pcomb = ax.errorbar(score, unumpy.nominal_values(_n_psi2s_data_comb), yerr=unumpy.std_devs(_n_psi2s_data_comb), ls='none', marker='.', color='orange')
    pkstar = ax.errorbar(score, unumpy.nominal_values(_n_psi2s_data_kstar), yerr=unumpy.std_devs(_n_psi2s_data_kstar), ls='none', marker='.', color='green')
    psig_mc = ax.errorbar(score, unumpy.nominal_values(_n_psi2s_mc_sig), yerr=unumpy.std_devs(_n_psi2s_mc_sig), ls='--', marker='', color='blue')
    pkstar_mc = ax.errorbar(score, unumpy.nominal_values(_n_psi2s_mc_kstar), yerr=unumpy.std_devs(_n_psi2s_mc_kstar), ls='--', marker='', color='green')
    ax.set_xlabel('BDT Score',loc='right', fontsize=18)
    ax.set_ylabel(r'$N_{Events}$ [A.U.]' if norm else r'$N_{Events}$',loc='top',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)

    if metadata["mass_window"]:
        leg_title = f'${metadata["mass_window"][0]} < m_{{B}} < {metadata["mass_window"][1]}$ GeV' 
    else:
        leg_title = '$4.75 < m_{B} < 5.7$ GeV' 
    ax.legend(
        [
            'Data',
            psig,
            pcomb,
            pkstar,
            'MC',
            psig_mc,
            pkstar_mc
        ], [
            '',
            r'$N_{Signal}$', 
            '$N_{Comb. Bkg.}$', 
            '$N_{K^{*} \ Bkg.}$',
            '',
            r'$N_{B^{+} \rightarrow J/\psi K^{+}}$', 
            r'$N_{B \rightarrow J/\psi K^{*}}$',
        ], 
        title=leg_title, fontsize=16, title_fontsize=16,handler_map={str: LegendTitle({'fontsize': 16})}
    )

    if show:
        fig.show()

    outpath = Path(path)
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)

def main(args):
    if args.mode=='all':
        jpsi_data_file = Path('.') / 'jpsi_kstar_k_scan_data.pkl'
        assert jpsi_data_file.is_file(), 'Cannot find jpsi data file'
        jpsi_output = Path('.') / 'plots' / 'jpsi_kstar_k_scan.pdf'
        if args.label:
            jpsi_output = jpsi_output.with_name('_'.join([str(jpsi_output.stem), args.label])+jpsi_output.suffix)
        with open(jpsi_data_file, 'rb') as f:
            jpsi_score_data = pickle.load(f)
        jpsi_kstar_k_scan_plotter(jpsi_score_data, jpsi_output)
        jpsi_comp_scan_plotter(jpsi_score_data, str(jpsi_output).replace('kstar_k','comp'))

        psi2s_data_file = Path('.') / 'psi2s_kstar_k_scan_data.pkl'
        assert psi2s_data_file.is_file(), 'Cannot find psi2s data file'
        psi2s_output = Path('.') / 'plots' / 'psi2s_kstar_k_scan.pdf'
        if args.label:
            psi2s_output = psi2s_output.with_name('_'.join([str(psi2s_output.stem), args.label])+psi2s_output.suffix)
        with open(psi2s_data_file, 'rb') as f:
            psi2s_score_data = pickle.load(f)
        psi2s_kstar_k_scan_plotter(psi2s_score_data, psi2s_output)
        psi2s_comp_scan_plotter(psi2s_score_data, str(psi2s_output).replace('kstar_k','comp'))

    elif args.mode=='jpsi':
        jpsi_data_file = Path('.') / 'jpsi_kstar_k_scan_data.pkl'
        assert jpsi_data_file.is_file(), 'Cannot find jpsi data file'
        jpsi_output = Path('.') / 'plots' / 'jpsi_kstar_k_scan.pdf'
        if args.label:
            jpsi_output = jpsi_output.with_name('_'.join([str(jpsi_output.stem), args.label])+jpsi_output.suffix)
        with open(jpsi_data_file, 'rb') as f:
            jpsi_score_data = pickle.load(f)
        jpsi_kstar_k_scan_plotter(jpsi_score_data, jpsi_output)
        jpsi_resonant_scan_plotter(jpsi_score_data, str(jpsi_output).replace('kstar_k','resonant'))
        jpsi_comp_scan_plotter(jpsi_score_data, str(jpsi_output).replace('kstar_k','comp'))

    elif args.mode=='psi2s':
        psi2s_data_file = Path('.') / 'psi2s_kstar_k_scan_data.pkl'
        assert psi2s_data_file.is_file(), 'Cannot find psi2s data file'
        psi2s_output = Path('.') / 'plots' / 'psi2s_kstar_k_scan.pdf'
        if args.label:
            psi2s_output = psi2s_output.with_name('_'.join([str(psi2s_output.stem), args.label])+psi2s_output.suffix)
        with open(psi2s_data_file, 'rb') as f:
            psi2s_score_data = pickle.load(f)

        psi2s_kstar_k_scan_plotter(psi2s_score_data, psi2s_output)
        psi2s_comp_scan_plotter(psi2s_score_data, str(psi2s_output).replace('kstar_k','comp'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True, choices=['all']+ALLOWED_MODES, help='which scan to perform')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    args, _ = parser.parse_known_args()

    main(args)
