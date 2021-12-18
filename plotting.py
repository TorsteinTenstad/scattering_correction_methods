from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
from pls import pls_mse


def draw_brakes(ax, x_pos, dx=0.005, dy=0.02):
    for a in [-0.005, 0.005]:
        pos = (x_pos + a)
        ax.plot([(pos-dx),(pos+dx)], [-dy,+dy], color='k', clip_on=False, transform=ax.transAxes)


cmap = cm.get_cmap('plasma')
set_colors = ['C5', 'C0', 'C2']

methods = ['Raw apparent absorption', 'Absorption after EMSC correction', 'Absorption coefficient estimated by optical model']
my_dpi = 140
default_figsize = (1920/my_dpi, 2160/my_dpi)


def single_sample_comparison(lambdas, concentrations, data_collections, i):
    fig, axs = plt.subplots(nrows=3, sharex='all', figsize=default_figsize)
    cs = [0, 13, 15]
    for ax, c in zip(axs, cs):
        raw_data = data_collections[0][0][c][i]
        EMSC_corrected = data_collections[1][0][c][i]
        optically_corrected = data_collections[2][0][c][i]
        ax2 = ax.twinx()
        ax.plot(lambdas*1e9, raw_data, label='raw_data', color=set_colors[0])
        ax.plot(lambdas*1e9, EMSC_corrected, label='EMSC_corrected', color=set_colors[1])
        ax2.plot(lambdas*1e9, optically_corrected, label='optically_corrected', color=set_colors[2])
        ax.text(0.35, 0.96, f'Limestone concentration: {concentrations[c]:.0f}%', horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=cmap(c/len(concentrations)),fc=(0.9, 0.9, 0.9),), size='x-large', transform=ax.transAxes)
        ax.set_ylabel('Absorbance [A.U.]')
        ax2.set_ylabel('Absorption coefficient [m^-1]')
    ax.set_xlabel('Wavelength [nm]')
    lines = [Line2D([0], [0], color=color, lw=4) for color in set_colors]
    axs[1].legend(lines, methods, loc='upper center', bbox_to_anchor=(0.35, 0.8))
    fig.subplots_adjust(top=0.989,
                        bottom=0.043,
                        left=0.058,
                        right=0.942,
                        hspace=0.0,
                        wspace=0.117)
    fig.savefig('single_sample_results.svg')


def plot_spectra(lambdas, concentrations, data_collections):
    n = len(data_collections)
    fig, axs = plt.subplots(nrows=n, sharex='all', figsize=default_figsize)
    fig.subplots_adjust(top=0.989,
                        bottom=0.041,
                        left=0.058,
                        right=0.988,
                        hspace=0.0,
                        wspace=0.2)
    for i, (ax, collection) in enumerate(zip(axs, data_collections)):
        training_set, test_set = collection
        ax.text(0.5, 0.96, methods[i], horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=set_colors[i],fc=(0.9, 0.9, 0.9),), size='x-large', transform=ax.transAxes)
        for c, spectra in enumerate(test_set):
            for spectrum in spectra:
                ax.plot(lambdas*1e9, spectrum, color=cmap(c/len(concentrations)), linewidth=0.5)
    axs[0].set_ylabel('Absorbance [A.U.]')
    axs[1].set_ylabel('Absorbance [A.U.]')
    axs[2].set_ylabel('Absorption coefficient [m^-1]')
    ax.set_xlabel('Wavelength [nm]')
    lines = [Line2D([0], [0], color=cmap(i/len(concentrations)), lw=4) for i, c in enumerate(concentrations)]
    labels = ['%.1f' % c for c in concentrations]
    axs[1].legend(lines, labels, loc='upper center', title='Limestone concentration [%]', ncol=8, bbox_to_anchor=(0.5, 0.8))
    fig.savefig('spectral_results.svg')


def plot_pls(concentrations, pls_results):
    n = len(pls_results)
    fig, axs = plt.subplots(nrows=n, sharex='all', figsize=default_figsize)
    fig.subplots_adjust(top=0.989,
                        bottom=0.041,
                        left=0.058,
                        right=0.988,
                        hspace=0.0,
                        wspace=0.2)
    for i, (ax, pls_result) in enumerate(zip(axs, pls_results)):
        cut = [27.5, 98.5]
        nudge = cut[1]-cut[0]
        ax.set_xticks([0, 5, 10, 15, 20, 25, 100-nudge])
        ax.set_xticklabels([0, 5, 10, 15, 20, 25, 100])
        lx = [np.min(concentrations), cut[0], cut[1]-nudge, np.max(concentrations)-nudge]
        ly = [np.min(concentrations), cut[0], cut[1], np.max(concentrations)]
        ax.plot(lx[0:2], ly[0:2], color='black')
        ax.plot(lx[2:4], ly[2:4], color='black')
        ax.text(0.04, 0.96, 'MSE = %.2f^2' % np.sqrt(pls_mse(concentrations, pls_result)[0]), horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2 ,ec=(0.6, 0.6, 0.6),fc=(0.95, 0.95, 0.95),), size='x-large', transform=ax.transAxes)
        ax.text(0.5, 0.96, f'PLSR based on {methods[i][0].lower()}{methods[i][1:]}', horizontalalignment='center', verticalalignment='top', bbox=dict(boxstyle="round", linewidth=2, ec=set_colors[i],fc=(0.9, 0.9, 0.9),), size='x-large', transform=ax.transAxes)
        for true_i, predicted in pls_result.items():
            
            true = concentrations[true_i]
            if true_i == len(pls_result.values())-1:
                true -= nudge
            ax.plot(true*np.ones_like(predicted), predicted, '.', color=cmap(true_i/len(concentrations)), ms=5)
            ax.plot(true, np.average(predicted), '_', color=cmap(true_i/len(concentrations)), ms=20)
        ax.set_ylabel('Predicted concentration [%]')
    draw_brakes(ax, 0.908)
    ax.set_xlabel('True concentration [%]')
    fig.savefig('pls_results.svg')

