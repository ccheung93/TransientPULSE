"""
Waveform propagation and visualization functions.

Contains the core propagation physics for time-dependent scalar field waveforms
and spectrogram generation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from constants import *
from propagation import propagation_time
from utils.logging_utils import get_logger

logger = get_logger()

def propagation(spec, density_profile, m, d, K, ts_sec, N_points_spectrogram=None,
                time_step_range=None, N_time_steps=None, save_waveform=True):
    """
    Propagate scalar field waveforms through a galactic density profile.

    Takes a spectrum and propagates each momentum mode, accounting for dispersion
    and time delays due to the varying density profile.

    Args:
        spec ([np.ndarray, np.ndarray]): Spectrum (x=momentum, y=flux)
        density_profile ([np.ndarray, np.ndarray]): Density profile (x=position, y=density at position)
        m (float): mass [eV]
        d (float): dilatonic coupling [dimensionless]
        K (float): fraction of energy density from a particular particle [dimensionless]
        ts_sec (float): burst duration in seconds
        N_points_spectrogram (int, optional): Number of points for spectrogram
        time_step_range (tuple, optional): (t_start, t_end) for time window
        N_time_steps (int, optional): Number of time steps
        save_waveform (bool): If True, save waveform plot to 'waveform_plot.pdf'

    Returns:
        tuple: (t_duration, phi_signal, N_points, E, spectrogram, valid)
            - t_duration: Time array at Earth [s]
            - phi_signal: Scalar field amplitude at Earth
            - N_points: Number of time points
            - E: Energy array [eV]
            - spectrogram: 2D array (energy × time) of energy density
            - valid: Boolean mask of valid frequency bins
    """
    logger.info("Starting propagation calculation")

    start_time = time.time()

    t_exp_sec = YEAR
    t_exp_ineV = t_exp_sec*SEC_TO_INEV
    ts_eV = ts_sec*SEC_TO_INEV
    N_time_steps = N_time_steps if N_time_steps else 1

    # Spectrum
    p, A = spec
    del spec
    E = np.sqrt(m**2 + p**2)

    # Density profile
    x, rho = density_profile
    R = x[-1] # distance from the Earth to Galactic center [kpc] TODO - parametrize this

    t_arrivals = np.array([propagation_time(m, E_i, x, rho, K, d) for E_i in E])
    t_fastest_absolute = np.min(t_arrivals)
    t_slowest = np.max(t_arrivals) + ts_eV

    valid1 = True
    spectrogram_array = np.zeros((len(E), int(N_points_spectrogram)))

    if valid1:
        # Looking at factor of max_to_min_amp_ratio from max flux
        max_to_min_amp_ratio = 1e7
        valid = np.where((A >= max(A)/max_to_min_amp_ratio))[0]
    else:
        # Looking at part of the waveform that arrives in the lifetime of the experiment, e.g. the tail of the spectrum
        valid = np.where((t_arrivals - t_fastest_absolute) <= t_exp_ineV)[0]

    # Define valid time window
    delta_t_s = [time_step / N_time_steps for time_step in time_step_range] if time_step_range else [0, ts_eV]
    logger.debug(f"delta_t_s: {delta_t_s[0]} to {delta_t_s[-1]} s")
    t_fastest = np.min(t_arrivals[valid]) + delta_t_s[0]
    t_slowest = np.max(t_arrivals[valid]) + delta_t_s[1]
    t_d = t_slowest - t_fastest
    logger.debug(f"Arrival time: {(t_fastest/SEC_TO_INEV)/3e7:.6f} years")

    N_points = min(int(2*t_d * np.max(E[valid]/(2*np.pi))), 1e4)
    logger.debug(f"Calculated N_points={int(2*t_d * np.max(E[valid]/(2*np.pi)))}, using N_points={N_points}")

    t_duration = np.linspace(t_fastest, t_slowest, int(N_points_spectrogram))
    t_duration_Earth_s = (t_duration - t_fastest_absolute)/SEC_TO_INEV

    phi_t_final = np.zeros(len(t_duration))

    delta_t_duration = t_d/len(t_duration)

    # Define indices for all bin starts
    i0 = valid[:-2]
    i1 = valid[1:-1]

    p_avg = (p[i0] + p[i1]) / 2
    A_avg = (A[i0] + A[i1]) / 2
    E_avg = np.sqrt(m**2 + p_avg**2)

    # Time window for each momentum mode
    t_start = t_arrivals[i1] + delta_t_s[0]     # Defined by fastest momentum mode in the momentum bin
    t_end = t_arrivals[i0] + delta_t_s[1]       # Defined by the slowest momentum mode in the bin

    # Indices in the time array
    start_index = ((t_start - t_fastest_absolute) /(delta_t_duration)).astype(int)
    stop_index = ((t_end- t_fastest_absolute) / (delta_t_duration)).astype(int)

    # TODO - check R dependence
    for i, (start, stop, p_a, A_a, E_a) in enumerate(zip(start_index, stop_index, p_avg, A_avg, E_avg)):
        time_window = t_duration[start:stop]
        spectrogram_array[i][start:stop] += 0.5 * (E_a**2) * (A_a/R)**2
        phi = (A_a/R) * np.cos(E_a * time_window - p_a * R)
        phi_t_final[start:stop] +=  phi

    end_time = time.time()
    logger.info(f'Propagation completed in {end_time - start_time:.2f}s')

    # Optionally save waveform plot
    if save_waveform:
        plot_waveform(t_duration_Earth_s, phi_t_final, filename='waveform_plot.pdf')

    return t_duration_Earth_s, phi_t_final, N_points, E, spectrogram_array, valid


def plot_waveform(t_duration, phi_signal, filename='waveform_plot.pdf'):
    """
    Plot and save the scalar field waveform at Earth.

    Args:
        t_duration (np.ndarray): Time array [s]
        phi_signal (np.ndarray): Scalar field amplitude
        filename (str): Output filename for plot
    """
    start_time = time.time()
    logger.info(f'Saving waveform plot to {filename}')

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 35, 'font.family': 'STIXGeneral'})
    plt.rcParams['hatch.color'] = 'lightgray'

    fig, ax = plt.subplots(figsize=(30, 21))
    ax.plot(t_duration, phi_signal)
    ax.set_xlabel('Time (s)', fontsize=35)
    ax.set_ylabel(r'$\phi(t)$', fontsize=35)
    ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    end_time = time.time()
    logger.info(f'Saved {filename} in {end_time - start_time:.2f}s')


def plot_spectrogram(N_points, t_min, t_max, E, spectrogram_array, valid=None, filename='spectrogram_plot.pdf'):
    """
    Plot and save the frequency vs. time spectrogram.

    Args:
        N_points (int): Number of time points
        t_min (float): Minimum time [s]
        t_max (float): Maximum time [s]
        E (np.ndarray): Energy array [eV]
        spectrogram_array (np.ndarray): 2D array (energy × time) of energy density
        valid (np.ndarray, optional): Valid frequency mask (unused, kept for compatibility)
        filename (str): Output filename for plot
    """
    start_time = time.time()
    logger.info("Starting spectrogram generation")
    delta_t = (t_max - t_min) / N_points
    logger.debug(f'delta_t = {delta_t}, N={N_points}')
    logger.debug(f'Time range: [{t_min:.2e}, {t_max:.2e}] s')

    freq = E*SEC_TO_INEV/(2*PI)

    # Plot full spectrogram with full frequency range (like old code)
    # valid is used during propagation but we plot everything
    freq_plot = freq
    spectrogram_plot = spectrogram_array

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 40, 'font.family': 'STIXGeneral'})

    fig, ax5 = plt.subplots(figsize = (30, 21))
    cmap_name = 'viridis'
    cmap = plt.colormaps[cmap_name]
    rgb = plt.colormaps[cmap_name](0)
    cmap.set_bad(rgb)
    masked_data = np.ma.masked_where(spectrogram_plot <= 0, spectrogram_plot)

    im = ax5.imshow(masked_data,
                    aspect="auto",
                    origin="lower",
                    extent = [t_min, t_max, freq_plot[0], freq_plot[-1]],
                    norm=mcolors.LogNorm(),
                    cmap=cmap)
    cbar = fig.colorbar(im, label=r"$\rho_*~[{\rm eV}^4]$")
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(r"$\rho_*~[{\rm eV}^4]$", fontsize=40)

    ax5.set_xlabel("Time (s)", fontsize=40)
    ax5.set_ylabel(r"Frequency $f$ [Hz]", fontsize=40)
    ax5.tick_params(labelsize=40)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    end_time = time.time()
    logger.info(f"Saved {filename} in {end_time - start_time:.2f}s")
