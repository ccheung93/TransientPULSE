"""
Waveform propagation and visualization functions.

Contains the core propagation physics for time-dependent scalar field waveforms
and spectrogram generation.
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.constants import *
from calculations.constraints import propagation_time, propagation_time_GW
from utils.logging_utils import get_logger

logger = get_logger()

def propagation(spec, density_profile, m, d, K, ts_sec, N_points_spectrogram=None,
                time_step_range=None, N_time_steps=None, save_waveform=True, global_time_range=None):
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
        time_step_range (tuple, optional): (t_start, t_end) for emission time window at source
        N_time_steps (int, optional): Number of time steps
        save_waveform (bool): If True, save waveform plot to 'waveform_plot.pdf'
        global_time_range (tuple, optional): (t_min, t_max) global time range for aligned spectrograms

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
    E = np.sqrt(m**2 + p**2)

    # Density profile
    x, rho = density_profile
    R = x[-1] - x[0]

    t_arrivals = np.array([propagation_time(m, E_i, x, rho, K, d) for E_i in E])
    t_fastest_absolute = propagation_time_GW(R)

    valid1 = True
    
    # Modes are screened if m_eff > E anywhere along the path.
    # The worst case is at maximum density, so check against that.
    beta_max = (8*PI/PLANCK_MASS_EV**2) * d * K * np.max(rho)
    not_screened = E**2 > m**2 + beta_max

    if valid1:
        # Looking at factor of max_to_min_amp_ratio from max flux
        max_to_min_amp_ratio = 1e7
        valid = np.where((A >= max(A)/max_to_min_amp_ratio) & not_screened)[0]
    else:
        # Looking at part of the waveform that arrives in the lifetime of the experiment, e.g. the tail of the spectrum
        valid = np.where(((t_arrivals - t_fastest_absolute) <= t_exp_ineV) & not_screened)[0]

    # Define emission time window at source
    # time_step_range already contains the specific time window for this emission step
    delta_t_s = time_step_range if time_step_range else [0, ts_eV]
    logger.debug(f"delta_t_s: {delta_t_s[0]} to {delta_t_s[-1]} s")

    # Use global time range if provided (for time-varying emissions), otherwise compute locally
    if global_time_range is not None:
        t_fastest, t_slowest = global_time_range
        logger.debug(f"Using global time range: {t_fastest} to {t_slowest}")
    else:
        t_fastest = t_fastest_absolute
        t_slowest = np.max(t_arrivals) + ts_eV
        logger.debug(f"Computed local time range: {t_fastest} to {t_slowest}")

    t_d = t_slowest - t_fastest
    logger.debug(f"Arrival time: {(t_fastest/SEC_TO_INEV)/3e7:.6f} years")

    # Default N_spectrogram points if not defined
    N_points_spectrogram = N_points_spectrogram if N_points_spectrogram else int(np.median([1e3, t_d/ts_eV, 1e7]))

    t_duration = np.linspace(t_fastest, t_slowest, int(N_points_spectrogram))
    t_duration_Earth_s = (t_duration - t_fastest_absolute)/SEC_TO_INEV

    phi_t_final = np.zeros(len(t_duration))

    # Define indices for all adjacent bin pairs
    i0 = valid[:-1]
    i1 = valid[1:]

    p_avg = (p[i0] + p[i1]) / 2
    A_avg = (A[i0] + A[i1]) / 2
    E_avg = np.sqrt(m**2 + p_avg**2)

    # Time window for each momentum mode
    t_start = t_arrivals[i1] + delta_t_s[0]     # Defined by fastest momentum mode in the momentum bin
    t_end = t_arrivals[i0] + delta_t_s[1]       # Defined by the slowest momentum mode in the bin

    # Width of momentum bin
    delta_p = p[1] - p[0]
    
    spec_value = 1/(4*PI*R**2 * ts_eV) * delta_p
    
    spectrogram_array = np.zeros((len(E), int(N_points_spectrogram)))
    
    for i, (i0_idx, p_a, A_a, E_a) in enumerate(zip(i0, p_avg, A_avg, E_avg)):
        time_mask = np.where((t_duration>t_start[i]) & (t_duration<t_end[i]))
        time_window = t_duration[time_mask]

        if len(time_mask[0]) > 0:
            spectrogram_array[i0_idx][time_mask] += spec_value * (A_a*E_a/p_a)
        else:
            # Assign contribution to the nearest time bin so the spectrogram has no gaps.
            nearest = np.clip(np.searchsorted(t_duration, t_start[i]), 0, len(t_duration) - 1)
            spectrogram_array[i0_idx][nearest] += spec_value * (A_a*E_a/p_a)
        phi = (A_a/R) * np.cos(E_a * time_window - p_a * R)
        phi_t_final[time_mask] +=  phi

    end_time = time.time()
    logger.info(f'Propagation completed in {end_time - start_time:.2f}s')

    # Optionally save waveform plot
    if save_waveform:
        plot_waveform(t_duration_Earth_s, phi_t_final, filename='waveform_plot.pdf')
        
    return t_duration_Earth_s, phi_t_final, N_points_spectrogram, E, spectrogram_array, valid


def plot_waveform(t_duration, phi_signal, filename='plots/waveform_plot.pdf'):
    """
    Plot and save the scalar field waveform at Earth.

    Args:
        t_duration (np.ndarray): Time array [s]
        phi_signal (np.ndarray): Scalar field amplitude
        filename (str): Output filename for plot
    """
    start_time = time.time()
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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


def plot_spectrogram(N_points, t_min, t_max, E, spectrogram_array, cutoff_min=None, cutoff_max=None, filename='plots/spectrogram_plot.pdf', show=False):
    """
    Plot and save the frequency vs. time spectrogram.

    Args:
        N_points (int): Number of time points
        t_min (float): Minimum time [s]
        t_max (float): Maximum time [s]
        E (np.ndarray): Energy array [eV]
        spectrogram_array (np.ndarray): 2D array (energy × time) of energy density
        filename (str): Output filename for plot
    """
    start_time = time.time()
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info("Starting spectrogram generation")
    delta_t = (t_max - t_min) / N_points
    logger.debug(f'delta_t = {delta_t}, N={N_points}')
    logger.debug(f'Time range: [{t_min:.2e}, {t_max:.2e}] s')

    freq = E*SEC_TO_INEV/(2*PI)
    t_duration = np.linspace(t_min, t_max, N_points)

    # Plot full spectrogram with full frequency range (like old code)
    # valid is used during propagation but we plot everything
    freq_plot = freq
    spectrogram_plot = spectrogram_array

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 40, 'font.family': 'STIXGeneral'})

    fig, ax = plt.subplots(figsize = (30, 21))
    X, Y = np.meshgrid(t_duration, freq_plot)
    cmap_name = 'viridis'
    cmap = plt.colormaps[cmap_name]
    rgb = plt.colormaps[cmap_name](0)
    cmap.set_bad(rgb)
    masked_data = np.ma.masked_where(spectrogram_plot <= 0, spectrogram_plot)

    im = ax.pcolormesh(X, Y,
                        masked_data,
                        shading='nearest',
                        norm=mcolors.LogNorm(),
                        cmap=cmap,
                        rasterized = True)
    
    cbar = fig.colorbar(im, label=r"$\rho_*~[{\rm eV}^4]$")
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label(r"$\rho_*~[{\rm eV}^4]$", fontsize=40)

    ax.set_xlabel("Time (s)", fontsize=40)
    ax.set_ylabel(r"Frequency $f$ [Hz]", fontsize=40)
    ax.tick_params(labelsize=40)
    
    # Calculate densities
    cutoff_min = cutoff_min if cutoff_min else 0
    cutoff_max = cutoff_max if cutoff_max else t_max
    _, rho_t, rho_f, rho_t_avg, f_avg, std_f = calc_densities(t_duration, spectrogram_array, freq, cutoff_min, cutoff_max)

    # Plot densities
    divider = mpl.axes_grid1.make_axes_locatable(ax)
    # below height and pad are in inches
    ax_x = divider.append_axes("top", 4, pad=0.1, sharex=ax)
    ax_y = divider.append_axes("right", 4, pad=0.1, sharey=ax)
    
    # make some labels invisible
    ax_x.xaxis.set_tick_params(labelbottom=False)
    ax_y.yaxis.set_tick_params(labelleft=False)
    
    ax_x.plot(t_duration[(t_duration <= cutoff_max)], rho_t, linewidth = 6,linestyle = '-',color = 'tab:blue')
    ax_x.plot([t_min, t_max],[rho_t_avg, rho_t_avg],linewidth = 6,linestyle = '--',color = 'tab:red')
    
    rho_t_min_nonzero = rho_t[np.where(rho_t > 0)].min()
    rho_t_max = rho_t.max()
    
    ax_x.set_yscale('log')
    ax_x.set_ylim(rho_t_min_nonzero,rho_t_max*10)
    ax_x.set_ylabel(r'$\rho(t)$')
    
    # ax_y.plot(y)
    ax_y.plot(rho_f,freq,linewidth = 6,linestyle = '-',color = 'tab:blue')
    ax_y.plot([0,rho_f.max()*2],[f_avg,f_avg],linewidth = 6,linestyle = '--',color = 'tab:red')
    ax_y.fill_between([0,rho_f.max()*2],[f_avg - std_f,f_avg - std_f],[f_avg + std_f,f_avg + std_f], color = 'tab:red',alpha = 0.2)
    ax_y.set_xscale('log')
    ax_y.set_xlabel(r'$\rho(f)$')
    

    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()

    end_time = time.time()
    logger.info(f"Saved {filename} in {end_time - start_time:.2f}s")
    
    arrival_window = [t_duration[np.nonzero(rho_t)][0], t_max]
    
    return rho_t_avg, arrival_window


def export_source_parameters(avg_density, burst_duration, R, mass, arrival_window=None,
                              coupling=None, K=None,
                              coupling_type='', coupling_order='',
                              to_file=False, filename=None):
    """Export source parameters for use in constraint plotting workflow

    Args:
        avg_density (float): Average energy density [eV^4]
        burst_duration (float): Burst duration [s]
        R (float): Distance [eV^-1 in natural units]
        mass (float): Scalar field mass [eV]
        arrival_window (tuple, optional): (t_min, t_max) arrival time window [s] for elongated shell calculation
        coupling (float, optional): Dilatonic coupling strength [dimensionless]
        K (float, optional): Energy density fraction
        coupling_type (str): Type of coupling ('photon', 'electron', 'gluon')
        coupling_order (str): Coupling order ('linear' or 'quad')
        to_file (bool): Whether to save to file
        filename (str, optional): Output filename

    Returns:
        dict: Dictionary of source parameters
    """
    from utils.constants import PC_TO_INEV, SOLAR_TO_EV, SEC_TO_INEV
    import numpy as np

    # Convert units for constraint plotting
    R_pc = R / PC_TO_INEV  # Convert from eV^-1 to parsecs
    tstar_sec = burst_duration  # Already in seconds

    # Calculate total energy from avg_density if available
    # E_tot = rho * Volume * duration
    # Volume ~ 4π R^2 * c * t_star (for spherical expansion)
    # Use arrival_window for elongated shell if provided, otherwise use burst_duration
    if avg_density is not None and R is not None:
        if arrival_window is not None:
            # Use arrival window to calculate elongated shell duration
            total_duration = arrival_window[1] - arrival_window[0]
        else:
            total_duration = burst_duration
        total_duration_nu = total_duration * SEC_TO_INEV  # Convert to natural units
        volume_nu = 4 * np.pi * R**2 * total_duration_nu  # Natural units
        Etot_eV = avg_density * volume_nu
        Etot_solar = Etot_eV / SOLAR_TO_EV
    else:
        Etot_solar = None

    params = {
        'MASS': mass,
        'DISTANCE': R_pc,
        'BURST_DURATION': tstar_sec,
        'ENERGY': Etot_solar,
        'COUPLING': coupling,
        'K': K,
        'COUPLING_TYPE': coupling_type,
        'COUPLING_ORDER': coupling_order,
        'AVG_DENSITY': avg_density
    }

    if to_file:
        filename = filename if filename else 'source.params'
        with open(filename, 'w') as f:
            for key, value in params.items():
                if value is not None:
                    f.write(f'{key}={value}\n')

    # Print summary
    print("Source Parameters:")
    for key, value in params.items():
        if value is not None:
            print(f'{key}={value}')

    return params


def create_source_from_propagation(avg_density, burst_duration, R, mass, arrival_window, 
                                     coupling_type='', coupling_order='',
                                     ULB_type=''):
    """Create a Source object from waveform propagation results

    This is a bridge function between the waveform propagation workflow (Part 1)
    and the constraint plotting workflow (Part 2).

    Args:
        avg_density (float): Average energy density [eV^4]
        burst_duration (float): Burst duration [s]
        R (float): Distance [eV^-1 in natural units]
        mass (float): Scalar field mass [eV]
        coupling_type (str): Type of coupling ('photon', 'electron', 'gluon')
        coupling_order (str): Coupling order ('linear' or 'quad')
        ULB_type (str): Type of ultralight boson ('scalar' or 'ALP')

    Returns:
        Source: Source object ready for constraint plotting

    Example:
        >>> # After waveform propagation
        >>> results = collection.propagate_all(N_points_spectrogram=2000)
        >>> R = collection.density_profile[0][-1] - collection.density_profile[0][0]
        >>> avg_density = plot_spectrogram(...)
        >>>
        >>> # Create Source for constraint plotting
        >>> source = create_source_from_propagation(
        ...     avg_density, physics.burst_duration, R, physics.mass,
        ...     coupling_type='photon', coupling_order='linear'
        ... )
    """
    from inputs.source import Source
    from utils.constants import PC_TO_INEV, SOLAR_TO_EV, SEC_TO_INEV
    import numpy as np

    # Convert units
    R_pc = R / PC_TO_INEV
    tstar_sec = burst_duration
    
    # Elongate shell
    total_duration = arrival_window[1] - arrival_window[0]

    # Calculate total energy
    burst_duration_nu = total_duration * SEC_TO_INEV
    volume_nu = 4 * np.pi * R**2 * burst_duration_nu
    Etot_eV = avg_density * volume_nu
    Etot_solar = Etot_eV / SOLAR_TO_EV

    # Create and return Source object
    source = Source(
        Etot=Etot_solar,
        mass=mass,
        tstar=tstar_sec,
        R=R_pc,
        ULB_type=ULB_type,
        coupling_type=coupling_type,
        coupling_order=coupling_order
    )

    return source


def load_source_from_file(filename='source.params', ULB_type=''):
    """Load a Source object from parameter file created by export_source_parameters

    Args:
        filename (str): Path to parameter file
        ULB_type (str): Type of ultralight boson ('scalar' or 'ALP')

    Returns:
        Source: Source object ready for constraint plotting

    Example:
        >>> # After exporting parameters
        >>> source = load_source_from_file('bosenova.param')
    """
    from inputs.source import Source
    import os

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Parameter file '{filename}' not found")

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value  # Keep as string if not numeric

    # Extract required parameters
    required = ['MASS', 'DISTANCE', 'BURST_DURATION', 'ENERGY', 'COUPLING_TYPE', 'COUPLING_ORDER']
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required parameters in file: {missing}")

    # Create Source object
    source = Source(
        Etot=params['ENERGY'],
        mass=params['MASS'],
        tstar=params['BURST_DURATION'],
        R=params['DISTANCE'],
        ULB_type=ULB_type,
        coupling_type=params['COUPLING_TYPE'],
        coupling_order=params['COUPLING_ORDER']
    )

    return source


def calc_densities(t_duration, spectrogram, freq, cutoff_min=None, cutoff_max=None):
    """
    
    """
    
    # Default cutoff values
    cutoff_min = cutoff_min if cutoff_min else 0
    cutoff_max = cutoff_max if cutoff_max else t_duration[-1]
    
    # Select portion of spectrogram within cutoff
    mask = (t_duration >= cutoff_min) & (t_duration <= cutoff_max)
    spectrogram = spectrogram[:, mask]
    
    rho_t = np.sum(spectrogram, axis=0)
    rho_f = np.sum(spectrogram, axis=1)
    
    delta_f = freq[1] - freq[0]
    rho_f_norm = rho_f / (np.sum(rho_f) * delta_f)
    f_avg = np.sum(freq * rho_f_norm) * delta_f
    w_avg = 2*PI*f_avg/SEC_TO_INEV
    rho_t_avg = np.mean(rho_t[rho_t > 0])
    std_f = np.sqrt(np.sum((freq-f_avg)**2 *rho_f_norm) * delta_f)
    print(f'frequency standard deviation = {std_f}')
    print(f'max density = {max(rho_t)}, {max(rho_f)}')
    print(f'avg density = {rho_t_avg}')
    
    return w_avg, rho_t, rho_f_norm, rho_t_avg, f_avg, std_f