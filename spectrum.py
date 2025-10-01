#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:07:21 2025

@author: arakawaj
"""

import numpy as np
import cmath
import math as math
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from IPython.core.display import HTML
import csv
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']
from matplotlib import rc
from matplotlib import ticker, cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, LogFormatterSciNotation 
from matplotlib import cm, colors
import matplotlib.image as mpimg

usetex: True
import time

import scipy
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian, hamming
from scipy.interpolate import make_interp_spline, BSpline

from decimal import Decimal, getcontext
import sys


# INEV_TO_METERS = 0.197e-18 * 1e9/3.086e13   # eV^-1 to parsecs
# eV_to_solar = (1.67e-27/2e30)*1e-9      # 1 eV in solar masses
# SEC_TO_INEV = 1e-9/(6.528e-25)          # 1 second in eV^-1
# sec_in_year = 3.154e7                   # number of seconds in 1 year
# c = 3e8                                 # speed of light in m/s
# GCM3_TO_EV4 = 4.247e18                  # g/cm^3 to eV^4
# vDM = 1e-3                              # Average velocity of galactic Dark Matter
# dt = 1 * 3.154e7
# PLANCK_MASS_EV = 1.2e28 #eV

INEV_TO_METERS = 1.97e-7                           # eV^-1 to meters
PC_TO_METERS = 3.086e16                            # parsecs to meters
PC_TO_INEV = PC_TO_METERS/INEV_TO_METERS           # parsecs to 1/eV
KPC_TO_INEV = 1000 * PC_TO_METERS/INEV_TO_METERS   # parsecs to 1/eV
GCM3_TO_EV4 = 4.2e18                               # g/cm^3 to eV^4
HBAR = 6.582e-16                                   # reduced Planck constant in eV*s
SEC_TO_INEV = 1/HBAR                               # 1 second to 1/eV
PLANCK_MASS_EV = 1.2e28                            # Planck mass in eV
PI = np.pi
YEAR = 3600*24*365

cmap = plt.get_cmap('viridis')

def read_medium_data(filename, i_R=0, i_rho=1):
    """ Read density (g/cm^3) vs. position (kpc) data from CSV file
    
    Args:
        filename (str): File name
        i_R (int): Column index for position data (kpc)
        i_rho (int): Column index for density data (g/cm^3)
    
    Returns:
        x (array): Position [eV^-1]
        rho (array): Density [eV^4]
    """
    x = []
    rho = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip header row
        for line in file:
            # Split by comma and remove empty fields
            parts = [p.strip() for p in line.split(',') if p.strip()]
            
            # Ensure we have enough columns
            if len(parts) > max(i_R, i_rho):
                try:
                    # Convert position and density
                    x.append(float(parts[i_R]) * KPC_TO_INEV)
                    rho.append(float(parts[i_rho]) * GCM3_TO_EV4)
                except (ValueError, IndexError):
                    continue  # Skip invalid lines
    
    return np.array(x), np.array(rho)

def read_spectrum_data(filename, i_w=0, i_A=1):
    """ Read spectrum data from CSV file
    
    Args:
        filename (str): File name
        i_R (int): Column index for position data (kpc)
        i_rho (int): Column index for density data (g/cm^3)
    
    Returns:
        x (array): Position [eV^-1]
        rho (array): Density [eV^4]
    """
    x = []
    y = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip header row
        for line in file:
            # Split by comma and remove empty fields
            parts = [p.strip() for p in line.split(',') if p.strip()]
            
            # Ensure we have enough columns
            if len(parts) > max(i_w, i_A):
                try:
                    # Convert position and density
                    x.append(float(parts[i_w]))
                    y.append(float(parts[i_A]))
                except (ValueError, IndexError):
                    continue  # Skip invalid lines
    
    return np.array(x), np.array(y)

def write_file(output_file,data,filetype):
    with open(output_file, 'w') as file:
        if filetype == 'csv':
            for i in range(len(data[0])):
                file.write(str(data[0][i])+',' + str(data[1][i])+'/n')
        if filetype == 'txt':
            for i in range(len(data[0])):
                file.write(str(data[0][i])+' ' + str(data[1][i])+'/n')
            

def interpolate_data(x,y,num):
    """ 
    Interpolates data, providing adjustable binning of input files
    
    Args:
        x (array of floats): Data in first column of file
        y (array of floats): Data in second column of file
        num (float): Number of points for interpolation
        
    Returns:
        xnew (array of floats): Redefined x values from interpolation (n points)
        ynew (array of floats): Redefined y values from interpolation (n points)
    """
    xnew = []
    ynew = []
        
    step = (math.log(x[-1],10)-math.log(x[0],10))/(num-1)
    xp = [x[0]*(10**((j*step))) for j in range(num)]
    xnew= np.array(xp)
    f = scipy.interpolate.interp1d(x,y,kind = 'linear',fill_value="extrapolate")
    ynew= np.array(f(xp))

    return xnew,ynew

def v_g(m, E, x, rho, K, d2):
    """ Calculates group velocity
        (using Eq. 20 in arXiv:2502.08716v1)
        
    Args:
        m (float): mass of phi [eV]
        E (array of floats): energies [eV]       # TO DO: cannot be array?
        x (array of floats): distance from source [eV^-1]
        rho (array of floats): SM density [eV^4]
        K (float): energy density fraction [unitless]
        d2 (float): quadratic dilatonic coupling [unitless]
        
    Returns:
        vg (float): group velocity
    """
    rho = rho * K
    
    beta = (8*PI)/PLANCK_MASS_EV**2 * d2 * rho
    meff = np.sqrt(m**2 + beta)
    q = E / meff
    return np.sqrt(1 - 1/q**2)

def t_prop(m, E, x, rho, K, d2):
    vg = v_g(m, E, x, rho, K, d2)
    temp = 0
    for i in range(len(vg)-1):
        x1, x2 = (x[i]), (x[i+1])
        # Trapezoidal Intgeral
        temp = temp + (0.5)*(1/vg[i+1] + 1/vg[i]) * (x2 - x1)
    return temp

def propagation(spec, density_profile, m, d, K, ts_sec):
    """
    Take a spectrum, and for each mode in it, propagate through the code:
    Read spec in as a csv file of form: momentum or energy , flux or power etc.
    First, determine the time it takes for the wave front to get to the detector:

    TO DO: Make times start at GW arrival. 
           Cutoff time at integration time of experiment. Probably no more than 1 year?
    Args:
        spec ([np.ndarray, np.ndarray]): Spectrum (x=momentum, y=flux)
        density_profile ([np.ndarray, np.ndarray]): Density profile (x=position, y=density at position)
        m (float): mass
        d (float): dilatonic coupling
        K (float): fraction of energy density comes from a particular particle
        ts_sec (float): burst duration in seconds

    Returns:
        t_duration (np.ndarray): array from 0 to duration of the signal at the Earth
        each_mode (np.ndarray): amplitudes of the waves
    """
    print("propagation started")
    
    start_time = time.time()
    
    t_exp_sec = YEAR
    t_exp_ineV = t_exp_sec*SEC_TO_INEV
    ts_eV = ts_sec*SEC_TO_INEV
    
    # Spectrum
    p, A = spec
    del spec
    E = np.sqrt(m**2 + p**2)
    
    # Density profile
    x, rho = density_profile
    R = x[-1] # distance from the Earth to Galactic center [kpc] TODO - parametrize this
    
    t_arrivals = np.array([t_prop(m, E_i, x, rho, K, d) for E_i in E])
    t_fastest = np.min(t_arrivals)
    t_slowest = np.max(t_arrivals) + ts_eV
    
    # TODO - check if there are "insane?" values in the spectrum
    # If there are, then we set a "valid" region:
    # if valid1:
    #     # Looking at factor of max_to_min_amp_ratio from max flux
    #     max_to_min_amp_ratio = 10
    #     valid = np.where((A >= max(A)/max_to_min_amp_ratio))[0]
    #

    valid1 = True
    if valid1:
        # Looking at factor of max_to_min_amp_ratio from max flux
        max_to_min_amp_ratio = 1000
        valid = np.where((A >= max(A)/max_to_min_amp_ratio))[0]
    else:
        # Looking at part of the waveform that arrives in the lifetime of the experiment, e.g. the tail of the spectrum
        valid = np.where((t_arrivals - t_fastest) <= t_exp_ineV)[0]
        
    # Plot valid region
    fig, ax = plt.subplots(1,1,figsize = (30,21), sharex = False,sharey = False)
    ax.plot(p[valid], A[valid])
    ax.set_xlabel('Momentum (eV)')
    ax.set_ylabel(r'$\phi(t)$')

    print('saving valid region of waveform...')
    plt.savefig('waveform_plot_valid.pdf', dpi=400)
    
    # Define valid time window
    t_fastest = np.min(t_arrivals[valid])
    t_slowest = np.max(t_arrivals[valid]) + ts_eV
    t_d = (t_slowest - t_fastest)
    
    p_fastest = np.max(p[valid])
    p_slowest = np.min(p[valid])
    
    print('time in years is:' + str((t_fastest/SEC_TO_INEV)/3e7))
    
    # TODO - Why is N_points shifting spectrogram signal downward?
    #N_points = int(100*t_d/(ts_sec*SEC_TO_INEV))
    N_points = int(10*t_d * np.max(p[valid]/(2*PI))) # 10 points per cycle
    #N_points = int(10 * t_exp_ineV * np.max(p[valid]))
    print(f"N_points={N_points}")
    
    t_duration = np.linspace(t_fastest, t_slowest, N_points)
    t_duration_Earth_s = (t_duration - t_fastest)/SEC_TO_INEV

    phi_t_final = np.zeros(len(t_duration))
    
    delta_t_duration = t_d/len(t_duration)
    
    # Define indices for all bin starts
    i0 = valid[:-2]
    i1 = valid[1:-1]
    
    p_avg = (p[i0] + p[i1]) / 2
    A_avg = (A[i0] + A[i1]) / 2
    E_avg = np.sqrt(m**2 + p_avg**2)
    
    # Time window for each momentum mode
    t_start = t_arrivals[i1]         # Defined by fastest momentum mode in the momentum bin
    t_end = t_arrivals[i0] + ts_eV   # Defined by the slowest momentum mode in the bin
    
    # Indices in the time array
    start_index = ((t_start - t_fastest) / delta_t_duration).astype(int)
    stop_index = ((t_end - t_fastest) / delta_t_duration).astype(int)
    
    for i, (start, stop, p_a, A_a, E_a) in enumerate(zip(start_index, stop_index, p_avg, A_avg, E_avg)):
        time_window = t_duration[start:stop]
        phi = A_a * np.cos(E_a * time_window - p_a * R)
        phi_t_final[start:stop] +=  phi

    # Plotting waveform
    fig, ax3 = plt.subplots(figsize = (30, 21))
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 35,'font.family':'STIXGeneral'})
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.rcParams['hatch.color'] = 'lightgray'
    plt.tight_layout()
    
    ax3.plot(t_duration_Earth_s, phi_t_final)
    ax3.set_xscale('log')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'$\phi(t)$')

    print('saving waveform_plot...')
    plt.savefig('waveform_plot.pdf', dpi=300)
    plt.close()
    end_time = time.time()
    print(f'TIMING >>> Time up to saving waveform_plot was {end_time - start_time:.2f}s.')
    
    ### Spectrogram
    start_time = time.time()
    print("spectrogram started") 
    delta_t = (t_duration[1] - t_duration[0]) / SEC_TO_INEV
    print(f'delta_t = {delta_t}, N={N_points}')
    
    # Scipy SFFT
    g_std = 5  # standard deviation for Gaussian window in samples
    hop = 10
    # w = gaussian(50, std=5, sym=True)  # symmetric Gaussian window
    w = hamming(1000, sym=True) # symmetric Hamming window
    SFT = ShortTimeFFT(w, hop=hop, fs=1/delta_t, scale_to='magnitude')
    Sx = SFT.spectrogram(phi_t_final)  # perform the STFT
    
    print(f"Size of spectrogram: {Sx.shape}")
    
    end_time = time.time()
    print(f"TIMING >>> Spectrogram took {end_time - start_time:.2f}s.")
    
    # Plot spectrogram
    start_time = time.time()
    n = len(phi_t_final)
    freqs = SFT.f
    times = SFT.t(n)
    times = times - times[0] + 1e-12 # shift so min(times) > 0
    
    fig, ax5 = plt.subplots(figsize = (30, 21))
    im = ax5.imshow(np.abs(Sx), 
                    aspect="auto", origin="lower",
                    #extent=[t_duration_Earth_s[0], t_duration_Earth_s[-1], freq[0], freq[-1]])
                    extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    fig.colorbar(im, label="Magnitude $|S_x(t, f)|$")
    ax5.set_yscale('linear')
    ax5.set_xscale('linear')
    ax5.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N_points)} slices, " +
                    rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                    rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
            # xlim = (t_duration_Earth_s[1], t_duration_Earth_s[-1])
            xlim = (times[0], times[-1])
            )
    plt.savefig('spectrogram_plot.pdf', bbox_inches='tight')
    plt.close()
    

    end_time = time.time()
    print(f"TIMING >>> Saved spectrogram_plot.pdf in {end_time - start_time:.2f}s.")
    
    return t_duration_Earth_s, phi_t_final


class SpectrumConfig:
    def __init__(self, spectrum_type='bosenova',
                 mass=1e-19, coupling=1e20, burst_duration=400,
                 K=1e-3, amplitude=1e20, peak_momentum=30, width=5,
                 density_profile_path='Galactic_Density_Profile.csv',
                 density_num_points=1000,
                 bosenova_path='Spectra/BosonStarSpectrumRelOnly.txt'):
        """_summary_

        Args:
            spectrum_type (str, optional): _description_. Defaults to 'bosenova'.
            mass (_type_, optional): _description_. Defaults to 1e-19.
            coupling (_type_, optional): _description_. Defaults to 1e20.
            burst_duration (int, optional): burst duration [s]. For bosenova, 400/mass converted to seconds 
            K (_type_, optional): _description_. Defaults to 1e-3.
            density_profile_path (str, optional): _description_. Defaults to 'Galactic_Density_Profile.csv'.
            bosenova_path (str, optional): _description_. Defaults to 'Spectra/BosonStarSpectrumRelOnly.txt'.
        """
        self.spectrum_type = spectrum_type
        self.mass = mass
        self.coupling = coupling
        self.burst_duration = burst_duration
        self.K = K
        self.density_profile_path = density_profile_path
        self.bosenova_path = bosenova_path
        self.density_num_points = density_num_points
        self.amplitude = amplitude
        self.peak_momentum = peak_momentum
        self.width = width

class SpectrumGenerator:
    def generate(self):
        raise NotImplementedError("Subclasses must implement generate()")
    
class BosenovaSpectrum(SpectrumGenerator):
    def __init__(self, filepath, mass, num_points=5000):
        self.filepath = filepath
        self.mass = mass
        self.num_points = num_points
    
    def generate(self):
        x_raw, y_raw = [], []
        with open(self.filepath, 'r') as f:
            for line in f:
                x_raw.append([float(value) for value in line.strip().split()][0])
                y_raw.append([float(value) for value in line.strip().split()][1])
        x_interp, y_interp = interpolate_data(x_raw, y_raw, self.num_points)
        
        momenta = x_interp * self.mass
        amplitudes = y_interp * self.mass
        
        return momenta, amplitudes

class GaussianSpectrum(SpectrumGenerator):
    def __init__(self, mass, burst_duration, peak_momentum, width, amplitude, num_points=10000):
        """_summary_

        Args:
            mass (_type_): _description_
            burst_duration (_type_): [s]
            peak_momentum (_type_): _description_
            width (_type_): _description_
            amplitude (_type_): _description_
            num_points (int, optional): _description_. Defaults to 10000.
        """
        self.mass = mass
        self.burst_duration = burst_duration
        self.peak_momentum = peak_momentum
        self.width = width
        self.amplitude = amplitude
        self.num_points = num_points
    
    def generate(self):
        tstar_ineV = self.burst_duration * SEC_TO_INEV
        
        dw = self.width
        p_peak = self.peak_momentum
        
        # momentum range 
        # p_initial = 0.7 * p_peak # 2 * PI / tstar_ineV
        # p_final = 1.3 * p_peak #100 / tstar_ineV
        p_initial = p_peak - dw*np.sqrt(np.log(10)) # 2 * PI / tstar_ineV
        p_final =  p_peak + dw*np.sqrt(np.log(10))
        momenta = np.linspace(p_initial, p_final, self.num_points)
        
        amplitudes = self.amplitude * np.exp(-((momenta-p_peak)**2)/(2*dw**2))
        
        # plot 
        fig, ax = plt.subplots(figsize = (30, 21))
        ax.plot(momenta, amplitudes)
        ax.set_xlabel('Momentum (eV)')
        ax.set_ylabel(r'$\phi(0)$')

        print('saving initial waveform...')
        plt.savefig(f'waveform_plot_initial_gaussian.pdf', dpi=300)
        plt.close()
        
        return momenta, amplitudes

def generate_spectrum(config):
    if config.spectrum_type == 'bosenova':
        return BosenovaSpectrum(config.bosenova_path, config.mass, num_points=5000)
    elif config.spectrum_type == 'gaussian':
        return GaussianSpectrum(config.mass, config.burst_duration, config.peak_momentum, config.width, config.amplitude)
    else:
        raise ValueError(f"Unknown spectrum type: {config.spectrum_type}")

def run_propagation(config):
    # Reading density profile from CSV file
    x, rho = read_medium_data(config.density_profile_path, i_R=0, i_rho=2)
    
    if config.spectrum_type == 'bosenova':
        x_interp, rho_interp = interpolate_data(x/10000, rho, config.density_num_points) # Converting 10 kpc to 1 pc
    else:
        x_interp, rho_interp = interpolate_data(x/1000, rho, config.density_num_points)
    density_profile = [x_interp, rho_interp]
    
    # Generate spectrum
    spectrum_generator = generate_spectrum(config)
    momenta, amplitudes = spectrum_generator.generate()
    spectrum = [momenta, amplitudes]
    
    # Save spectrum figure
    
    
    # Run propagation
    t_duration, phi_signal = propagation(spectrum, density_profile, config.mass, config.coupling, config.K, config.burst_duration)
    
    return t_duration, phi_signal

def save_spectrum_figure(spectrum):
    fig, ax = plt.subplots(1,1,figsize = (30,21),sharex = False,sharey = False)
    ax.plot(spectrum)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\phi_0(t)$')

    print('saving initial waveform...')
    plt.savefig('initial_waveform_plot.pdf', dpi=400)

if __name__ == '__main__':
    mass = 1e-19
    burst_duration_bosenova = 400 / (mass * SEC_TO_INEV)
    
    config_bosenova = SpectrumConfig(
        spectrum_type='bosenova', # 'gaussian'
        mass=mass,
        coupling=1e20,
        burst_duration=burst_duration_bosenova,
        K=1e-3,
        amplitude=1e20,
        density_profile_path='Galactic_Density_Profile.csv',
        density_num_points=1000,
        bosenova_path='Spectra/BosonStarSpectrumRelOnly.txt'
    )
    
    burst_duration_gaussian = 2*np.pi*10/(mass * SEC_TO_INEV)
    width = 2 * mass
    peak_momentum = 10 * mass
    config_gaussian = SpectrumConfig(
        spectrum_type='gaussian',
        mass=1e-19, 
        coupling=0,
        burst_duration=burst_duration_gaussian,
        K=1e-3,
        amplitude=1e20,
        density_profile_path='Galactic_Density_Profile.csv',
        density_num_points=1000,
        peak_momentum=peak_momentum,
        width=width
    )
    
    try:
        t_duration, phi = run_propagation(config_gaussian)
    except Exception as e:
        print(f"Error during propagation: {e}")
    