import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import scipy

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

def propagation(spec, density_profile, m, d, K, ts_sec, N_points_spectrogram=None, time_step_range=None, N_time_steps=None):
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
    N_time_steps = N_time_steps if N_time_steps else 1
    
    # Spectrum
    p, A = spec
    del spec
    E = np.sqrt(m**2 + p**2)
    
    # Density profile
    x, rho = density_profile
    R = x[-1] # distance from the Earth to Galactic center [kpc] TODO - parametrize this
    
    t_arrivals = np.array([t_prop(m, E_i, x, rho, K, d) for E_i in E])
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
    delta_t_s = [time_step / N_time_steps for time_step in time_step_range] if time_step_range else [0, 0]
    t_fastest = np.min(t_arrivals[valid]) + delta_t_s[0]
    t_slowest = np.max(t_arrivals[valid]) + delta_t_s[1]
    t_d = t_slowest - t_fastest
    print('time in years is:' + str((t_fastest/SEC_TO_INEV)/3e7))

    N_points = min(int(2*t_d * np.max(E[valid]/(2*np.pi))), 1e4)
    print(f"N_points={int(2*t_d * np.max(E[valid]/(2*np.pi)))}", N_points)
    
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

    # Plotting waveform
    fig, ax3 = plt.subplots(figsize = (30, 21))
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 35,'font.family':'STIXGeneral'})
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.rcParams['hatch.color'] = 'lightgray'
    plt.tight_layout()
    
    ax3.plot(t_duration_Earth_s, phi_t_final)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'$\phi(t)$')

    print('saving waveform_plot...')
    plt.savefig('waveform_plot.pdf', dpi=300)
    plt.close()
    end_time = time.time()
    print(f'TIMING >>> Time up to saving waveform_plot was {end_time - start_time:.2f}s.')
    
    return t_duration_Earth_s, phi_t_final, N_points, E, spectrogram_array

def plot_spectrogram(N_points, t_min, t_max, E, spectrogram_array):
    ### Spectrogram
    start_time = time.time()
    print("spectrogram started") 
    delta_t = (t_max - t_min) / N_points
    print(f'delta_t = {delta_t}, N={N_points}')
    
    end_time = time.time()
    print(f"TIMING >>> Spectrogram took {end_time - start_time:.2f}s.")
    print('TIME_RANGE', t_min, t_max)
    # Plot spectrogram
    start_time = time.time()
    freq = E*SEC_TO_INEV/(2*PI)
    
    fig, ax5 = plt.subplots(figsize = (30, 21))
    cmap_name = 'viridis'
    cmap = plt.colormaps[cmap_name]
    rgb = plt.colormaps[cmap_name](0)
    cmap.set_bad(rgb)
    masked_data = np.ma.masked_where(spectrogram_array <= 0, spectrogram_array)
    
    im = ax5.imshow(masked_data, 
                    aspect="auto", 
                    origin="lower", 
                    extent = [t_min, t_max, freq[0], freq[-1]], 
                    norm=mcolors.LogNorm(), 
                    cmap=cmap)
    fig.colorbar(im, label=r"$\rho_*~[{\rm eV}^4]$")
    ax5.set(xlabel="Time (s)",
            ylabel=r"Frequency $f$ [Hz]"
            )
    
    plt.savefig('spectrogram_plot.pdf', bbox_inches='tight')
    plt.close()

    end_time = time.time()
    print(f"TIMING >>> Saved spectrogram_plot.pdf in {end_time - start_time:.2f}s.")

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
    def __init__(self, filepath, mass, num_points=1000):
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
    def __init__(self, mass, burst_duration, peak_momentum, width, amplitude, num_points=1000):
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
        dw = self.width
        p_peak = self.peak_momentum
        
        # momentum range 
        p_initial = p_peak - dw*np.sqrt(2*np.log(10)) # 2 * PI / tstar_ineV
        p_final =  p_peak + dw*np.sqrt(2*np.log(10))
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
        return BosenovaSpectrum(config.bosenova_path, config.mass, num_points=3000)
    elif config.spectrum_type == 'gaussian':
        return GaussianSpectrum(config.mass, config.burst_duration, config.peak_momentum, config.width, config.amplitude)
    else:
        raise ValueError(f"Unknown spectrum type: {config.spectrum_type}")

def run_propagation(config, N_points_spectrogram, delta_t_s=None, N_time_steps=None):
    # Reading density profile from CSV file
    x, rho = read_medium_data(config.density_profile_path, i_R=0, i_rho=2)
    
    if config.spectrum_type == 'bosenova':
        x_interp, rho_interp = interpolate_data(x/10000, rho, config.density_num_points) # Converting 10 kpc to 1 pc
    else:
        x_interp, rho_interp = interpolate_data(x, rho, config.density_num_points)
    density_profile = [x_interp, rho_interp]
    
    # Generate spectrum
    spectrum_generator = generate_spectrum(config)
    momenta, amplitudes = spectrum_generator.generate()
    spectrum = [momenta, amplitudes]
    ####TO-DO: When introducing time dependence, use nested arrays for different amplitudes.
    # Save spectrum figure
    
    
    # Run propagation
    t_duration, phi_signal, N_points, E, spectrogram = propagation(spectrum, 
                                                                   density_profile, 
                                                                   config.mass, 
                                                                   config.coupling, 
                                                                   config.K, 
                                                                   config.burst_duration, 
                                                                   N_points_spectrogram,
                                                                   delta_t_s, 
                                                                   N_time_steps)
    
    return t_duration, N_points, E, spectrogram

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
        coupling=1.5e22,
        burst_duration=burst_duration_bosenova,
        K=1e-3,
        amplitude=1e20,
        density_profile_path='Galactic_Density_Profile.csv',
        density_num_points=1000,
        bosenova_path='Spectra/BosonStarSpectrumRelOnly.txt'
    )
    
    ##### TODO - Investigate the burst parameter relationships so that everything is nicely consistent.
    
    mul = 100
    burst_duration_gaussian = 2*np.pi*mul/(mass * SEC_TO_INEV) #1e1
    width = 2e1*mass#10/ (burst_duration_gaussian * SEC_TO_INEV)
    peak_momentum = 2e2 * mass #30/ (burst_duration_gaussian * SEC_TO_INEV)
    config_gaussian = SpectrumConfig(
        spectrum_type='gaussian',
        mass=mass,
        coupling=1e20,
        burst_duration=burst_duration_gaussian,
        K=1e-3,
        amplitude=1e50,
        density_profile_path='Galactic_Density_Profile.csv',
        density_num_points=2000,
        peak_momentum=peak_momentum,
        width=width
    )
    
    time_dependent = True
    
    if time_dependent:
        N_time_steps = 20
        N_points_spectrogram = 2e3
        gaussian_configs = []
        for i in range(N_time_steps):
            config_gaussian = SpectrumConfig(
                spectrum_type='gaussian',
                mass=mass,
                coupling=1e20,
                burst_duration=burst_duration_gaussian,
                K=1e-3,
                amplitude=1e50*np.exp(-(i-(N_time_steps/2))**2/(N_time_steps/3)**2),
                density_profile_path='Galactic_Density_Profile.csv',
                density_num_points=2000,
                peak_momentum=peak_momentum,
                width=width
            )
            gaussian_configs.append(config_gaussian)

        t_durations = []
        N_points_array = []
        E_array = []
        spectrogram_array = []
        for i, config in enumerate(gaussian_configs):
            delta_t_s = [i * burst_duration_gaussian * SEC_TO_INEV, 
                         (i + 1) * burst_duration_gaussian * SEC_TO_INEV] 
            t_duration, N_points, E, spectrogram = run_propagation(config, N_points_spectrogram, delta_t_s, N_time_steps)
            t_durations.append(t_duration)
            N_points_array.append(N_points)
            E_array.append(E)
            spectrogram_array.append(spectrogram)

        spectrogram_total = np.sum(spectrogram_array, axis=0)
        t_min = min(t_dur.min() for t_dur in t_durations)
        t_max = max(t_dur.max() for t_dur in t_durations)
        plot_spectrogram(N_points_spectrogram, t_min, t_max, E, spectrogram_total)
    else:
        try:
            t_duration, N_points, E, spectrogram = run_propagation(config_gaussian)
            plot_spectrogram(N_points, t_duration, E, spectrogram)
        except Exception as e:
            print(f"Error during propagation: {e}")
    