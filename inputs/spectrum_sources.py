"""
Spectrum source classes for the new architecture.

Provides flexible spectrum generation and loading for propagation simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import interpolate_data


class SpectrumSource:
    """Abstract base class for all spectrum sources"""

    def get_spectrum(self, time_index=None):
        """
        Returns (momenta, amplitudes) for a given time slice

        Args:
            time_index (int, optional): Index for time-dependent spectra

        Returns:
            tuple: (momenta array, amplitudes array)
        """
        raise NotImplementedError("Subclasses must implement get_spectrum()")

    def get_num_time_steps(self):
        """Returns number of time steps (1 for static spectra)"""
        return 1

    def get_time_window(self, time_index):
        """
        Returns (t_start, t_end) for this time slice

        Args:
            time_index (int): Time step index

        Returns:
            tuple: (t_start, t_end) or None for static spectra
        """
        return None


class CSVSpectrum(SpectrumSource):
    """Load spectrum from CSV file"""

    def __init__(self, filepath, i_p=0, i_A=1, skip_header=True, separator=None, num_points=None,
                 scaled_momentum=1.0, scaled_amplitude=1.0):
        """
        Args:
            filepath (str): Path to CSV file
            i_p (int): Column index for momentum
            i_A (int): Column index for amplitude
            skip_header (bool): Whether to skip the first line (default True)
            separator (str): Separator character (None = auto-detect comma vs whitespace)
            num_points (int, optional): If specified, interpolate to this many points
            scaled_momentum (float or callable): Scale factor for momentum, or function to apply
            scaled_amplitude (float or callable): Scale factor for amplitude, or function to apply
        """
        self.filepath = filepath
        self.i_p = i_p
        self.i_A = i_A
        self.skip_header = skip_header
        self.separator = separator
        self.num_points = num_points
        self.scaled_momentum = scaled_momentum
        self.scaled_amplitude = scaled_amplitude
        self.momenta, self.amplitudes = self._load_data()

    def _load_data(self):
        """Load spectrum from CSV file with flexible separator handling"""
        x = []
        y = []

        with open(self.filepath, 'r') as file:
            if self.skip_header:
                next(file)  # Skip header row

            for line in file:
                # Auto-detect separator if not specified
                if self.separator is None:
                    # Try comma first, then whitespace
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    if len(parts) <= 1:  # No commas found or only one value
                        parts = line.strip().split()
                else:
                    parts = [p.strip() for p in line.split(self.separator) if p.strip()]

                # Ensure we have enough columns
                if len(parts) > max(self.i_p, self.i_A):
                    try:
                        x.append(float(parts[self.i_p]))
                        y.append(float(parts[self.i_A]))
                    except (ValueError, IndexError):
                        continue  # Skip invalid lines

        x_array = np.array(x)
        y_array = np.array(y)

        # Apply interpolation if num_points is specified
        if self.num_points is not None:
            x_array, y_array = interpolate_data(x_array, y_array, self.num_points)

        # Apply scaling
        if callable(self.scaled_momentum):
            x_array = self.scaled_momentum(x_array)
        else:
            x_array = x_array * self.scaled_momentum

        if callable(self.scaled_amplitude):
            y_array = self.scaled_amplitude(y_array)
        else:
            y_array = y_array * self.scaled_amplitude

        return x_array, y_array

    def get_spectrum(self, time_index=None):
        """Returns the static spectrum

        Args:
            time_index (int, optional): Ignored for static spectra

        Returns:
            tuple: (momenta array, amplitudes array)
        """
        return self.momenta, self.amplitudes


class AnalyticSpectrum(SpectrumSource):
    """Generate spectrum from analytic formula

    Currently supports: 'gaussian'

    For non-analytic spectra (e.g., bosenova from file), use CSVSpectrum instead,
    or create a custom SpectrumSource subclass.
    """

    def __init__(self, formula_type, mass, num_points=1000, **params):
        """
        Args:
            formula_type (str): Type of spectrum (currently only 'gaussian')
            mass (float): Scalar field mass [eV]
            num_points (int): Number of momentum points
            **params: Additional parameters specific to formula type
                Gaussian: peak_momentum, width, amplitude
        """
        self.formula_type = formula_type
        self.mass = mass
        self.num_points = num_points
        self.params = params

    def get_spectrum(self, time_index=None):
        """Generate spectrum based on formula type

        Args:
            time_index (int, optional): Ignored for static spectra

        Returns:
            tuple: (momenta array, amplitudes array)
        """
        if self.formula_type == 'gaussian':
            return self._generate_gaussian()
        else:
            raise ValueError(
                f"Unknown formula type: '{self.formula_type}'. "
                f"AnalyticSpectrum currently supports: 'gaussian'. "
                f"For file-based spectra, use CSVSpectrum. "
                f"To implement your own analytic formula, create a custom class that inherits from SpectrumSource."
            )

    def _generate_gaussian(self):
        """Generate Gaussian spectrum"""
        peak_momentum = self.params.get('peak_momentum')
        width = self.params.get('width')
        amplitude = self.params.get('amplitude')

        if None in [peak_momentum, width, amplitude]:
            raise ValueError("Gaussian spectrum requires: peak_momentum, width, amplitude")

        # Momentum range
        p_initial = peak_momentum - width * np.sqrt(2 * np.log(10))
        p_final = peak_momentum + width * np.sqrt(2 * np.log(10))
        momenta = np.linspace(p_initial, p_final, self.num_points)

        amplitudes = amplitude * np.exp(-((momenta - peak_momentum)**2) / (2 * width**2)) / np.sqrt(self.num_points)

        return momenta, amplitudes


class TimeVaryingSpectrum(SpectrumSource):
    """Wrapper for time-dependent amplitude modulation"""

    def __init__(self, base_spectrum, amplitude_profile, time_windows=None):
        """
        Args:
            base_spectrum (SpectrumSource): Base spectrum shape
            amplitude_profile (list or array): Amplitude scale factors for each time step
            time_windows (list of tuples, optional): [(t_start, t_end), ...] for each time step
        """
        self.base_spectrum = base_spectrum
        self.amplitude_profile = np.array(amplitude_profile)
        self.time_windows = time_windows
        self.num_time_steps = len(self.amplitude_profile)

    def get_spectrum(self, time_index):
        """Returns spectrum scaled by amplitude profile"""
        if time_index is None or time_index >= self.num_time_steps:
            raise ValueError(f"time_index must be between 0 and {self.num_time_steps-1}")

        momenta, amplitudes = self.base_spectrum.get_spectrum()
        scale_factor = self.amplitude_profile[time_index]

        return momenta, amplitudes * scale_factor

    def get_num_time_steps(self):
        """Returns number of time steps"""
        return self.num_time_steps

    def get_time_window(self, time_index):
        """Returns time window for this time step"""
        if self.time_windows is not None and time_index < len(self.time_windows):
            return self.time_windows[time_index]
        return None


def plot_spectrum(spectrum_source, time_index=None, filename='spectrum_initial.pdf',
                  xlabel='Momentum (eV)', ylabel='Amplitude', title=None):
    """
    Plot and save a spectrum from a SpectrumSource.

    Args:
        spectrum_source (SpectrumSource): Source to plot
        time_index (int, optional): Time index for time-varying spectra
        filename (str): Output filename
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        title (str, optional): Plot title
    """
    momenta, amplitudes = spectrum_source.get_spectrum(time_index)

    _, ax = plt.subplots(figsize=(30, 21))
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 35, 'font.family': 'STIXGeneral'})
    plt.tight_layout()

    ax.plot(momenta, amplitudes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Saved spectrum plot to {filename}')
