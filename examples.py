"""
Examples demonstrating the new architecture for waveform propagation.

This file contains example usage of the WaveformCollection API with various
spectrum sources: single Gaussian, time-varying, composite, and CSV (bosenova).
"""

import numpy as np
from constants import SEC_TO_INEV
from utils.data_utils import read_medium_data, interpolate_data
from inputs.spectrum_sources import (SpectrumSource, CSVSpectrum, AnalyticSpectrum,
                                      TimeVaryingSpectrum, CompositeSpectrum, plot_spectrum)
from inputs.configs import PhysicsConfig, PropagationConfig
from utils.logging_utils import setup_logging, get_logger
from waveform.collection import WaveformCollection
from waveform.propagation import plot_spectrogram

def example_new_architecture_single_gaussian():
    """Example: Single static Gaussian spectrum using new architecture"""
    logger = get_logger()
    logger.info("\n=== Example 1: Single Static Gaussian ===")

    mass = 1e-19
    burst_duration = 2 * np.pi * 100 / (mass * SEC_TO_INEV)

    # Create spectrum source
    spectrum = AnalyticSpectrum(
        'gaussian',
        mass=mass,
        num_points=1000,
        peak_momentum=2e2 * mass,
        width=2e1 * mass,
        amplitude=1e50
    )

    # Optionally plot the initial spectrum before propagation
    # plot_spectrum(spectrum, filename='gaussian_initial.pdf')

    # Configure physics and propagation
    physics = PhysicsConfig(mass=mass, coupling=1e20, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=2000)

    # Create collection and propagate
    collection = WaveformCollection(spectrum, physics, propagation_config)
    results = collection.propagate_all(N_points_spectrogram=2000)

    # Plot spectrogram
    plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], results['valid'])

    logger.info("Single Gaussian propagation complete. Spectrogram saved.")
    return results


def example_new_architecture_time_varying():
    """Example: Time-varying Gaussian with amplitude modulation"""
    logger = get_logger()
    logger.info("\n=== Example 2: Time-Varying Gaussian ===")

    mass = 1e-19
    burst_duration = 2 * np.pi * 100 / (mass * SEC_TO_INEV)

    # Create base Gaussian spectrum
    base_gaussian = AnalyticSpectrum(
        'gaussian',
        mass=mass,
        num_points=1000,
        peak_momentum=2e2 * mass,
        width=2e1 * mass,
        amplitude=1e50
    )

    # Create time-varying wrapper with Gaussian amplitude profile
    N_time_steps = 20
    amplitude_profile = [np.exp(-(i - N_time_steps/2)**2 / (N_time_steps/3)**2)
                         for i in range(N_time_steps)]

    # Define time windows for each step
    time_windows = [(i * burst_duration * SEC_TO_INEV, (i + 1) * burst_duration * SEC_TO_INEV)
                    for i in range(N_time_steps)]

    time_varying = TimeVaryingSpectrum(base_gaussian, amplitude_profile, time_windows)

    # Configure and propagate
    physics = PhysicsConfig(mass=mass, coupling=1e20, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=2000)

    collection = WaveformCollection(time_varying, physics, propagation_config)
    results = collection.propagate_all(N_points_spectrogram=2000, save_waveform=False)

    # Plot spectrogram
    plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], results['valid'])

    logger.info(f"Time-varying propagation complete with {N_time_steps} steps")
    return results


def example_new_architecture_composite():
    """Example: Multiple Gaussian emission lines combined"""
    logger = get_logger()
    logger.info("\n=== Example 3: Composite Spectrum (Multiple Emission Lines) ===")

    mass = 1e-19
    burst_duration = 2 * np.pi * 100 / (mass * SEC_TO_INEV)

    # Create two Gaussian emission lines at different energies
    # Both will be propagated independently and combined in the spectrogram
    gaussian1 = AnalyticSpectrum(
        'gaussian',
        mass=mass,
        num_points=1000,
        peak_momentum=1e2 * mass,
        width=1e1 * mass,
        amplitude=4e50
    )

    gaussian2 = AnalyticSpectrum(
        'gaussian',
        mass=mass,
        num_points=1000,
        peak_momentum=1.5e2 * mass,
        width=1e1 * mass,
        amplitude=5e50
    )

    # Combine into composite spectrum
    composite = CompositeSpectrum([gaussian1, gaussian2])

    # Optionally plot the initial composite spectrum before propagation
    plot_spectrum(composite, filename='composite_initial.pdf', xlabel='Momentum (eV)', ylabel='Amplitude')

    # Configure and propagate
    physics = PhysicsConfig(mass=mass, coupling=1e20, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=2000)

    collection = WaveformCollection(composite, physics, propagation_config)
    results = collection.propagate_all(N_points_spectrogram=2000, save_waveform=True)

    # Plot spectrogram
    plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], results['valid'])

    logger.info("Composite spectrum propagation complete. Spectrogram saved.")
    return results


def example_new_architecture_csv():
    """Example: Load spectrum from CSV file (Bosenova spectrum)"""
    logger = get_logger()
    logger.info("\n=== Example 4: CSV Spectrum (Bosenova) ===")

    mass = 1e-19
    burst_duration = 400 / (mass * SEC_TO_INEV)

    # Load bosenova spectrum from file (no header, space-separated)
    # Use num_points=3000 for interpolation to match old implementation
    spectrum = CSVSpectrum('Spectra/BosonStarSpectrumRelOnly.txt',
                           i_p=0, i_A=1, skip_header=False, num_points=3000)

    # The file contains dimensionless values - need to scale properly
    # Following old code normalization: momenta * mass, amplitudes * sqrt(1/1e-85)
    momenta, amplitudes = spectrum.get_spectrum()
    scaled_momenta = momenta * mass
    scaled_amplitudes = np.sqrt(amplitudes * (1/1e-85))  # Match old BosenovaSpectrum normalization

    # Create a wrapper class to return scaled values
    class ScaledSpectrum(SpectrumSource):
        def get_spectrum(self, time_index=None):
            return scaled_momenta, scaled_amplitudes

    scaled_spectrum = ScaledSpectrum()

    physics = PhysicsConfig(mass=mass, coupling=1.5e22, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=1000)

    # For bosenova, need to scale the density profile distance from 10kpc to 1pc (x/10000) before creating collection
    # We'll create a custom WaveformCollection with modified density loading
    collection = WaveformCollection(scaled_spectrum, physics, propagation_config)
    # Override the density profile to apply bosenova scaling
    x, rho = read_medium_data('Galactic_Density_Profile.csv', i_R=0, i_rho=2)
    x_interp, rho_interp = interpolate_data(x/10000, rho, 1000)  # Bosenova: Convert 10 kpc to 1 pc
    collection.density_profile = [x_interp, rho_interp]

    results = collection.propagate_all(N_points_spectrogram=2000, save_waveform=False)

    # Plot spectrogram
    plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], results['valid'])

    logger.info("CSV (Bosenova) spectrum propagation complete. Spectrogram saved.")
    return results

if __name__ == '__main__':
    # Configure logging (optional - defaults to INFO level)
    # Options:
    #   level='DEBUG'  - Show all messages including detailed debug info
    #   level='INFO'   - Show informational messages and above (default)
    #   level='WARNING' - Show only warnings and errors
    #   console_level='INFO', file_level='DEBUG' - Different levels for console vs file
    setup_logging(log_file='propagation.log', level='INFO')

    # Uncomment the example you want to run:
    # example_new_architecture_single_gaussian()
    # example_new_architecture_time_varying()
    # example_new_architecture_composite()
    example_new_architecture_csv()
