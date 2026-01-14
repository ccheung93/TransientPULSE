"""
Examples demonstrating the new architecture for waveform propagation.

This file contains example usage of the WaveformCollection API with various
spectrum sources: single Gaussian, time-varying, and CSV (bosenova).
"""

import numpy as np
from utils.constants import SEC_TO_INEV, KPC_TO_INEV, GCM3_TO_EV4
from utils.data_utils import read_medium_data, interpolate_data
from inputs.spectrum_sources import (SpectrumSource, CSVSpectrum, AnalyticSpectrum, TimeVaryingSpectrum)
from inputs.configs import PhysicsConfig, PropagationConfig
from utils.logging_utils import setup_logging, get_logger
from waveform.collection import WaveformCollection
from waveform.propagation import plot_spectrogram, export_source_parameters

def example_new_architecture_single_gaussian(filename=None):
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
    avg_density = plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], filename=filename)

    logger.info("Single Gaussian propagation complete. Spectrogram saved.")
    
    # Export source parameters
    R = collection.density_profile[0][-1] - collection.density_profile[0][0]
    export_source_parameters(avg_density, burst_duration, R, mass, True, 'gaussian.param')
    
    return results


def example_new_architecture_time_varying(filename=None):
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
    time_windows = [(i * burst_duration * SEC_TO_INEV / N_time_steps, (i + 1) * burst_duration * SEC_TO_INEV / N_time_steps)
                    for i in range(N_time_steps)]
    
    # Print emission details
    base_amplitude = base_gaussian.params.get('amplitude', 1.0)
    logger.info(f"\nEmission Configuration:")
    logger.info(f"  Number of time steps: {N_time_steps}")
    logger.info(f"  Burst duration: {burst_duration:.3e} s")
    logger.info(f"  Base amplitude: {base_amplitude:.3e}")
    logger.info(f"\nTime-varying emission schedule:")
    for i in range(N_time_steps):
        amp_factor = amplitude_profile[i]
        effective_amp = base_amplitude * amp_factor
        t_start = time_windows[i][0] / SEC_TO_INEV  # Convert to seconds
        t_end = time_windows[i][1] / SEC_TO_INEV
        energy_density_factor = amp_factor**2
        logger.info(f"  Step {i}:")
        logger.info(f"    Amplitude factor: {amp_factor:.4f}")
        logger.info(f"    Effective amplitude: {effective_amp:.3e}")
        logger.info(f"    Energy density factor: {energy_density_factor:.4f} ({energy_density_factor:.2e})")
        logger.info(f"    Emission time window: [{t_start:.3e}, {t_end:.3e}] s")
        
    time_varying = TimeVaryingSpectrum(base_gaussian, amplitude_profile, time_windows)

    # Configure and propagate
    physics = PhysicsConfig(mass=mass, coupling=1e20, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=2000)

    collection = WaveformCollection(time_varying, physics, propagation_config)
    results = collection.propagate_all(N_points_spectrogram=2000, save_waveform=False)

    # Plot spectrogram
    avg_density = plot_spectrogram(results['N_points'], results['t_min'], results['t_max'], results['E'], results['spectrogram'], filename=filename)

    logger.info(f"Time-varying propagation complete with {N_time_steps} steps")
    
    # Export source parameters
    R = collection.density_profile[0][-1] - collection.density_profile[0][0]
    export_source_parameters(avg_density, burst_duration, R, mass, True, 'gaussian_tv.param')
    
    return results


def example_bosenova_csv(filename=None):
    """Example: Load spectrum from CSV file (Bosenova spectrum)"""
    logger = get_logger()
    logger.info("\n=== Example 3: CSV Spectrum (Bosenova) ===")

    mass = 1e-19
    burst_duration = 400 / (mass * SEC_TO_INEV)

    # Load bosenova spectrum from file (no header, space-separated)
    # Use num_points=3000 for interpolation to match old implementation
    spectrum = CSVSpectrum('Spectra/BosonStarSpectrumRelOnly.txt', i_p=0, i_A=1, skip_header=False, num_points=1000)

    # The file contains dimensionless values - need to scale properly
    # Following old code normalization: momenta * mass, amplitudes * sqrt(1/1e-85)
    momenta, amplitudes = spectrum.get_spectrum()
    scaled_momenta = momenta * mass
    scaled_amplitudes = np.sqrt(amplitudes * (1/1e-85))  # TODO - define this

    # Create a wrapper class to return scaled values
    class ScaledSpectrum(SpectrumSource):
        def get_spectrum(self, time_index=None):
            return scaled_momenta, scaled_amplitudes

    scaled_spectrum = ScaledSpectrum()

    physics = PhysicsConfig(mass=mass, coupling=1e22, K=1e-3, burst_duration=burst_duration)
    propagation_config = PropagationConfig('Galactic_Density_Profile.csv', density_num_points=1000)

    # For bosenova, need to scale the density profile distance from 10kpc to 1pc (x/10000) before creating collection
    # We'll create a custom WaveformCollection with modified density loading
    collection = WaveformCollection(scaled_spectrum, physics, propagation_config)
    # Override the density profile to apply bosenova scaling
    x, rho = read_medium_data('Galactic_Density_Profile.csv', i_R=0, i_rho=2)
    x_interp, rho_interp = interpolate_data(x/10000, rho, 1000)  # Bosenova: Convert 10 kpc to 1 pc
    collection.density_profile = [x_interp * KPC_TO_INEV, rho_interp * GCM3_TO_EV4]

    results = collection.propagate_all(N_points_spectrogram=1000, save_waveform=False)

    # Plot spectrogram
    avg_density = plot_spectrogram(results['N_points'], results['t_min'], results['t_max'],
                     results['E'], results['spectrogram'], filename=filename)

    logger.info("CSV (Bosenova) spectrum propagation complete. Spectrogram saved.")
    
    # Export source parameters
    R = collection.density_profile[0][-1] - collection.density_profile[0][0]
    export_source_parameters(avg_density, burst_duration, R, mass, True, 'bosenova.param')
    
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
    example_new_architecture_single_gaussian(filename='spectrogram_single_gaussian.pdf')
    example_new_architecture_time_varying(filename='spectrogram_time_varying.pdf')
    example_bosenova_csv(filename='spectrogram_bosenova.pdf')
