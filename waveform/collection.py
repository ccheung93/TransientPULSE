"""
WaveformCollection - Main API for propagating scalar field waveforms.

This orchestrator class manages the workflow of propagating single or multiple
waveforms through a galactic density profile.
"""

import numpy as np
from utils.data_utils import read_medium_data, interpolate_data


class WaveformCollection:
    """Container for managing single or multiple waveforms"""

    def __init__(self, spectrum_source, physics_config, propagation_config):
        """
        Args:
            spectrum_source (SpectrumSource): Source of spectrum data
            physics_config (PhysicsConfig): Physics parameters
            propagation_config (PropagationConfig): Propagation environment parameters
        """
        self.spectrum_source = spectrum_source
        self.physics = physics_config
        self.propagation = propagation_config
        self.results = []
        self.density_profile = None

    def _load_density_profile(self):
        """Load and interpolate density profile"""
        x, rho = read_medium_data(self.propagation.density_profile_path, i_R=0, i_rho=2)
        x_interp, rho_interp = interpolate_data(x, rho, self.propagation.density_num_points)
        self.density_profile = [x_interp, rho_interp]

    def propagate_all(self, N_points_spectrogram, save_waveform=True):
        """
        Propagate all time steps

        Args:
            N_points_spectrogram (int): Number of points for spectrogram output
            save_waveform (bool): If True, save waveform plots for each time step

        Returns:
            dict: Aggregated results with keys 't_duration', 'spectrogram', 'E'
        """
        if self.density_profile is None:
            self._load_density_profile()

        # Check if this is a composite spectrum that should be propagated independently
        if hasattr(self.spectrum_source, 'get_independent_sources'):
            return self._propagate_composite(N_points_spectrogram, save_waveform)

        num_steps = self.spectrum_source.get_num_time_steps()

        # Clear previous results
        self.results = []

        for i in range(num_steps):
            spectrum = self.spectrum_source.get_spectrum(i)
            time_window = self.spectrum_source.get_time_window(i)

            result = self._propagate_single(spectrum, time_window, N_points_spectrogram, num_steps, save_waveform)
            self.results.append(result)

        return self._aggregate_results(N_points_spectrogram)

    def _propagate_composite(self, N_points_spectrogram, save_waveform=True):
        """
        Propagate composite spectrum by treating each source independently

        Args:
            N_points_spectrogram (int): Number of points for spectrogram output
            save_waveform (bool): If True, save overlay plot of all waveforms

        Returns:
            dict: Aggregated results from all independent sources
        """
        independent_sources = self.spectrum_source.get_independent_sources()
        all_source_results = []

        for source in independent_sources:
            # Create temporary collection for this source
            temp_collection = WaveformCollection(source, self.physics, self.propagation)
            temp_collection.density_profile = self.density_profile

            # Propagate this source (don't save individual waveforms yet)
            source_results = temp_collection.propagate_all(N_points_spectrogram, save_waveform=False)
            all_source_results.append(source_results)

        # Optionally plot all waveforms overlaid on the same axes
        if save_waveform:
            self._plot_composite_waveforms(all_source_results)

        # Combine spectrograms from all sources
        combined_spectrogram = np.sum([r['spectrogram'] for r in all_source_results], axis=0)

        # Find global time range
        t_min = min(r['t_min'] for r in all_source_results)
        t_max = max(r['t_max'] for r in all_source_results)

        # Use first source for E and valid arrays (they should all be similar)
        return {
            't_min': t_min,
            't_max': t_max,
            'N_points': N_points_spectrogram,
            'E': all_source_results[0]['E'],
            'spectrogram': combined_spectrogram,
            'valid': all_source_results[0]['valid']
        }

    def _plot_composite_waveforms(self, all_source_results):
        """
        Plot all individual waveforms from composite sources on the same axes

        Args:
            all_source_results (list): List of result dicts from each source
        """
        import matplotlib.pyplot as plt
        from utils.logging_utils import get_logger
        import time

        logger = get_logger()
        start_time = time.time()
        logger.info('Saving composite waveform plot to waveform_plot.pdf')

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams.update({'font.size': 35, 'font.family': 'STIXGeneral'})
        plt.rcParams['hatch.color'] = 'lightgray'

        fig, ax = plt.subplots(figsize=(30, 21))

        # Plot each waveform
        for i, result in enumerate(all_source_results):
            ax.plot(result['t_duration'], result['phi_signal'],
                   label=f'Source {i+1}', linewidth=2)

        ax.set_xlabel('Time (s)', fontsize=35)
        ax.set_ylabel(r'$\phi(t)$', fontsize=35)
        ax.tick_params(labelsize=30)
        ax.legend(fontsize=30)

        plt.tight_layout()
        plt.savefig('waveform_plot.pdf', dpi=300)
        plt.close()

        end_time = time.time()
        logger.info(f'Saved waveform_plot.pdf in {end_time - start_time:.2f}s')

    def _propagate_single(self, spectrum, time_window, N_points_spectrogram, N_time_steps, save_waveform=True):
        """
        Propagate a single spectrum snapshot

        Args:
            spectrum (tuple): (momenta, amplitudes)
            time_window (tuple or None): (t_start, t_end) in seconds * SEC_TO_INEV
            N_points_spectrogram (int): Number of points for spectrogram
            N_time_steps (int): Total number of time steps
            save_waveform (bool): If True, save waveform plot

        Returns:
            dict: Results with keys 't_duration', 'phi_signal', 'N_points', 'E', 'spectrogram'
        """
        # Import here to avoid circular dependency
        from waveform.propagation import propagation

        # Convert time_window if provided
        delta_t_s = None
        if time_window is not None:
            delta_t_s = list(time_window)

        t_duration, phi_signal, N_points, E, spectrogram, valid = propagation(
            spectrum,
            self.density_profile,
            self.physics.mass,
            self.physics.coupling,
            self.physics.K,
            self.physics.burst_duration,
            N_points_spectrogram,
            delta_t_s,
            N_time_steps,
            save_waveform
        )

        return {
            't_duration': t_duration,
            'phi_signal': phi_signal,
            'N_points': N_points,
            'E': E,
            'spectrogram': spectrogram,
            'valid': valid
        }

    def _aggregate_results(self, N_points_spectrogram):
        """
        Combine results from all time steps

        Args:
            N_points_spectrogram (int): Number of points for spectrogram

        Returns:
            dict: Aggregated results
        """
        if not self.results:
            raise ValueError("No results to aggregate. Run propagate_all() first.")

        if len(self.results) == 1:
            # Single time step - normalize return format for consistency
            result = self.results[0]
            return {
                't_min': result['t_duration'].min(),
                't_max': result['t_duration'].max(),
                'N_points': N_points_spectrogram,
                'E': result['E'],
                'spectrogram': result['spectrogram'],
                't_duration': result['t_duration'],
                'phi_signal': result['phi_signal'],
                'valid': result['valid']
            }

        # Multiple time steps - aggregate spectrograms
        spectrogram_total = np.sum([r['spectrogram'] for r in self.results], axis=0)

        t_min = min(r['t_duration'].min() for r in self.results)
        t_max = max(r['t_duration'].max() for r in self.results)

        # Use E and valid from first result (should be the same for all)
        E = self.results[0]['E']
        valid = self.results[0]['valid']

        return {
            't_min': t_min,
            't_max': t_max,
            'N_points': N_points_spectrogram,
            'E': E,
            'spectrogram': spectrogram_total,
            'valid': valid
        }
