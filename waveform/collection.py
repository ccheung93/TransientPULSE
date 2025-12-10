"""
WaveformCollection - Main API for propagating scalar field waveforms.

This orchestrator class manages the workflow of propagating single or multiple
waveforms through a galactic density profile.
"""

import numpy as np
from utils.data_utils import read_medium_data, interpolate_data
from constants import KPC_TO_INEV, GCM3_TO_EV4
from propagation import propagation_time_GW

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

    def _load_density_profile(self, xi=None, xf=None):
        """
        Load and interpolate density profile then select a range
        
        Args:
            xi (float): initial density profile position
            xf (float): final density profile position
        
        Returns:
            ([np.ndarray, np.ndarray]): Density profile (x=position, y=density at position)
        """
        x, rho = read_medium_data(self.propagation.density_profile_path, i_R=0, i_rho=2)
        x_interp, rho_interp = interpolate_data(x, rho, self.propagation.density_num_points)
        
        if (xi is None and xf is not None) or (xf is None and xi is not None):
            return ValueError('Include both xi and xf if defining range for density profile.')
        
        if xi is not None and xi < x[0]:
            return ValueError('xi precedes initial position value in inputted density profile.')
        
        if xf is not None and xf > x[-1]:
            return ValueError('xf exceeds final position value in inputted density profile.')

        if xi is not None and xf is not None:
            mask = np.where((x_interp >= xi) & (x_interp <= xf))
            x_interp = x_interp[mask]
            rho_interp = rho_interp[mask]
        
        x_interp = x_interp* KPC_TO_INEV
        rho_interp = rho_interp * GCM3_TO_EV4
        
        self.density_profile = [x_interp, rho_interp]            

    def _compute_global_time_range(self, num_steps):
        """
        Compute global time range across all emission windows for aligned spectrograms

        Args:
            num_steps (int): Number of time steps

        Returns:
            tuple: (t_min, t_max) in natural units [1/eV]
        """
        from propagation import propagation_time
        from constants import SEC_TO_INEV

        # Get first spectrum to determine energy range
        spectrum = self.spectrum_source.get_spectrum(0)
        p, A = spectrum
        m = self.physics.mass
        E = np.sqrt(m**2 + p**2)

        # Calculate propagation times for all energies
        x, rho = self.density_profile
        t_arrivals = np.array([propagation_time(m, E_i, x, rho, self.physics.K, self.physics.coupling) for E_i in E])

        # Get min and max arrival times
        R = x[-1] - x[0]
        t_min_prop = propagation_time_GW(R)
        t_max_prop = np.max(t_arrivals)

        # Add emission time windows
        t_min_global = t_min_prop
        t_max_global = t_max_prop

        for i in range(num_steps):
            time_window = self.spectrum_source.get_time_window(i)
            if time_window is not None:
                # Arrival time = propagation time + emission time
                t_min_global = min(t_min_global, t_min_prop + time_window[0])
                t_max_global = max(t_max_global, t_max_prop + time_window[1])

        return (t_min_global, t_max_global)

    def propagate_all(self, N_points_spectrogram, xi=None, xf=None, save_waveform=True):
        """
        Propagate all time steps

        Args:
            N_points_spectrogram (int): Number of points for spectrogram output
            xi (float): initial density profile position
            xf (float): final density profile position
            save_waveform (bool): If True, save waveform plots for each time step

        Returns:
            dict: Aggregated results with keys 't_duration', 'spectrogram', 'E'
        """
        if self.density_profile is None:
            print(f'Loading density profile... selecting range between x_i={xi} and x_f={xf}')
            self._load_density_profile(xi=xi, xf=xf)

        num_steps = self.spectrum_source.get_num_time_steps()
        print(f'Propagating {num_steps} time steps')

        # For time-varying spectra, pre-compute global time range for aligned spectrograms
        global_time_range = None
        if num_steps > 1:
            global_time_range = self._compute_global_time_range(num_steps)

        # Clear previous results
        self.results = []

        for i in range(num_steps):
            spectrum = self.spectrum_source.get_spectrum(i)
            time_window = self.spectrum_source.get_time_window(i)
            result = self._propagate_single(spectrum, time_window, N_points_spectrogram, num_steps,
                                           save_waveform, global_time_range)
            self.results.append(result)

        return self._aggregate_results(N_points_spectrogram)


    def _propagate_single(self, spectrum, time_window, N_points_spectrogram, N_time_steps,
                          save_waveform=True, global_time_range=None):
        """
        Propagate a single spectrum snapshot

        Args:
            spectrum (tuple): (momenta, amplitudes)
            time_window (tuple or None): (t_start, t_end) in seconds * SEC_TO_INEV
            N_points_spectrogram (int): Number of points for spectrogram
            N_time_steps (int): Total number of time steps
            save_waveform (bool): If True, save waveform plot
            global_time_range (tuple, optional): (t_min, t_max) for aligned spectrograms

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
            save_waveform,
            global_time_range
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
