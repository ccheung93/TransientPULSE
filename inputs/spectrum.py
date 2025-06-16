import numpy as np
from propagation import *
from inputs.source import Source
from inputs.experiment import Experiment

class Spectrum:
    """Class representing an emission spectrum."""
    
    def __init__(self, file: str = None, source: Source = None, experiment: Experiment = None, aw: float = 1):
        self.w = None
        self.aw = aw
        self.dw = 0
        self.rho = None
        self.trf = None
        self.E_unc = None
        self.constraint = None
        self.coupling_probe = None
        self.coupling_time_delays = {}
        self.is_axion = True if source.ULB_type == 'ALP' else False
        
        if file:
            self.load_from_csv(file)
        elif source:
            self.generate_model(source)
        else:
            raise ValueError('Expecting either a file directory or a source model.')
        
        if source:
            if experiment is None:
                raise ValueError('Experiment must be provided when using a source.')
            self._compute_derived_quantities(source, experiment)
            self.coupling_probe = coupling_probe(source.Etot, source.tstar, source.R, self.w, source.mass, experiment.sensitivity, aw=1, t_int=experiment.integration_time, t_int_DM=experiment.integration_time_DM, coupling_type=source.coupling_type, coupling_order=source.coupling_order, axion=self.is_axion)
            
    def load_from_csv(self, file):
        # Load spectrum data from file
        raise NotImplementedError('To be implemented...')
    
    def generate_model(self, source):
        """Create a synthetic spectrum from a source model.

        Args:
            source (Source): Instance of the Source class containing source parameters. 
                Imported from `source.py`.
        """
        # Create synthetic spectrum from source
        wmp_contour = np.logspace(0, 30, 1000)
        self.w = source.mass*wmp_contour
        
    def _compute_derived_quantities(self, source, experiment):
        """Compute derived physical quantities from the model spectrum and source.

        Args:
            source (Source): Instance of the Source class containing source parameters. 
                Imported from `source.py`.
            experiment (Experiment): Instance of the Experiment class containing experimental details. 
                Imported from `experiment.py`.
        """
        t_star = source.tstar * SEC_TO_INEV
        R = source.R * PC_TO_INEV
        
        q = calc_k_over_m(source.mass, self.w)
        self.dw = calc_energy_spread(t_star, aw=self.aw)
        
        dx_spread = calc_wave_spread(q, self.w, self.dw, R)
        total_wave_spread = calc_total_wave_spread(t_star, dx_spread)
        
        self.rho = calc_rho(source.Etot, R, total_wave_spread)
        
        tau_star = calc_tau_star(source.mass, q, self.dw, R, t_star)
        tau_DM = calc_tau_DM(self.w)
        
        self.trf = calc_timing_rescaling_factor(
            total_wave_spread, tau_star, tau_DM, 
            t_int=experiment.integration_time, 
            t_int_DM=experiment.integration_time_DM
        )
        
        self.E_unc = E_from_uncert(t_star)
        
        self.constraint = coupling_conversion(coupling_type=source.coupling_type, axion=self.is_axion)
        for label, dt in experiment.time_delays.items():
            self.coupling_time_delays[label] = coupling_from_time_delay(dt, source.R, source.mass, self.w, 1e-6, 0, axion=self.is_axion)