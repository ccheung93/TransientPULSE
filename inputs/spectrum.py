import numpy as np
from calculations.constraints import *
from utils.constants import *
from inputs.source import Source
from inputs.experiment import Experiment

class SignalModel:
    """Class representing an emission signal model for constraint calculations."""

    def __init__(self, file: str = None, source: Source = None, experiment: Experiment = None, aw: float = 1):
        """Initialize a signal model from either a file or a source model

        Args:
            file (str, optional): Path to CSV file containing spectrum data. Defaults to None.
            source (Source, optional): Source object for generating model spectrum. Defaults to None.
            experiment (Experiment, optional): Experiment object (required when source is provided). Defaults to None.
            aw (float, optional): Wavepacket uncertainty parameter. Defaults to 1.

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If neither file nor source is provided, or if source is provided without experiment
        """
        # Validate aw parameter
        self.aw = self._validate_aw(aw)

        # Initialize attributes
        self.w = None
        self.dw = 0
        self.rho = None
        self.trf = None
        self.E_unc = None
        self.constraint = None
        self.coupling_probe = None
        self.coupling_time_delays = {}

        # Validate input combination
        if file is not None and source is not None:
            raise ValueError(
                "Cannot specify both 'file' and 'source'. Choose one:\n"
                "  - file='path/to/spectrum.csv' for loading from file\n"
                "  - source=Source(...) for generating model spectrum"
            )

        if file is None and source is None:
            raise ValueError(
                "Must specify either 'file' or 'source'.\n"
                "Examples:\n"
                "  - SignalModel(file='path/to/spectrum.csv')\n"
                "  - SignalModel(source=Source(...), experiment=Experiment(...))"
            )

        # Validate source type if provided
        if source is not None:
            if not isinstance(source, Source):
                raise TypeError(
                    f"source must be a Source object, got {type(source).__name__}.\n"
                    f"Example: source=Source(Etot=1e-2, mass=1e-20, tstar=1, R=1e4, "
                    f"ULB_type='scalar', coupling_type='electron')"
                )

            # Check for required experiment
            if experiment is None:
                raise ValueError(
                    "experiment must be provided when using a source.\n"
                    "Example: experiment=Experiment(integration_time=86400, "
                    "integration_time_DM=31536000, sensitivity=1e-18, time_delays={'day': 86400})"
                )

            # Validate experiment type
            if not isinstance(experiment, Experiment):
                raise TypeError(
                    f"experiment must be an Experiment object, got {type(experiment).__name__}.\n"
                    f"Example: experiment=Experiment(integration_time=86400, "
                    "integration_time_DM=31536000, sensitivity=1e-18, time_delays={'day': 86400})"
                )

            self.is_axion = True if source.ULB_type == 'ALP' else False

        # Validate file type if provided
        if file is not None:
            if not isinstance(file, str):
                raise TypeError(
                    f"file must be a string path, got {type(file).__name__}.\n"
                    f"Example: file='data/BosonStarSpectrumRelOnly.txt'"
                )
            self.is_axion = False  # Default for file-based loading

        # Load or generate spectrum
        if file:
            self.load_from_csv(file)
        elif source:
            self.generate_model(source)

        # Compute derived quantities if using source model
        if source:
            self._compute_derived_quantities(source, experiment)
            self.coupling_probe = coupling_probe(
                source.Etot, source.tstar, source.R, self.w, source.mass,
                experiment.sensitivity, aw=1, t_int=experiment.integration_time,
                t_int_DM=experiment.integration_time_DM, coupling_type=source.coupling_type,
                coupling_order=source.coupling_order, axion=self.is_axion
            )

    @staticmethod
    def _validate_aw(aw):
        """Validate wavepacket uncertainty parameter

        Args:
            aw: The value to validate

        Returns:
            float: The validated value

        Raises:
            TypeError: If aw is not numeric
            ValueError: If aw is not >= 1
        """
        try:
            aw_float = float(aw)
        except (TypeError, ValueError):
            raise TypeError(
                f"aw must be a number, got {type(aw).__name__}.\n"
                f"aw is the wavepacket uncertainty parameter (aw >= 1).\n"
                f"Example: aw=1 (minimal uncertainty)"
            )

        if aw_float < 1:
            raise ValueError(
                f"aw must be >= 1 (uncertainty principle), got {aw_float}.\n"
                f"aw represents the wavepacket uncertainty: dw * t_star = aw >= 1.\n"
                f"Example: aw=1 (minimal uncertainty, Gaussian wavepacket)"
            )

        return aw_float
    
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

        self.constraint = coupling_conversion(
            coupling_type=source.coupling_type,
            coupling_order=source.coupling_order,
            axion=self.is_axion
        )
        for label, dt in experiment.time_delays.items():
            self.coupling_time_delays[label] = coupling_from_time_delay(
                dt, source.R, source.mass, self.w, 1e-6, 0, axion=self.is_axion
            )