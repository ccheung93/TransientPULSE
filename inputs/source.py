from propagation import SOLAR_TO_EV

class Source:
    """Class representing a source of ultralight bosonic (ULB) fields."""
    
    VALID_ULB_TYPES = {'scalar', 'ALP'}
    VALID_COUPLINGS = {
        'scalar': {'electron', 'photon', 'gluon'},
        'ALP': {'proton', 'electron', 'neutron', 'photon', 'EDM'}
    }
    
    def __init__(self, Etot, mass, tstar, R, ULB_type, coupling_type, coupling_order=None):
        """Initialize the source parameters

        Args:
            Etot (float): Total energy [solar mass]
            mass (float): Mass of phi field [eV]
            tstar (float): Intrinsic burst duration [s]
            R (float): Distance between source and detector [pc]
            ULB_type (str): Type of ULB ('scalar' or 'ALP')
            coupling_type (str): Type of coupling (e.g. 'electron', 'photon')
            coupling_order (str, optional): Coupling order (for scalar only). Defaults to None.
        """
        self.Etot = Etot * SOLAR_TO_EV
        self.mass = mass
        self.tstar = tstar
        self.R = R
        
        self._validate_and_set_ULB_type(ULB_type)
        self._validate_and_set_coupling(coupling_type, coupling_order)
    
    def _validate_and_set_ULB_type(self, ULB_type):
        if ULB_type not in self.VALID_ULB_TYPES:
            raise ValueError(f'ULB_type must be one of {self.VALID_ULB_TYPES}.')
        self.ULB_type = ULB_type
    
    def _validate_and_set_coupling(self, coupling_type, coupling_order):
        valid_types = self.VALID_COUPLINGS[self.ULB_type]
        if coupling_type not in valid_types:
            raise ValueError(f'coupling_type "{coupling_type}" is not compatible with ULB_type "{self.ULB_type}. Must be one of {valid_types}.')
        self.coupling_type = coupling_type
        self.coupling_order = coupling_order if self.ULB_type == 'scalar' else self.ULB_type