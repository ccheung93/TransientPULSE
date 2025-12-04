from propagation import SOLAR_TO_EV
from utils.validation import validate_positive_float, validate_choice


class Source:
    """Class representing a source of ultralight bosonic (ULB) fields."""

    VALID_ULB_TYPES = {'scalar', 'ALP'}
    VALID_COUPLINGS = {
        'scalar': {'electron', 'photon', 'gluon'},
        'ALP': {'proton', 'electron', 'neutron', 'photon', 'EDM'}
    }
    VALID_COUPLING_ORDERS = {'linear', 'quad', None}

    def __init__(self, Etot, mass, tstar, R, ULB_type, coupling_type, coupling_order=None):
        """Initialize the source parameters

        Args:
            Etot (float): Total energy [solar mass]
            mass (float): Mass of phi field [eV]
            tstar (float): Intrinsic burst duration [s]
            R (float): Distance between source and detector [pc]
            ULB_type (str): Type of ULB ('scalar' or 'ALP')
            coupling_type (str): Type of coupling (e.g. 'electron', 'photon')
            coupling_order (str or int, optional): Coupling order (for scalar only).
                Accepts: 'linear'/1 or 'quad'/2. Defaults to None.

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If numerical values are non-positive or string values are invalid

        Examples:
            >>> # Using strings
            >>> source = Source(1e-2, 1e-20, 1, 1e4, 'scalar', 'electron', 'linear')
            >>> # Using integers (equivalent)
            >>> source = Source(1e-2, 1e-20, 1, 1e4, 'scalar', 'electron', 1)
        """
        # Validate and set numeric parameters
        Etot_validated = validate_positive_float(Etot, 'Etot', 'solar masses')
        self.Etot = Etot_validated * SOLAR_TO_EV

        self.mass = validate_positive_float(mass, 'mass', 'eV')
        self.tstar = validate_positive_float(tstar, 'tstar', 'seconds')
        self.R = validate_positive_float(R, 'R', 'parsecs')

        # Validate and set categorical parameters
        self._validate_and_set_ULB_type(ULB_type)
        self._validate_and_set_coupling(coupling_type, coupling_order)

    def _validate_and_set_ULB_type(self, ULB_type):
        """Validate and set ULB type

        Args:
            ULB_type: The ULB type to validate

        Raises:
            TypeError: If ULB_type is not a string
            ValueError: If ULB_type is not a valid option
        """
        self.ULB_type = validate_choice(
            ULB_type, 'ULB_type', self.VALID_ULB_TYPES
        )

    def _validate_and_set_coupling(self, coupling_type, coupling_order):
        """Validate and set coupling type and order

        Args:
            coupling_type: The coupling type to validate
            coupling_order: The coupling order to validate (for scalar only)

        Raises:
            TypeError: If coupling_type is not a string
            ValueError: If coupling_type is not compatible with ULB_type or coupling_order is invalid
        """
        # Validate coupling_type
        valid_types = self.VALID_COUPLINGS[self.ULB_type]
        self.coupling_type = validate_choice(
            coupling_type, 'coupling_type', valid_types
        )

        # Validate coupling_order
        if self.ULB_type == 'scalar':
            # Convert integer to string if needed (1 -> 'linear', 2 -> 'quad')
            if isinstance(coupling_order, int):
                if coupling_order == 1:
                    coupling_order = 'linear'
                elif coupling_order == 2:
                    coupling_order = 'quad'
                else:
                    raise ValueError(
                        f"coupling_order integer must be 1 or 2, got {coupling_order}.\n"
                        f"Valid options: 1 (linear), 2 (quad), 'linear', 'quad', or None\n"
                        f"Example: coupling_order=1 or coupling_order='linear'"
                    )

            # Validate string/None values
            if coupling_order is not None and not isinstance(coupling_order, str):
                raise TypeError(
                    f"coupling_order must be a string, integer (1 or 2), or None, "
                    f"got {type(coupling_order).__name__}.\n"
                    f"Valid options: 1 (linear), 2 (quad), 'linear', 'quad', or None\n"
                    f"Example: coupling_order=1 or coupling_order='linear'"
                )

            if coupling_order not in self.VALID_COUPLING_ORDERS:
                raise ValueError(
                    f"coupling_order '{coupling_order}' is not valid.\n"
                    f"Valid options: 1 (linear), 2 (quad), 'linear', 'quad', or None\n"
                    f"Example: coupling_order=1 or coupling_order='linear'"
                )

            self.coupling_order = coupling_order
        else:
            # For ALP, coupling_order should be set to ULB_type
            if coupling_order is not None:
                import warnings
                warnings.warn(
                    f"coupling_order is ignored for ULB_type='{self.ULB_type}'. "
                    f"It will be set to '{self.ULB_type}' automatically.",
                    UserWarning
                )
            self.coupling_order = self.ULB_type