class Experiment:
    """Class representing the details of an experiment."""
    
    def __init__(self, integration_time, integration_time_DM, sensitivity, time_delays):
        """Initialize the experimental details

        Args:
            integration_time (float): Integration time for transient search [s]
            integration_time_DM (float): Integration time for DM experiment to reach sensitivity [s]
            sensitivity (float): Fractional frequency sensitivity of experiment to transient signal
            time_delays (dict[float]): Dictionary of time delays [s]. Key values label the time delays.

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If numerical values are non-positive or time_delays is empty
        """
        # Validate and set integration_time
        self.integration_time = self._validate_positive_float(
            integration_time, 'integration_time', 'seconds'
        )

        # Validate and set integration_time_DM
        self.integration_time_DM = self._validate_positive_float(
            integration_time_DM, 'integration_time_DM', 'seconds'
        )

        # Validate and set sensitivity
        self.sensitivity = self._validate_positive_float(
            sensitivity, 'sensitivity', 'fractional units'
        )

        # Validate and set time_delays
        self.time_delays = self._validate_time_delays(time_delays)

    @staticmethod
    def _validate_positive_float(value, param_name, units):
        """Validate that a parameter is a positive number

        Args:
            value: The value to validate
            param_name (str): Name of the parameter for error messages
            units (str): Physical units for error messages

        Returns:
            float: The validated value

        Raises:
            TypeError: If value is not numeric
            ValueError: If value is not positive
        """
        # Check type
        if value is None:
            raise TypeError(
                f"{param_name} cannot be None. "
                f"Expected a positive number in {units}."
            )

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"{param_name} must be a number, got {type(value).__name__}. "
                f"Expected a positive number in {units}.\n"
                f"Example: {param_name}=1.0"
            )

        # Check positivity
        if value_float <= 0:
            raise ValueError(
                f"{param_name} must be positive, got {value_float}. "
                f"Expected a positive number in {units}.\n"
                f"Example: {param_name}=1.0"
            )

        return value_float

    @staticmethod
    def _validate_time_delays(time_delays):
        """Validate time_delays dictionary

        Args:
            time_delays: Dictionary of time delays to validate

        Returns:
            dict: The validated dictionary

        Raises:
            TypeError: If not a dictionary or values are not numeric
            ValueError: If dictionary is empty or has non-positive values
        """
        if not isinstance(time_delays, dict):
            raise TypeError(
                f"time_delays must be a dictionary, got {type(time_delays).__name__}.\n"
                f"Expected format: time_delays={{'day': 86400, 'year': 31536000}}"
            )

        if len(time_delays) == 0:
            raise ValueError(
                "time_delays dictionary cannot be empty.\n"
                "Expected format: time_delays={'day': 86400, 'year': 31536000}"
            )

        # Validate each entry
        validated = {}
        for key, value in time_delays.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"time_delays keys must be strings, got {type(key).__name__} for key {key}.\n"
                    f"Expected format: time_delays={{'day': 86400, 'year': 31536000}}"
                )

            try:
                value_float = float(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"time_delays['{key}'] must be a number, got {type(value).__name__}.\n"
                    f"Expected positive time delay in seconds.\n"
                    f"Example: time_delays={{'{key}': 86400}}"
                )

            if value_float <= 0:
                raise ValueError(
                    f"time_delays['{key}'] must be positive, got {value_float}.\n"
                    f"Expected positive time delay in seconds.\n"
                    f"Example: time_delays={{'{key}': 86400}}"
                )

            validated[key] = value_float

        return validated