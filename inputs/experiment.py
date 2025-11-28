from utils.validation import validate_positive_float, validate_positive_dict_values

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
        # Validate and set integration times
        self.integration_time = validate_positive_float(
            integration_time, 'integration_time', 'seconds'
        )

        self.integration_time_DM = validate_positive_float(
            integration_time_DM, 'integration_time_DM', 'seconds'
        )

        self.sensitivity = validate_positive_float(
            sensitivity, 'sensitivity', 'fractional units'
        )

        # Validate and set time_delays dictionary
        self.time_delays = validate_positive_dict_values(time_delays, 'time_delays')
