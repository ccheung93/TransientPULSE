class Experiment:
    """Class representing the details of an experiment."""
    
    def __init__(self, integration_time, integration_time_DM, sensitivity, time_delays):
        """Initialize the experimental details

        Args:
            integration_time (float): Integration time for transient search [s]
            integration_time_DM (float): Integration time for DM experiment to reach sensitivity [s]
            sensitivity (float): Fractional frequency sensitivity of experiment to transient signal
            time_delays (dict[float]): Dictionary of time delays [s]. Key values label the time delays.
        """
        self.integration_time = integration_time
        self.integration_time_DM = integration_time_DM
        self.sensitivity = sensitivity
        self.time_delays = time_delays