"""
Configuration classes for the new architecture.

Provides clean separation of physics parameters and propagation environment settings.
"""


class PhysicsConfig:
    """Configuration for physics parameters only"""

    def __init__(self, mass, coupling, K, burst_duration=None):
        """
        Args:
            mass (float): Scalar field mass [eV]
            coupling (float): Dilatonic coupling strength [dimensionless]
            K (float): Energy density fraction for coupling type
            burst_duration (float, optional): Duration of emission at source [s]
        """
        self.mass = mass
        self.coupling = coupling
        self.K = K
        self.burst_duration = burst_duration


class PropagationConfig:
    """Configuration for propagation environment"""

    def __init__(self, density_profile_path, density_num_points=1000):
        """
        Args:
            density_profile_path (str): Path to density profile CSV file
            density_num_points (int): Number of interpolation points for density profile
        """
        self.density_profile_path = density_profile_path
        self.density_num_points = density_num_points
