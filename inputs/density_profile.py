class DensityProfile:
    """Class representing a density profile."""
    
    def __init__(self, file=None, function=None):
        if file:
            self.load_profile(file)
        elif function:
            self.profile = function
        else:
            raise ValueError("Must provide either file or function.")

    def load_profile(self, file):
        pass