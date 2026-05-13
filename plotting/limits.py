from utils.expt_params import MICROSCOPE_PATHS, FIFTHFORCE_PATHS

def load_external_limits(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if not values or values[0].startswith('#'):
                continue
            x.append(float(values[0]))
            y.append(float(values[1]))

    return x, y

def load_microscope_value(coupling_type):
    """Return the flat MICROSCOPE EP-violation constraint for the given coupling type."""
    path = MICROSCOPE_PATHS[coupling_type]
    _, y = load_external_limits(path)
    return y[0]

def load_fifthforce_value(coupling_type):
    """Return the flat fifth-force constraint for the given coupling type, or None if unavailable."""
    path = FIFTHFORCE_PATHS.get(coupling_type)
    if path is None:
        return None
    _, y = load_external_limits(path)
    return y[0]

def load_linear_constraints(Elist):
    Microscope_x, Microscope_y = load_external_limits('data/Linear Scalar Photon/MICROSCOPE.txt')
    FifthForce_x, FifthForce_y = load_external_limits('data/Linear Scalar Photon/FifthForce.txt')
    Microscope_m = [Microscope_y[0]] * len(Elist)
    FifthForce_m = [FifthForce_y[0]] * len(Elist)

    return Microscope_m, FifthForce_m