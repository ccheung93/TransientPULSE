def load_external_limits(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            x.append(float(values[0]))
            y.append(float(values[1]))
    
    return x, y

def load_linear_constraints(Elist):
    Microscope_x, Microscope_y = load_external_limits('Linear Scalar Photon/MICROSCOPE.txt')
    FifthForce_x, FifthForce_y = load_external_limits('Linear Scalar Photon/FifthForce.txt')
    Microscope_m = [Microscope_y[0]] * len(Elist)    
    FifthForce_m = [FifthForce_y[0]] * len(Elist)
    
    return Microscope_m, FifthForce_m