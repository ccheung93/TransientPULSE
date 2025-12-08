import numpy as np
import scipy.interpolate
from constants import KPC_TO_INEV, GCM3_TO_EV4

def read_medium_data(filename, i_R=0, i_rho=1):
    """Read density (g/cm^3) vs. position (kpc) data from CSV file

    Args:
        filename (str): File name
        i_R (int): Column index for position data (kpc)
        i_rho (int): Column index for density data (g/cm^3)

    Returns:
        x (array): Position [eV^-1]
        rho (array): Density [eV^4]
    """
    x = []
    rho = []

    with open(filename, 'r') as file:
        next(file)  # Skip header row
        for line in file:
            # Split by comma and remove empty fields
            parts = [p.strip() for p in line.split(',') if p.strip()]

            # Ensure we have enough columns
            if len(parts) > max(i_R, i_rho):
                try:
                    x.append(float(parts[i_R]))
                    rho.append(float(parts[i_rho]))
                except (ValueError, IndexError):
                    continue  # Skip invalid lines

    return np.array(x), np.array(rho)


def interpolate_data(x, y, num):
    """Interpolates data with linearly spacing

    Creates linearly-spaced interpolation points between the first and last
    x values, then linearly interpolates the y values at these points.

    Args:
        x (array): Input x values (must be positive for linear spacing)
        y (array): Input y values corresponding to x
        num (int): Number of points for interpolation

    Returns:
        tuple: (xnew, ynew) - Interpolated x and y arrays with num points each
    """
    # Create linearly-spaced x values
    xnew = np.linspace(x[0], x[-1], num)

    # Interpolate y values at new x positions
    f = scipy.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
    ynew = f(xnew)

    return xnew, ynew
