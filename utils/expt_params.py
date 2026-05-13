"""
Experimental parameters and benchmark values.
"""

from utils.constants import MASS_ELECTRON, GIGA_TO_BASE

# Define normalization multipliers for electron and photon couplings
DEFAULT_NORMALIZATION_MULTIPLIER = {
    'electron': 1/(2*MASS_ELECTRON/GIGA_TO_BASE), # 1/GeV
    'photon': 1/4,
    'proton': 1,
    'neutron': 1,
    'EDM': 1
}

G_DM_BENCHMARKS = {
    'electron': 2e-14,   # 1/GeV
    'photon': 3e-16,     # 1/GeV
    'proton': 1e-12,     # 1/GeV
    'neutron': 1e-11,    # 1/GeV
    'EDM': 1e-18         # 1/GeV^2
}

# Astrophysical constraint benchmarks in the convention of Eq. 5 in arXiv:2502.08716v1
ASTROPHYSICAL_CONSTRAINT_BENCHMARKS = {
    'electron': 1e-13,
    'photon': 5e-13,
    'proton': 6e-10,
    'neutron': 1.3e-9,
    'EDM': 6.7e-9
}

# Fraction of energy density of specific regimes ('space', 'earth', 'atmosphere') that is from the coupling type
ENERGY_DENSITY_FRACTIONS = {
    'space': {
        'photon': 6.3e-4,
        'electron': 4.4e-4,
        'gluon': 1
    },
    'earth': {
        'photon': 1.9e-3,
        'electron': 2.4e-4,
        'gluon': 1
    },
    'atmosphere': {
        'photon': 9.5e-4,
        'electron': 2.7e-4,
        'gluon': 1
    }
}

MICROSCOPE_PATHS = {
    'photon': 'data/Linear Scalar Photon/MICROSCOPE.txt',
    'electron': 'data/Linear Scalar Electron/MICROSCOPE.txt',
    'gluon': 'data/Linear Scalar Gluon/Microscope.txt',
}

FIFTHFORCE_PATHS = {
    'photon': 'data/Linear Scalar Photon/FifthForce.txt',
    'electron': 'data/Linear Scalar Electron/FifthForce.txt',
}

# Fractional sensitivity of the coupling type to a dark matter signal
DM_SENSITIVITIES = {
    'photon': 1e-19/6000,
    'electron': 1e-17,
    'gluon': 1e-24
}
