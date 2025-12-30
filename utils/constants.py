"""
Physical constants and unit conversion factors.

Centralized location for all constants used across the codebase.
"""

import numpy as np

# CONVERSION FACTORS
SPEED_OF_LIGHT = 3e8                       # speed of light in m/s
HBAR = 6.582e-16                           # reduced Planck constant in eV*s

INEV_TO_METERS = 1.97e-7                   # eV^-1 to meters
PC_TO_METERS = 3.086e16                    # parsecs to meters
PC_TO_INEV = PC_TO_METERS/INEV_TO_METERS   # parsecs to 1/eV
KPC_TO_INEV = 1000*PC_TO_INEV              # kiloparsecs to 1/eV
EV_TO_JOULES = 1.60218e-19                 # eV to joules
EV_TO_KG = EV_TO_JOULES/SPEED_OF_LIGHT**2  # eV to kg from m = E/c^2

MASS_SUN_KG = 2e30                         # mass of the sun in kg
SOLAR_TO_EV = MASS_SUN_KG/EV_TO_KG         # eV to solar mass

SEC_TO_INEV = 1/HBAR                       # 1 second to 1/eV

DAY_TO_SEC = 60*60*24                      # number of seconds in 1 day
YEAR_TO_SEC = 365*DAY_TO_SEC               # number of seconds in 1 year
YEAR = YEAR_TO_SEC                         # alias for compatibility

GCM3_TO_EV4 = 4.2e18                       # g/cm^3 to eV^4
PLANCK_MASS_EV = 1.2e28                    # Planck mass in eV
GIGA_TO_BASE = 1e9                         # giga to base in SI
METERS_TO_INEV = 5.07e6                    # meters to 1/eV
AVG_VEL_DM = 1e-3                          # average velocity of galactic Dark Matter

MASS_ELECTRON = 0.511e6                    # mass of electron in eV

PI = np.pi

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

# Fractional sensitivity of the coupling type to a dark matter signal
DM_SENSITIVITIES = {
    'photon': 1e-19/6000,
    'electron': 1e-17,
    'gluon': 1e-24
}

# Densities of ISM and IGM converted from g/cm^3 to eV^4
RHO_ISM_GCM3 = 1.67e-24
RHO_ISM = RHO_ISM_GCM3 * GCM3_TO_EV4
RHO_IGM_GCM3 = 1.67e-30
RHO_IGM = RHO_IGM_GCM3 * GCM3_TO_EV4

# Ambient dark matter density
RHO_DM_GEV_CM3 = 0.4 # GeV/cm^3
RHO_DM_EV4 = RHO_DM_GEV_CM3 * GIGA_TO_BASE * (INEV_TO_METERS * 100)**3

# Density and radius of the Earth, atmosphere and experiment
RHO_E = 5.5 * GCM3_TO_EV4
R_E = 6.371e6 * METERS_TO_INEV
RHO_ATM = 1e-3 * GCM3_TO_EV4
R_ATM = 1e4 * METERS_TO_INEV
RHO_EXP = RHO_E
R_EXP = 1 * METERS_TO_INEV
