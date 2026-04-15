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

# Densities of ISM and IGM converted from g/cm^3 to eV^4
RHO_ISM_GCM3 = 1.67e-24
RHO_ISM = RHO_ISM_GCM3 * GCM3_TO_EV4
RHO_IGM_GCM3 = 1.67e-30
RHO_IGM = RHO_IGM_GCM3 * GCM3_TO_EV4

# Local dark matter density
RHO_DM_GEV_CM3 = 0.4 # GeV/cm^3
RHO_DM_EV4 = RHO_DM_GEV_CM3 * GIGA_TO_BASE * (INEV_TO_METERS * 100)**3

# Density and radius of the Earth, atmosphere and experiment
RHO_E = 5.5 * GCM3_TO_EV4
R_E = 6.371e6 * METERS_TO_INEV
RHO_ATM = 1e-3 * GCM3_TO_EV4
R_ATM = 1e4 * METERS_TO_INEV
RHO_EXP = RHO_E
R_EXP = 1 * METERS_TO_INEV
