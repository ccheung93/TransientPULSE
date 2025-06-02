import numpy as np

# CONVERSION FACTORS
SPEED_OF_LIGHT = 3e8                       # speed of light in m/s
HBAR = 6.582e-16                           # reduced Planck constant in eV*s

INEV_TO_METERS = 1.97e-7                   # eV^-1 to meters
PC_TO_METERS = 3.086e16                    # parsecs to meters
PC_TO_INEV = PC_TO_METERS/INEV_TO_METERS   # parsecs to 1/eV
EV_TO_JOULES = 1.60218e-19                 # eV to joules
EV_TO_KG = EV_TO_JOULES/SPEED_OF_LIGHT**2  # eV to kg from m = E/c^2

MASS_SUN_KG = 2e30                         # mass of the sun in kg
SOLAR_TO_EV = MASS_SUN_KG/EV_TO_KG         # eV to solar mass

SEC_TO_INEV = 1/HBAR                       # 1 second to 1/eV

DAY_TO_SEC = 60*60*24                      # number of seconds in 1 day
YEAR_TO_SEC = 365*DAY_TO_SEC               # number of seconds in 1 year

GCM3_TO_EV4 = 4.2e18                       # g/cm^3 to eV^4
PLANCK_MASS_EV = 1.2e28                    # Planck mass in eV
GIGA_TO_BASE = 1e9                         # giga to base in SI
METERS_TO_INEV = 5.07e6                    # meters to 1/eV
AVG_VEL_DM = 1e-3                          # average velocity of galactic Dark Matter

MASS_ELECTRON = 0.511e6                    # mass of electron in eV

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

ASTROPHYSICAL_CONSTRAINT_BENCHMARKS = {
    'electron': 1e-13,
    'photon': 5e-13,
    'proton': 6e-10,
    'neutron': 1.3e-9,
    'EDM': 6.7e-9
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

PI = np.pi 

def signal_duration(Etot, m_phi, energies, burst_duration, distance, aw=1, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC): 
    """ Calculate the energy density of phi at the Earth and rescaling factor calculated for an array of signal durations - NOTE - new name for this function
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        m_phi (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
        distance (float): distance between source and detector [pc]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    default = 1 for minimal uncertainty (corresponds to a Gaussian wavepacket)
        t_int (float): integration time [s], default = 1 day
        t_int_DM (float): integration time for dark matter experiment [s], default = 1 year

    Returns:
        rho (float): energy density of phi at Earth [eV^4]
        timing_rescaling_factor (array_like): rescaling factors for each energy [unitless]
    """ 
    # Convert to proper units
    t_star = burst_duration*SEC_TO_INEV
    R = distance*PC_TO_INEV
    
    # Compute q values
    q = calc_q(m_phi, energies)
    
    # Compute spread in energies from uncertainty principle
    dw = calc_energy_spread(t_star, aw=aw)
    
    # Compute signal duration
    dx_spread = calc_wave_spread(q, energies, dw, R)
    total_wave_spread = calc_total_wave_spread(t_star, dx_spread)
    
    # Compute energy density of phi at Earth
    rho = calc_rho(Etot, R, total_wave_spread) # NOTE - am i understanding right that the thickness of this spherical wave is just the signal duration calculated with the the dw/w * R/q^2 term?
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    tau_star = calc_tau_star(m_phi, q, dw, R, t_star)
    tau_DM = calc_tau_DM(energies)
    timing_rescaling_factor = calc_timing_rescaling_factor(total_wave_spread, tau_star, tau_DM, t_int=t_int, t_int_DM=t_int_DM)
    
    return rho, timing_rescaling_factor

def calc_q(m_phi, w):
    """ Calculate momentum-to-mass ratio q NOTE - confusing when also defining q=w/m_eff (Eq.36)
                 k     sqrt(w^2 - m^2)
            q = --- = ----------------- = sqrt((w/m)^2 - 1)
                 m           m

    Args:
        m_phi (float): mass of phi field [eV]
        w (array_like): individual particle energies [eV]

    Returns:
        array_like: array of q values calculated for each energy
    """
    return np.sqrt(np.maximum((w/m_phi)**2 - 1, 0))

def calc_energy_spread(t_star, aw=1):
    """ Calculate spread in energies from uncertainty principle: 
                     aw
            dw = ----------
                   t_star
    Args:
        t_star (float): the duration of the burst emission at the source [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle

    Returns:
        float: spread in energies [eV]
    """
    return aw/t_star

def calc_wave_spread(q, w, dw, R):
    """ Calculate wave spread during propagation (second term of Eq. 43)

    Args:
        q (array_like): q values [unitless]
        w (array_like): individual particle energies [eV]
        dw (float): spread in energies [eV]
        R (float): distance between source and detector [1/eV]

    Returns:
        array_like: wave spread during propagation [1/eV]
    """
    return (dw/w)*(R/np.maximum(q**2, 1e-99))

def calc_total_wave_spread(dx_burst, dx_spread):
    """ Calculate total spread of the wavepacket - NOTE - Eq. 31 vs 43, also signal duration at Earth? (Eq. 43)
            x_star_tilde = x_star + delta_x_spread (Eq. 31)
                                     dw      R
            t_star_tilde = t_star + ----- -------  (Eq. 43)
                                      w     q^2
    Args:
        dx_burst (array_like): width of wave shell at the source [1/eV]
        dx_spread (array_like): spread of wave shell due to propagation [1/eV]

    Returns:
        array_like: total spread of the wavepacket [1/eV]  NOTE - signal duration?
    """
    return dx_burst + dx_spread

def calc_tau_star(m_phi, q, dw, R, t_star):
    """ Calculate effective coherence time observed by the detector (Eq. 40)
                         2*pi          2*pi*R
            tau_star = -------- + ----------------
                          dw        q^3*m*t_star
    Args:
        m_phi (float): mass of phi field [eV]
        q (array_like): q values [unitless]
        dw (float): spread in energies from uncertainty principle [1/eV]
        R (float): distance between source and detector [1/eV]
        t_star (float): the duration of the burst emission at the source [1/eV]

    Returns:
        array_like: effective coherence times
    """
    return 2*PI/dw + 2*PI*R/(np.maximum(q**3, 1e-99) * m_phi * t_star)

def calc_tau_DM(mass_DM, v_DM=AVG_VEL_DM):
    """ Calculate effective coherence time for non-relativistic ambient DM (Eq. 46)
                           2*pi
            tau_DM = ---------------
                       m_DM*v_DM^2
    Args:
        mass_DM (array_like): mass of ambient DM
        v_DM (float): velocity of DM, default = average velocity of DM (1e-3)

    Returns:
        array_like: effective coherence times for DM
    """
    return 2*PI/(mass_DM*v_DM**2)
    
def calc_timing_rescaling_factor(signal_duration, tau_star, tau_DM, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC):
    """ Calculate timing rescaling factor (fraction from Eq. 46 in arXiv:2502.08716v1)
                    t_int_DM^(1/4) * min(tau_DM^(1/4), t_int_DM^(1/4))
    trf = ----------------------------------------------------------------------------
            min(sig_dur^(1/4), t_int^(1/4)) * min(tau_star^(1/4), t_int_star^(1/4))
    Args:
        signal_duration (array_like): signal duration at the detector
        tau_star (array_like): effective coherence times for phi field
        tau_DM (array_like): effective coherence times for DM
        t_int (float): integration time for transient search [s], default = 1 day
        t_int_DM (float): integration time for DM experiment to reach sensitivity [s], default = 1 year

    Returns:
        array_like: array of rescaling factors calculated for each signal duration [unitless]
    """
    # Convert integration time to seconds and then to inverse eV
    integration_time_s = t_int if t_int else DAY_TO_SEC
    t_int = np.full_like(signal_duration, integration_time_s * SEC_TO_INEV)
    
    # Set integration time in seconds for dark matter experiment
    integration_time_DM_s = t_int_DM if t_int_DM else YEAR_TO_SEC
    t_int_DM = np.full_like(signal_duration, integration_time_DM_s * SEC_TO_INEV)
    
    timing_rescaling_factor = [
        (t_dm**(1/4)) * min(tau_dm**(1/4), t_dm**(1/4)) /
        (min(sig_dur**(1/4), t_i**(1/4)) * min(tau_s**(1/4), t_i**(1/4)))
        for t_dm, tau_dm, sig_dur, t_i, tau_s in zip(t_int_DM, tau_DM, signal_duration, t_int, tau_star)
    ]
    
    return timing_rescaling_factor

def calc_rho(Etot, R, dx):
    """ Calculate the energy density of phi at detector in eV^4 (Eq. 30)
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        R (float): distance between source and detector [1/eV]
        dx (array_like): thickness of the spherical wave shell of phi at the detector [1/eV]

    Returns:
        float: energy density of phi at detector [eV^4]
    """ 
    return Etot/(4*PI*R**2*dx)

def coupling_probe(Etot, t, R, w, m, eta, aw=1, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC, coupling_order=None, coupling_type=None, axion=False):
    """ Calculate value of coupling we can probe

    Args:
        w (float): energy of phi emitted by the source [eV]
        rho (float): density of phi at the Earth
        timing_rescaling_factor (array of floats): rescaling factors
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        coupling_order (int): coupling order
        coupling_type (str): coupling type
        axion (boolean): calculating coupling for axions or not

    Returns:
        float: value of coupling we can probe
    """
    rho, timing_rescaling_factor = signal_duration(Etot, m, w, t, R, aw=aw, t_int=t_int, t_int_DM=t_int_DM)
    
    if axion:
        g_DM = G_DM_BENCHMARKS[coupling_type] * DEFAULT_NORMALIZATION_MULTIPLIER[coupling_type]
        v_star = np.sqrt(np.maximum(0, 1 - m**2 / w**2))
        velocity_ratio = AVG_VEL_DM/np.maximum(v_star, 1e-99) if coupling_type in ['electron', 'proton', 'neutron'] else 1
        coupling = g_DM * np.sqrt(RHO_DM_EV4/rho) * velocity_ratio * timing_rescaling_factor
    else:
        coupling_order = convert_coupling_order_str_to_int(coupling_order)
        phi = np.sqrt(2*rho)/w
        coupling = eta * (PLANCK_MASS_EV/(2*np.sqrt(PI)*phi))**coupling_order * timing_rescaling_factor

    return coupling

def convert_coupling_order_str_to_int(coupling_order):
    """ Converts a coupling order from string to integer ('linear' -> 1, 'quad' -> 2)
    
    Args:
        coupling_order (str): string coupling order

    Returns:
        int: integer coupling order
    """
    if isinstance(coupling_order, int):
        if coupling_order in [1, 2]:
            return coupling_order
        else:
            raise ValueError(f"{coupling_order} is not a valid coupling order.")

    if isinstance(coupling_order, str):
        if coupling_order == 'linear':
            return 1
        elif coupling_order == 'quad':
            return 2
        else:
            raise ValueError(f"{coupling_order} is not a valid coupling order.")

def coupling_from_Lambda(Lambda, coupling_order=None, coupling_type=None, constraint=None, multiplier=None, axion=False):
    """ Calculate coupling from Lambda for a given coupling order and type
        TODO - discuss if axion calculations belong in this function
    Args:
        Lambda (float): energy scale for new physics
        coupling_order (int): coupling order
        coupling_type (str): coupling type
        constraint (float): astrophysical constraint
        multipler (float): coupling normalization multiplier
        axion (boolean): calculating coupling for axions or not
    
    Returns:
        float: dilatonic coupling calculated from a given Lambda value
    """
    if axion:
        constraint = constraint if constraint else ASTROPHYSICAL_CONSTRAINT_BENCHMARKS[coupling_type]
        multiplier = multiplier if multiplier else DEFAULT_NORMALIZATION_MULTIPLIER[coupling_type]
        return constraint * multiplier
    else:
        return ((1/np.sqrt(4*PI))*(PLANCK_MASS_EV/Lambda))**coupling_order

def q_from_time_delay(dt, R):
    """ Calculate q from a time delay (Eq. 36) # TODO - discuss renaming this 'q' to 'omega_over_m'?

    Args:
        dt (float): time_delay [m]
        R (float): propagation distance [m]

    Returns:
        float: q value
    """
    return 1 / np.sqrt(1 - 1 / (1 + dt / R)**2)

def coupling_from_time_delay(dt, R, m, E, Dg, K, axion=False):
    """ Calculate coupling from a time delay 
        (calculates d_i^(2) from Eq.39 in arXiv:2502.08716v1)
    
    Args:
        dt (float): time delay [s]
        R (float): propagation distance [pc]
        m (float): mass of phi [eV]
        E (array_like): energies [eV]
        Dg (float): distance per galaxy of signal propagation [pc]
        K (float): energy density fraction [unitless]
        axion (boolean): calculating coupling for axions or not
        
    Returns:
        array_like: couplings from a time delay
    """
    R_meters = R * PC_TO_METERS
    dt_c = dt * SPEED_OF_LIGHT
    
    q = q_from_time_delay(dt_c, R_meters) # TODO - explain difference between q_from_time_delay and q from coupling_probe
    
    if axion:
        Em = m * q  # TODO - explain how we're defining axion coupling here
        return [0 if Ei < Em else 1e100 for Ei in E]
    
    # Galaxy number density [galaxies / Gpc^3]
    number_density = 0.006e9
    Ng = number_density * (R/GIGA_TO_BASE)**3
    
    prefactor = PLANCK_MASS_EV**2/(8*PI)
    if R < 1e5:
        # ISM regime:
        #          M_pl^2                       1
        # d2 = -------------- [ E^2 ( 1 - ------------ ) - m^2 ]
        #       8*pi*rho_ISM               (1+dt/R)^2
        #
        coupling = prefactor * (E**2/q**2 - m**2)/(RHO_ISM*K)
    else:
        # ISM+IGM regime:
        #          M_pl^2                     2E^2*dt/R - m^2
        # d2 = -------------- [ -------------------------------------------- ]
        #           8*pi         Ng*(Dg/R)*rho_ISM + (1-Ng*(Dg/R))*rho_IGM
        #
        Ng_eff = Ng**(1/3) + 1
        numer = 2*E**2*dt_c/R_meters - m**2
        denom = Ng_eff*Dg/R*RHO_ISM*K + (1-(Ng_eff*Dg/R))*RHO_IGM*K
        coupling = prefactor * numer / denom
    return coupling

def omegaoverm_noscreen(dt, R): # NOTE - this may be the same as q_from_time_delay - discuss
    """ Returns omega/m without screening (beta(x) = 0) (Solve Eq. 35 for q)
             omega              R+dt
            ------- = ------------------------
               m        sqrt(2*R*dt + dt**2)
          
    Args:
        dt (float): time delay [s]
        R (float): distance between source and detection [pc]
    
    Returns:
        omega_over_m (float): frequency to mass ratio
    """
    R = R * PC_TO_METERS
    dt = dt * SPEED_OF_LIGHT
    return (R + dt)/(np.sqrt(dt * (2*R+dt)))

def coupling_critical(E, R, rho, m, K):
    """ Calculate the critical values of the quadratic dilatonic coupling 
    
    Args:
        E (array of floats): energies [eV]
        R (float): length scale [1/eV]
        rho (float): energy density [eV^4]
        m (float): mass of phi [eV]
        K (float): energy density fraction [unitless]
        
    Returns:
        float: critical value of quadratic dilatonic coupling
    """
    return PLANCK_MASS_EV**2 / (8*PI*rho*K) * (1/(2*R)**2 + E**2 - m**2)

def E_from_uncert(burst_duration):
    """ Calculate the energy from the burst duration using the Heisenberg uncertainty relation 
    
    Args: 
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
    
    Returns:
        float: energy [eV]
    """
    return 2*PI/(burst_duration*SEC_TO_INEV)