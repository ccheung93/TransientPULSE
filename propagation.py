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
    'electron': 1/(2*MASS_ELECTRON/GIGA_TO_BASE),
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

def signal_duration(Etot, m_phi, energies, burst_duration, distance, aw, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC, axion=False): 
    """ Calculate the energy density of phi at the Earth and rescaling factor calculated for an array of signal durations
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        m_phi (float): mass of phi field [eV]
        energies (array_like): array of individual particle energies [eV]
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
        distance (float): distance between source and detector [pc]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty
        t_int (float): integration time [s], default = 1 day
        t_int_DM (float): integration time for dark matter experiment [s], default = 1 year

    Returns:
        rho (float): energy density of phi at Earth [eV^4]
        timing_rescaling_factor (array_like): rescaling factors for each energy [unitless]
    """ 
    # Convert to proper units
    t_star = burst_duration*SEC_TO_INEV
    R = distance*PC_TO_INEV
    
    # Compute energy density of phi at Earth
    rho = calc_rho(Etot, m_phi, energies, t_star, R, aw, axion)
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    timing_rescaling_factor = calc_timing_rescaling_factor(m_phi, energies, t_star, R, aw, t_int=t_int, t_int_DM=t_int_DM, axion=axion)
    
    return rho, timing_rescaling_factor

def calc_timing_rescaling_factor(m_phi, w, t_star, R, aw, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC, axion=False):
    """ Calculate timing rescaling factor (fraction from Eq. 46 in arXiv:2502.08716v1)
                    t_int_DM^(1/4) * min(tau_DM^(1/4), t_int_DM^(1/4))
    trf = ----------------------------------------------------------------------------
            min(sig_dur^(1/4), t_int^(1/4)) * min(tau_star^(1/4), t_int_star^(1/4))
    Args:
        m_phi (float): mass of phi field [eV]
        w (array_like): array of individual particle energies [eV]
        t_star (float): the duration of the burst emission at the source [1/eV]
        R (float): distance between source and detector [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty
        t_int (float): integration time [s], default = 1 day
        t_int_DM (float): integration time for dark matter experiment [s], default = 1 year

    Returns:
        array_like: array of rescaling factors calculated for each signal duration [unitless]
    """
    # Convert integration time to seconds and then to inverse eV
    integration_time_s = t_int if t_int else DAY_TO_SEC
    t_int = np.full_like(w, integration_time_s * SEC_TO_INEV)
    
    # Set integration time in seconds for dark matter experiment
    integration_time_DM_s = t_int_DM if t_int_DM else YEAR_TO_SEC
    t_int_DM = np.full_like(w, integration_time_DM_s * SEC_TO_INEV)
    
    # Spread in energies from uncertainty principle: dw * t_star = aw -> dw = aw / t_star
    dw = aw/t_star
    
    # Spread of the phi wave during propagation
    q = w/m_phi
    dx_spread = (dw/w)*(R/q**2)
    
    # Calculate t_star_tilde, the signal duration at Earth (Eq. 43)
    signal_duration = np.sqrt(t_star**2 + dx_spread**2)

    # Calculate effective coherence time observed by the detector (Eq. 40)
    #              2*pi          2*pi*R
    # tau_star = -------- + ----------------
    #               dw        q^3*m*t_star
    tau_star = 2*PI/dw + 2*PI*R/(q**3 * m_phi * t_star)
    
    # Calculate coherence times for non-relativistic, ambient DM (Eq. 46)
    #            2*pi
    # tau_DM = ----------, m_DM = w
    #           m_DM*v^2
    mass_DM = w
    tau_DM = 2*PI/(mass_DM*AVG_VEL_DM**2)
    
    timing_rescaling_factor = [
        (t_dm**(1/4)) * min(tau_dm**(1/4), t_dm**(1/4)) /
        (min(sig_dur**(1/4), t_i**(1/4)) * min(tau_s**(1/4), t_i**(1/4)))
        for t_dm, tau_dm, sig_dur, t_i, tau_s in zip(t_int_DM, tau_DM, signal_duration, t_int, tau_star)
    ]
    
    return timing_rescaling_factor

def calc_rho(Etot, m_phi, w, t_star, R, aw, axion=False):
    """ Calculate the energy density of phi at Earth in eV^4
    
    Args:
        Etot (float): total energy of emitted phi particles from the source [eV]
        m_phi (float): mass of phi field [eV]
        w (array_like): array of individual particle energies [eV]
        t_star (float): the duration of the burst emission at the source [1/eV]
        R (float): distance between source and detector [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle:
                    dw * t_star >= 1, where dw = spread in energies and t_star = burst_duration
                    for a given wavepacket, aw = dw * t_star, where aw >= 1
                    aw = 1 corresponds to a Gaussian wavepacket, since it has minimum uncertainty

    Returns:
        float: energy density of phi at Earth [eV^4]
    """ 
    # Spread in energies from uncertainty principle: dw * t_star = aw -> dw = aw / t_star
    dw = aw/t_star
    
    # Physical size of the phi wave at the source, assuming it's traveling at c = 1.
    dx_burst = t_star
    
    # Spread of the phi wave during propagation
    q = np.sqrt((w/m_phi)**2-1) if axion else np.sqrt(np.maximum((w/m_phi)**2 - 1, 0))
    dx_spread = (dw/w)*(R/np.maximum(q**2, 1e-99))
    
    # Total spread of the wavepacket
    dx = dx_burst + dx_spread
    
    # Energy density of phi at the detector location (Eq. 30)
    rho = Etot/(4*PI*R**2*dx)
    
    return rho

def coupling_probe(w, m, rho, timing_rescaling_factor, eta, coupling_order=None, coupling_type=None, axion=False):
    """ Calculate value of dilatonic coupling we can probe

    Args:
        w (float): energy of phi emitted by the source
        rho (float): density of phi at the Earth
        timing_rescaling_factor (array of floats): rescaling factors
        eta (float): fractional sensitivity of coupling_type to dark matter signal
        coupling_order (int): coupling order
        coupling_type (str): coupling type
        axion (boolean): calculating coupling for axions or not

    Returns:
        float: value of dilatonic coupling we can probe
    """
    if axion:
        g_DM = G_DM_BENCHMARKS[coupling_type] * DEFAULT_NORMALIZATION_MULTIPLIER[coupling_type]
        v_star = np.sqrt(1-m**2/w**2)
        velocity_ratio = AVG_VEL_DM/v_star if coupling_type in ['electron', 'proton', 'neutron'] else 1
        coupling = g_DM * np.sqrt(RHO_DM_EV4/rho) * velocity_ratio * timing_rescaling_factor
    else:
        phi = np.sqrt(2*rho)/w
        coupling = eta * (PLANCK_MASS_EV/(2*np.sqrt(PI)*phi))**coupling_order * timing_rescaling_factor

    return coupling

def coupling_from_Lambda(Lambda, coupling_order=None, coupling_type=None, constraint=None, multiplier=None, axion=False):
    """ Calculate coupling from Lambda for a given coupling order and type
    
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

def coupling_from_delta_t(dt, R, m, E, Dg, K, axion=False):
    """ Calculate value of quadratic dilatonic coupling from a time delay 
        (calculates d_i^(2) from Eq.39 in arXiv:2502.08716v1)
    
    Args:
        dt (float): time delay [s]
        R (float): propagation distance [pc]
        m (float): mass of phi [eV]
        E (array of floats): energies [eV]
        Dg (float): distance per galaxy of signal propagation [pc]
        K (float): energy density fraction [unitless]
        axion (boolean): calculating coupling for axions or not
        
    Returns:
        float: value of quadratic dilatonic coupling from a time delay
    """
    if axion:
        moverE = np.sqrt(1 - 1 / (1 + (dt * SPEED_OF_LIGHT) / (R * PC_TO_METERS))**2)
        Em = m / moverE
        return [0 if Ei < Em else 1e100 for Ei in E]
    
    # Galaxy number density [galaxies / Gpc^3]
    number_density = 0.006e9
    Ng = number_density * (R/GIGA_TO_BASE)**3
    
    prefactor = PLANCK_MASS_EV**2/(8*PI)
    R_meters = R * PC_TO_METERS
    dt_c = dt * SPEED_OF_LIGHT
    
    # ISM regime:
    #          M_pl^2                       1
    # d2 = -------------- [ E^2 ( 1 - ------------ ) - m^2 ]
    #       8*pi*rho_ISM               (1+dt/R)^2
    #
    if R < 1e5:
        coupling = prefactor * (E**2*(1 - 1/(1 + dt_c/R_meters)**2) - m**2)/(RHO_ISM*K)
    # ISM+IGM regime:
    #          M_pl^2                     2E^2*dt/R - m^2
    # d2 = -------------- [ -------------------------------------------- ]
    #           8*pi         Ng*(Dg/R)*rho_ISM + (1-Ng*(Dg/R))*rho_IGM
    #
    else:
        Ng_eff = Ng**(1/3) + 1
        numer = 2*E**2*dt_c/R_meters - m**2
        denom = Ng_eff*Dg/R*RHO_ISM*K + (1-(Ng_eff*Dg/R))*RHO_IGM*K
        coupling = prefactor * numer / denom
    return coupling

def omegaoverm_noscreen(dt, R):
    """ Returns omega/m without screening (beta(x) = 0)
        omega           R+dt
        ----- = ---------------------
          m     sqrt(2*R*dt + dt**2)
          
    Args:
        dt (float): time delay 
        R (float): distance between source and detection
    
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
    coupling = PLANCK_MASS_EV**2 / (8*PI*rho*K) * (1/(2*R)**2 + E**2 - m**2)
    return coupling

def E_from_uncert(burst_duration):
    """ Calculate the energy from the burst duration using the Heisenberg uncertainty relation 
    
    Args: 
        burst_duration (float): t_star, the duration of the burst emission at the source [s]
    
    Returns:
        float: energy [eV]
    """
    return 2*PI/(burst_duration*SEC_TO_INEV)