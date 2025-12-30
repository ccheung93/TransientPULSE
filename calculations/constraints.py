import numpy as np
from utils.constants import * 

def signal(Etot, m_phi, energies, burst_duration, distance, aw=1, t_int=DAY_TO_SEC, t_int_DM=YEAR_TO_SEC, rho=None): 
    """ Calculate the energy density of phi at the Earth and timing rescaling factor calculated for a spectrum
    
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
    q = calc_k_over_m(m_phi, energies)
    
    # Compute spread in energies from uncertainty principle
    dw = calc_energy_spread(t_star, aw=aw)
    
    # Compute signal duration
    dx_spread = calc_wave_spread(q, energies, dw, R)
    total_wave_spread = calc_total_wave_spread(t_star, dx_spread)
    
    # Compute energy density of phi at Earth
    rho = rho if rho else calc_rho(Etot, R, total_wave_spread)
    
    # Compute rescaling factor (fraction in Eq. 46 in arXiv:2502.08716v1)
    tau_star = calc_tau_star(m_phi, q, dw, R, t_star)
    tau_DM = calc_tau_DM(energies)
    timing_rescaling_factor = calc_timing_rescaling_factor(total_wave_spread, tau_star, tau_DM, t_int=t_int, t_int_DM=t_int_DM)
    
    return rho, timing_rescaling_factor

def calc_k_over_m(m_phi, w):
    """ Calculate momentum-to-mass ratio
             k     sqrt(w^2 - m^2)
            --- = ----------------- = sqrt((w/m)^2 - 1)
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
            dw = ------------
                   2*t_star
    Args:
        t_star (float): the duration of the burst emission at the source [1/eV]
        aw (float): wavepacket uncertainty parameter based on uncertainty principle

    Returns:
        float: spread in energies [eV]
    """
    return aw/(2*t_star)

def calc_wave_spread(q, w, dw, R):
    """ Calculate wave spread during propagation (second term of Eq. 43)
                              dw      R
            delta_x_spread = ----- -------
                               w     q^2   
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
    """ Calculate total spread of the wavepacket
            x_star_tilde = x_star + delta_x_spread (Eq. 31)
    Args:
        dx_burst (array_like): width of wave shell at the source [1/eV]
        dx_spread (array_like): spread of wave shell due to propagation [1/eV]

    Returns:
        array_like: total spread of the wavepacket [1/eV]
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
    
    t_int_14 = t_int**(1/4)
    t_int_DM_14 = t_int_DM**(1/4)
    sig_dur_14 = signal_duration**(1/4)
    tau_star_14 = tau_star**(1/4)
    tau_DM_14 = tau_DM**(1/4)
    
    timing_rescaling_factor = (
        t_int_DM_14 * np.minimum(tau_DM_14, t_int_DM_14) /
        (np.minimum(sig_dur_14, t_int_14) * np.minimum(tau_star_14, t_int_14))
    )
    
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
    rho, timing_rescaling_factor = signal(Etot, m, w, t, R, aw=aw, t_int=t_int, t_int_DM=t_int_DM)
    
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

def coupling_conversion(Lambda=None, coupling_order=None, coupling_type=None, constraint=None, multiplier=None, axion=False):
    """ Converts between other common normalization conventions 
        For scalars: from energy scale (Lambda) to dilatonic coupling
        For axions: from different normalization conventions for astrophysical bounds or lab bounds

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
    elif Lambda:
        if not isinstance(coupling_order, int): 
            coupling_order = convert_coupling_order_str_to_int(coupling_order)
        return ((1/np.sqrt(4*PI))*(PLANCK_MASS_EV/Lambda))**coupling_order
    else:
        if coupling_type == 'electron':
            return 5e31
        elif coupling_type in ['photon', 'gluon']:
            default_conversions = {
                'photon': 1e12, # 1 TeV
                'gluon': 15e12  # 15 TeV
            }
            if not isinstance(coupling_order, int): 
                coupling_order = convert_coupling_order_str_to_int(coupling_order)
            return ((1/np.sqrt(4*PI))*(PLANCK_MASS_EV/default_conversions[coupling_type]))**coupling_order
        else:
            return KeyError(f"{coupling_type} is not a valid coupling type for coupling_conversion.")

def q_from_time_delay(dt, R):
    """ Calculate q from a time delay (Eq. 36), assuming constant beta

    Args:
        dt (float): time_delay [m]
        R (float): propagation distance [m]

    Returns:
        float: q value
    """
    return 1 / np.sqrt(1 - 1 / (1 + dt / R)**2)

def coupling_from_time_delay(dt, R, m, E, Dg, K, axion=False):
    """ Calculate coupling from a time delay 
        For scalars: calculates d_i^(2) from Eq.39 in arXiv:2502.08716v1
        For axions: coupling corresponds to a constant value since axions do not exhibit screening
    
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
    
    q = q_from_time_delay(dt_c, R_meters)
    
    if axion:
        # axions do not exhibit screening, so time delays correspond to a constant q value, regardless of the coupling
        Em = m * q  
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
        # Handle division by zero safely
        with np.errstate(divide='ignore', invalid='ignore'):
            coupling = prefactor * (E**2/q**2 - m**2)/(RHO_ISM*K)
            # Set coupling to inf where denominator is zero (physically means strong coupling limit)
            coupling = np.where(np.isfinite(coupling), coupling, np.inf)
    else:
        # ISM+IGM regime:
        #          M_pl^2                     2E^2*dt/R - m^2
        # d2 = -------------- [ -------------------------------------------- ]
        #           8*pi         Ng*(Dg/R)*rho_ISM + (1-Ng*(Dg/R))*rho_IGM
        #
        Ng_eff = Ng**(1/3) + 1
        numer = 2*E**2*dt_c/R_meters - m**2
        denom = Ng_eff*Dg/R*RHO_ISM*K + (1-(Ng_eff*Dg/R))*RHO_IGM*K
        with np.errstate(divide='ignore', invalid='ignore'):
            coupling = prefactor * numer / denom
            coupling = np.where(np.isfinite(coupling), coupling, np.inf)
    return coupling

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
        burst_duration (float): t_star, the duration of the burst emission at the source [1/eV]
    
    Returns:
        float: energy [eV]
    """
    return 2*PI/burst_duration

def group_velocity(m, E, rho, K, coupling):
    """ Calculate group velocity (using Eq. 20 in arXiv:2502.08716v1)
        
    Args:
        m (float): mass of phi [eV]
        E (array of floats): energies [eV]       # TODO: cannot be array?
        rho (array of floats): SM density [eV^4]
        K (float): energy density fraction [unitless]
        coupling (float): dilatonic coupling [unitless]
        
    Returns:
        vg (float): group velocity
    """
    rho = rho * K
    
    beta = (8*PI/PLANCK_MASS_EV**2) * coupling * rho
    m_eff = np.sqrt(m**2 + beta)
    q = np.asarray(E) / m_eff
    
    v_g = np.sqrt(1 - np.where(q**2 > 1, 1 / q**2, 0.0))
    
    return v_g

def propagation_time(m, E, x, rho, K, coupling):
    """ Calculate propagation time via trapezoidal integration of 1/v_g

    Integrates the inverse group velocity along the propagation path to compute
    the total travel time.

    Args:
        m (float): mass of phi [eV]
        E (float or array): energy of phi particles [eV]
        x (array): propagation distance array [1/eV]
        rho (array): SM density profile [eV^4]
        K (float): energy density fraction [unitless]
        coupling (float): dilatonic coupling [unitless]

    Returns:
        float: total propagation time [1/eV]
    """
    v_g = group_velocity(m, E, rho, K, coupling)
    x = np.asarray(x)
    v_g_inv = 1 / v_g

    # Trapezoidal integration of 1/v_g over x
    dx = np.diff(x)
    vg_avg = 0.5 * (v_g_inv[:-1] + v_g_inv[1:])

    return np.sum(vg_avg * dx)

def propagation_time_GW(R):
    """ Calculates propagation time for gravitational wave (GW)
    
    Args:
        R (float): distance between GW source and detector
    
    Returns:
        float: total propagation time [1/eV]
    """
    return R

