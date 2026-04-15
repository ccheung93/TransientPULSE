import os
import numpy as np
import matplotlib.pyplot as plt
import time

from calculations.constraints import *
from utils.constants import *
from utils.expt_params import *
from plotting.limits import *
from plotting.plots import *

def get_distance_label(R):
    """ Returns distance scale label for a given value in parsecs 
    
    Args:
        R (float): distance in parsecs
        
    Returns:
        distance (str): a string label representing the distance scale in kpc or Mpc
    """
    if R == 1e4:
        distance_label = '10kpc'
    elif R == 1e7:
        distance_label = '10Mpc'
    else:
        raise ValueError(f"Unsupported distance value: {R}")
    
    return distance_label

def linear_plot(ax, i, j, coupling, m, Elist, R, dday, dyear, qyear, qday, Microscope_m, FifthForce_m, E_unc, coupling_type, filename):
    """ Plots for linear coupling_order """
    plot_E_unc(ax, E_unc)
    plot_MICROSCOPE(ax, Elist, Microscope_m)
    plot_FifthForce(ax, Elist, FifthForce_m)
    plot_coupling(ax, Elist, coupling)
    plot_mass_exclusion(ax, m)
    if m > 1e-20:
        pos_x, pos_y = m/200, 1e-7
        label_mass_exclusion(ax, pos_x, pos_y, facecolor='whitesmoke')
    
    condition_mask = (Elist > E_unc) & (Elist > m * qday)
    fillregion_x = Elist[condition_mask]
    coupling_fill = coupling[condition_mask]
    fillregion_y = [Microscope_m[l] for l in range(len(fillregion_x))]
    plot_fill_region(ax, fillregion_x, fillregion_y, coupling_fill)

    plot_coupling_from_time_delay(ax, Elist, dday, 'tab:purple')
    plot_coupling_from_time_delay(ax, Elist, dyear, 'tab:red')
    label_coupling_from_time_delay(ax, m*qday/4, 1e-7, 'day', 'tab:purple')
    label_coupling_from_time_delay(ax, m*qyear/4, 1e-7, 'yr', 'tab:red')
    
    plot_parameter_list(ax, i, j, coupling_type, 'linear', filename)

def quad_plot(ax, i, j, coupling, m, Elist, d_screen_earth, d_screen_exp, d_screen_atm, dday1, dyear1, dday30, dyear30, qyear, qday, R, E_unc, K_E, K_atm, coupling_type, coupling_order, filename):
    """ Plots for quadratic coupling_order """
    plot_E_unc(ax, E_unc)
    if m > 1e-20:
        plot_mass_exclusion(ax, m)
        label_mass_exclusion(ax, m/200, 1e12, facecolor='whitesmoke')
        
    plot_coupling(ax, Elist, coupling)
    plot_crit_couplings(ax, Elist, d_screen_earth, d_screen_exp, d_screen_atm)
    label_critical_screening(ax, K_E, K_atm, coupling_type, filename)
    plot_coupling_from_time_delay(ax, Elist, dday30, 'tab:purple')
    plot_coupling_from_time_delay(ax, Elist, dyear30, 'tab:red')
    condition_mask = Elist > E_unc
    fillregion_x = Elist[condition_mask]
    coupling_fill = coupling[condition_mask]
    d_exp = d_screen_exp[condition_mask]
    
    constraint = coupling_conversion(coupling_type=coupling_type, coupling_order=coupling_order)
    plot_supernova(ax, Elist, constraint, coupling_type)
    
    if R < 1e5:
        dday30_fill = dday30[condition_mask]
        fillregion_y = np.minimum(d_exp, dday30_fill)
        
        if coupling_type == 'photon':
            label_coupling_from_time_delay(ax, 6e-16, 9e26, 'day', 'tab:purple', rotation=37)
            label_coupling_from_time_delay(ax, 3e-17, 1e27, 'yr', 'tab:red', rotation=37)
        elif coupling_type == 'electron':
            label_coupling_from_time_delay(ax, 6e-16, 8e26, 'day', 'tab:purple', rotation=39)
            label_coupling_from_time_delay(ax, 3e-17, 8.5e26, 'yr', 'tab:red', rotation=39)
        elif coupling_type == 'gluon':
            label_coupling_from_time_delay(ax, 6e-16, 5e23, 'day', 'tab:purple', rotation=38)
            label_coupling_from_time_delay(ax, 3e-17, 6e23, 'yr', 'tab:red', rotation=38)
    else:
        plot_coupling_from_time_delay(ax, Elist, dday1, 'tab:purple')
        plot_coupling_from_time_delay(ax, Elist, dyear1, 'tab:red')

        dday_fill = dday1[condition_mask]
        fillregion_y = np.minimum(d_exp, dday_fill)
        
        plot_fill_coupling_from_time_delay(ax, Elist, dday1, dday30, dyear1, dyear30)
        label_coupling_from_time_delay(ax, m*qday/4, 1e16, 'day', 'tab:purple')
        label_coupling_from_time_delay(ax, m*qyear/4, 1e16, 'yr', 'tab:red')
        
    plot_fill_region(ax, fillregion_x, fillregion_y, coupling_fill)
    plot_parameter_list(ax, i, j, coupling_type, 'quad', filename)

def plots(R, Etot, coupling_type, coupling_order, save_plots=True, show_plots=False):
    """Generate dilatonic coupling plots 

    Args:
        R (float): distance between the source and the experiment
        Etot (float): total energy of the burst [M_sun]
        coupling_type (str): type of coupling ('photon', 'electron' or 'gluon')
        coupling_order (str): coupling order ('linear' or 'quadratic')
    """
    
    # Benchmark parameters
    mass_benchmarks = [1e-21, 1e-18] # mass benchmarks in eV
    ts_benchmarks = [1, 1e2] # burst duration benchmarks in seconds
    m_bench = 1e-21 # in eV
    
    mass = [[mass_benchmarks[0], mass_benchmarks[1]],
            [mass_benchmarks[0], mass_benchmarks[1]]]
    
    ts = [[ts_benchmarks[0], ts_benchmarks[0]],
          [ts_benchmarks[1], ts_benchmarks[1]]]
    
    # Set up general parameters
    distance_label = get_distance_label(R) 
    filename = f"plots/{distance_label}_{coupling_type}_{coupling_order}_dilatoniccoupling.pdf"
    wmp_contour = np.logspace(0,30,1000)
    
    Elist = mass[0][0]*wmp_contour
    Etot = Etot * SOLAR_TO_EV

    K_space = ENERGY_DENSITY_FRACTIONS['space'][coupling_type]
    K_E = ENERGY_DENSITY_FRACTIONS['earth'][coupling_type]
    K_atm = ENERGY_DENSITY_FRACTIONS['atmosphere'][coupling_type]
    eta = DM_SENSITIVITIES[coupling_type]

    # Load constraints for linear coupling order
    if coupling_order == "linear":
        Microscope_m, FifthForce_m = load_linear_constraints(Elist)

    # Setup plot
    set_matplotlib_style()
    fig, ax = plt.subplots(2, 2, figsize = (30, 21), sharex = True, sharey = True)

    for i in range(2):
        for j in range(2):
            t = ts[i][j]
            m = mass[i][j]
            E_unc = E_from_uncert(t*SEC_TO_INEV)
            axij = ax[i][j]
            
            coupling = coupling_probe(Etot, t, R, Elist, m, eta, t_int=YEAR_TO_SEC, t_int_DM=1e6, coupling_order=coupling_order)
            
            qyear = q_from_time_delay(YEAR_TO_SEC*SPEED_OF_LIGHT, R*PC_TO_METERS)
            qday = q_from_time_delay(DAY_TO_SEC*SPEED_OF_LIGHT, R*PC_TO_METERS)
            
            if coupling_order == 'linear':
                dday30 = coupling_from_time_delay(DAY_TO_SEC, R, m, Elist, 30e3, K_space)
                dyear30 = coupling_from_time_delay(YEAR_TO_SEC, R, m, Elist, 30e3, K_space)
                
                setup_axes(axij, xlims=(.3e-20, 0.9e-6), ylims=(1e-9, 0.9))
                linear_plot(axij, i, j, coupling, m, Elist, R, dday30, dyear30, qyear, qday, Microscope_m, FifthForce_m, E_unc, coupling_type, filename)
                
            elif coupling_order == 'quad':
                d_screen_earth = coupling_critical(Elist, R_E, RHO_E, m, K_E)
                d_screen_atm = coupling_critical(Elist, R_ATM, RHO_ATM, m, K_atm)
                d_screen_exp = coupling_critical(Elist, R_EXP, RHO_EXP, m, K_E)
                
                dday1 = coupling_from_time_delay(DAY_TO_SEC, R, m, Elist, 1e3, K_space)
                dyear1 = coupling_from_time_delay(YEAR_TO_SEC, R, m, Elist, 1e3, K_space)
                
                dday30 = coupling_from_time_delay(DAY_TO_SEC, R, m, Elist, 30e3, K_space)
                dyear30 = coupling_from_time_delay(YEAR_TO_SEC, R, m, Elist, 30e3, K_space)
                
                ylims = {
                    'photon': (.5e6, 8e32),
                    'electron': (.5e9, 5e33),
                    'gluon': (.5e5, 8e30)
                }[coupling_type]
                setup_axes(axij, xlims=(.3e-20, 0.9e-6), ylims=ylims)
                quad_plot(axij, i, j, coupling, m, Elist, d_screen_earth, d_screen_exp, d_screen_atm, dday1, dyear1, dday30, dyear30, qyear, qday, R, E_unc, K_E, K_atm, coupling_type, coupling_order, filename)

            # Figure title and subplot time labels
            title = rf'$\log_{{10}}(m_{{\phi}}/\mathrm{{eV}}) = {int(np.log10(mass[0][j]))}$'
            setup_title(ax[0,j], title, 20)
            time_label = rf'$t_*$ = {int(ts[i][0])} s'
            setup_time_label(ax[i,1], time_label, 40)

    setup_axis_labels(fig, coupling_order, coupling_type)
    
    if save_plots:
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi = 300)
    if show_plots: plt.show()
    
    
if __name__ == "__main__":
    coupling_types = ['photon','electron','gluon']
    coupling_orders = ['linear','quad']
    
    # Extragalactic
    R_EG = 1e7 # pc
    E_EG = 1   # solar-masses
    
    # Galactic
    R_GC = 1e4
    E_GC = 1e-2

    start_time = time.time()
    for i in coupling_types:
        for j in coupling_orders:
            plots(R_GC,E_GC,i,j)
            plots(R_EG,E_EG,i,j)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")