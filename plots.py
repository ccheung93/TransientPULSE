import numpy as np
from propagation import DAY_TO_SEC, YEAR_TO_SEC

COUPLING_LABELS = {
    'linear': {
        'photon': r'$\log_{10}(d^{(1)}_e)$',
        'electron': r'$\log_{10}(d^{(1)}_{m_e})$',
        'gluon': r'$\log_{10}(d^{(1)}_g)$'
    },
    'quad': {
        'photon': r'$\log_{10}(d^{(2)}_e)$',
        'electron': r'$\log_{10}(d^{(2)}_{m_e})$',
        'gluon': r'$\log_{10}(d^{(2)}_g)$'
    },
    'axion': {
        'proton': r'$\log_{10}(g_{\phi PP}/\text{GeV}^{-1})$',
        'electron': r'$\log_{10}(g_{\phi e}/\text{GeV}^{-1})$',
        'neutron': r'$\log_{10}(g_{\phi NN}/\text{GeV}^{-1})$',
        'photon': r'$\log_{10}(g_{\phi\gamma\gamma}/\text{GeV}^{-1})$',
        'EDM': r'$\log_{10}(|g_{\phi P\gamma}|/\text{GeV}^{-2})$'
    }
}

def setup_axes(ax, formatter, coupling_order):
    """ Set up axes for subplot (i, j) """
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(np.logspace(-20,-6,7))
    ax.set_xlim(.3e-20,0.9e-6)
    if coupling_order == 'linear': 
        ax.set_ylim(1e-9,0.9e0)
    ax.tick_params(direction="in")
    
def setup_axis_labels(fig, coupling_order, coupling_type):
    shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
    shadowaxes.set_ylabel(COUPLING_LABELS[coupling_order][coupling_type], fontsize = 45)
    shadowaxes.set_xlabel(r'$\log_{10}(\omega/\rm{eV})$', fontsize= 45)
    shadowaxes.xaxis.labelpad=50
    shadowaxes.yaxis.labelpad=50
    
def add_boxed_label(ax, x, y, txt, rotation = 0, fontsize = 25, 
                    color = 'black', edgecolor = 'black', facecolor = 'white', 
                    alpha = 1 , boxstyle = 'round,pad=0.1'):
    """Adds a text box label at position (x, y)
    
    Args:
        ax (matplotlib.axes.Axes): the plot to add label to
        x, y (float): coordinates to place label
        rotation (float): angle in degrees to rotate
        fontsize (int): font size
        color (str): color
        edgecolor (str): box edge color
        facecolor (str): box fill color
        alpha (float): box transparency
        boxstyle (str): box style
    """
    ax.text(x, y, txt, 
            rotation = rotation, 
            fontsize = fontsize,
            color = color,
            bbox = dict(facecolor = facecolor,
                        alpha = alpha,
                        edgecolor = edgecolor,
                        boxstyle = boxstyle
                        )
            )
    
def plot_MICROSCOPE(ax, range_x, microscope_m):
    """ Plot MICROSCOPE EP violation limits """
    ax.plot(range_x, microscope_m, color = 'gray', linewidth = 2)
    
    # Shade in region indicating parameter space excluded by MICROSCOPE EP tests
    upper_bound = np.full_like(range_x, 1e50, dtype=float)
    ax.fill_between(range_x, microscope_m, upper_bound, color = 'gray', alpha = 0.1)
    
    # Place label for MICROSCOPE
    pos_x = 3.5e-11
    pos_y = microscope_m[0]*1.3
    ax.text(pos_x, pos_y, r'${\rm MICROSCOPE}$', color = 'k')

def plot_FifthForce(ax, range_x, fifthForce_m):
    """ Plot fifth-force limits """
    ax.plot(range_x, fifthForce_m)

def plot_coupling_from_time_delay(ax, range_x, dday, dyear):
    """ Plot the detection delays of 1 day and 1 year relative to a light-speed signal """
    ax.plot(range_x, dday, color ='tab:purple', linewidth = 2, linestyle = '--'  )
    ax.plot(range_x, dyear, color = 'tab:red', linewidth = 2, linestyle = '--'  )
    
def plot_fill_coupling_from_time_delay(ax, range_x, dday1, dday30, dyear1, dyear30):
    """ Fill in region between dilatonic coupling curves"""
    ax.fill_between(range_x, dday1, dday30, color = 'tab:purple', alpha = 0.1)
    ax.fill_between(range_x, dyear1, dyear30, color = 'tab:red', alpha = 0.1)

def label_coupling_from_time_delay(ax, pos_x, pos_y, time, color, rotation=90):
    bbox_style = dict(facecolor = 'white', 
                      alpha = 1, 
                      edgecolor = color, 
                      boxstyle = 'round,pad=.1')
    
    label = rf'$\delta t\, \gtrsim \, 1~{{\rm {time}}}~\uparrow$'

    ax.text(pos_x, pos_y, label, rotation = rotation, fontsize = 25, color = color, bbox = bbox_style)
    
def plot_fill_region(ax, fillregion_x, fillregion_y, coupling):
    """ Shade in the viable parameter space """
    ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green', alpha = 0.3)

def plot_coupling(ax, m_bench, coupling, wmp_contour):
    """ Plot the projected sensitivity of future experiments """
    ax.plot(m_bench*wmp_contour, coupling, c = 'k', linewidth = 2, alpha = 1)
    
def plot_crit_couplings(ax, range_x, d_earth, d_exp, d_atm):
    """ Plot the critical screening from the Earth, atmosphere, and experimental apparatus """
    ax.plot(range_x, d_earth, color = 'tab:blue', linewidth = 3)
    ax.plot(range_x, d_atm, color = 'tab:blue', linestyle = 'dashed')
    ax.plot(range_x, d_exp, color = 'tab:blue', linestyle = 'dotted')
    
    # Shade in the region above the critical coupling at Earth
    ax.fill_between(range_x, d_earth, 1e100, color = 'tab:blue', alpha = .05)

def plot_E_unc(ax, E_unc):
    """ Plot the scalar energy calculated from the uncertainty principle"""
    ax.axvline(E_unc, color = 'chocolate', linestyle = '--')
    
    # Shade in the region up to E_unc
    ymin = 1e-50
    ymax = 1e50
    ax.fill_betweenx([ymin, ymax], x1=ymin, x2=E_unc, color = 'chocolate', alpha = 0.1)

def label_uncertainty_exclusion(ax, E_unc=None, R=None, coupling_type=None):
    """ Label region in parameter space that is excluded due to the uncertainty principle """
    if E_unc is not None and R is not None:
        if R < 1e5:
            pos_x, pos_y = 1e-20, 1e-2
        else:
            pos_x, pos_y = E_unc/200, 1e-1
    elif coupling_type is not None:
        pos_x = 1e-19
        pos_y = {
            'photon': 3e29,
            'electron': 3e29,
            'gluon': 3e26
        }[coupling_type]
    else:
        raise ValueError('Must provide either (E_unc and R) or coupling_type.')
    
    label = r'$\omega\,t_{*} \lesssim \, 2\pi$'
    bbox_style = dict(facecolor='white', 
                      alpha = 1, 
                      edgecolor='chocolate', 
                      boxstyle='round,pad=.1')
    
    ax.text(pos_x, pos_y, label, color = 'tab:brown', bbox = bbox_style)

def plot_mass_exclusion(ax, m, coupling_order):
    """ Plot region excluded due to the scalar energy being less than its mass """
    ax.axvline(m, c = 'k', linestyle = '--')

    # Shade in the region up to the mass constraint
    if coupling_order == "linear":
        min_y = 1e-50
    elif coupling_order == "quad":
        min_y = 1e-10 # NOTE - how is this decided?
    ax.fill_between([1e-30, m], min_y, 1e50, facecolor = 'none', hatch = "/", edgecolor = 'k', alpha = 0.3)
    
def label_mass_exclusion(ax, m, coupling_order):
    """ Label region in parameter space where omega < scalar field mass """
    if m <= 1e-20: return
    
    pos_x = m/200
    pos_y = {
        "linear": 1e-7,
        "quad": 1e12
    }
    txt = r'$\omega<m_{\phi}$'
    bbox_style = dict(facecolor = 'whitesmoke',
                      alpha = 1,
                      edgecolor = 'k',
                      boxstyle = 'round,pad=.1')
    ax.text(pos_x, pos_y[coupling_order], txt, color = 'k', bbox = bbox_style)

def plot_supernova(ax, Elist, constraint, coupling_type):
    if coupling_type not in ['photon', 'electron', 'gluon']:
        raise ValueError(f"{coupling_type} is not a valid coupling type for plot_supernova.")
    
    config = {
        "photon": {
            "ylim": (.5e6, 8e32),
            "lbl_y": 3e31,
            "txt": r'${\rm Supernova}~\gamma \gamma \rightarrow \phi \phi$',
        },
        "electron": {
            "ylim": (.5e9, 5e33),
            "lbl_y": 1e32,
            "txt": r'${\rm Supernova}~e^+ e^- \rightarrow \phi \phi$',
        },
        "gluon": {
            "ylim": (.5e4, 5e30),
            "lbl_y": 3e29,
            "txt": r'${\rm Supernova}~N N \rightarrow N N \phi \phi$',
        }
    }[coupling_type]
    
    lbl_x = 1e-19
    
    constraint = np.ones_like(Elist)*constraint
    ax.set_ylim(*config["ylim"])
    ax.text(lbl_x, config["lbl_y"], config["txt"], fontsize = 30, color = 'black')
    ax.plot(Elist, constraint, color = 'gray', linewidth = 3)
    ax.fill_between(Elist, constraint, 1e100, color = 'gray', alpha = 0.1)
        
def label_critical_screening(ax, K_E, K_atm, coupling_type, filename):
    """ Label critical screening lines """
    distance_scales = ["10Mpc_", "10kpc_"]
    
    # Define labels for critical screening
    crit_screening_labels = {
        "photon": {
            "eth": r'$d_{e,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{e,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{e,{\rm crit}}^{(2)\, \rm app}$'
        },
        "electron": {
            "eth": r'$d_{m_e,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{m_e,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{m_e,{\rm crit}}^{(2)\, \rm app}$'
        },
        "gluon": {
            "eth": r'$d_{g,{\rm crit}}^{(2)\, \rm\oplus}$',
            "atm": r'$d_{g,{\rm crit}}^{(2)\, \rm atm}$',
            "exp": r'$d_{g,{\rm crit}}^{(2)\, \rm app}$'
        }
    }
    
    # Define positions of text for critical screening
    crit_screening_positions = {
        "photon": {
            "10Mpc_": {
                "eth": (3e-13, 2e12/K_E),
                "atm": (1e-12, 5e21/K_atm),
                "exp": (5e-11, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (7e-15, 2e13/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        },
        
        "electron": {
            "10Mpc_": {
                "eth": (2e-13, 2e11/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (1e-10, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (5e-15, 2e9/K_E),
                "atm": (2e-13, 5e21/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        },
        
        "gluon": {
            "10Mpc_": {
                "eth": (2e-13, 2e11/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-12, 7e25/K_E)
            },
            "10kpc_": {
                "eth": (2e-14, 2e9/K_E),
                "atm": (2e-12, 1e19/K_atm),
                "exp": (5e-11, 7e25/K_E)
            }
        }
    }
    
    for prefix in distance_scales:
        if filename.startswith(prefix):
            pos = crit_screening_positions[coupling_type][prefix]
            lbl = crit_screening_labels[coupling_type]
            ax.text(*pos["eth"], lbl["eth"], fontsize = 35, color = 'tab:blue')
            ax.text(*pos["atm"], lbl["atm"], fontsize = 35, color = 'tab:blue')
            ax.text(*pos["exp"], lbl["exp"], fontsize = 35, color = 'tab:blue')
            
def plot_parameter_list(ax, i, j, coupling_type, coupling_order, filename):
    """ Plot parameter lists in bottom right plot """
    
    # Only plot parameter list for bottom right subplot ax(1, 1)
    if (i, j) != (1, 1): return
    
    distance_scales = ["10Mpc_", "10kpc_"]
    
    bbox_style = dict(facecolor='tab:purple', 
                      alpha = 0.0, 
                      edgecolor = 'white', 
                      boxstyle='round,pad=.2')
    
    # Define positions for parameter list for different plots
    parameter_list_positions = {
        "photon": {
            "10Mpc_": (1e-11, 1e7),
            "10kpc_": (1e-11, 1e8)
        },
        "electron": {
            "10Mpc_": (1e-10, 1e10),
            "10kpc_": (1e-11, 1e10)
        },
        "gluon": {
            "10Mpc_": (1e-11, 1e5),
            "10kpc_": (1e-11, 1e7)
        }
    }
    
    # Define Etot in solar masses and R in parsecs
    parameter_list = {
        "photon": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$'+ '\n' + r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        },
        "electron": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$' + '\n' + r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        },
        "gluon": {
            "10Mpc_": r'$E_{\rm tot} = M_{\odot}$'+'\n' +r'$R = 10~{\rm Mpc}$',
            "10kpc_": r'$E_{\rm tot} = 10^{-2} M_{\odot}$' + '\n' + r'$R = 10~{\rm kpc}$'
        }
    }
    
    # Define eta_DM (sensitivity to DM signal) values for different fields
    eta_DM = {
        "photon": r'$\,\eta_{\rm DM} = 10^{-19}/5900$',
        "electron": r'$\,\eta_{\rm DM} = 10^{-17}$',
        "gluon": r'$\,\eta_{\rm DM} = 10^{-19}/10^5$'
    }

    for prefix in distance_scales:
        if filename.startswith(prefix):
            if coupling_order == "linear":
                pos = (5e-12, 4e-9) if coupling_type == "photon" else (2e-11, 4e-9)
            else:
                pos = parameter_list_positions[coupling_type][prefix]
            txt = parameter_list[coupling_type][prefix] + '\n' + eta_DM[coupling_type]
            ax.text(*pos, txt, bbox = bbox_style)