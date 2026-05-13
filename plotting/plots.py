import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

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
    'ALP': {
        'proton': r'$\log_{10}(g_{\phi PP}/\text{GeV}^{-1})$',
        'electron': r'$\log_{10}(g_{\phi e}/\text{GeV}^{-1})$',
        'neutron': r'$\log_{10}(g_{\phi NN}/\text{GeV}^{-1})$',
        'photon': r'$\log_{10}(g_{\phi\gamma\gamma}/\text{GeV}^{-1})$',
        'EDM': r'$\log_{10}(|g_{\phi P\gamma}|/\text{GeV}^{-2})$'
    }
}

ALP_EXCLUSION_LABELS = {
    'proton': r'${\rm SN1987A}$',
    'electron': r'${\rm Red\ Giants\ (\omega Cen)}$',
    'neutron': r'${\rm Neutron\ star\ cooling}$',
    'photon': r'${\rm H1821+643}~\gamma \gamma \rightarrow \phi \phi$',
    'EDM': r'${\rm Planck+BAO}$'
}

class Plot:
    DEFAULT_LEGEND_CONFIG = {
        'fontsize': 30,
        'loc': 'upper right',
        'frameon': True,
        'bbox_to_anchor': (1.0, 1.0),
        'title_fontsize': 32
    }

    DEFAULT_LABEL_POSITIONS = {
        'E_unc': None,
        'mass_exclusion': None,
    }

    def __init__(self, xlims, ylims, exclude_mass=False, include_legend=True,
                 legend_config=None, label_positions=None):
        """
        Args:
            xlims (tuple): X-axis limits (min, max)
            ylims (tuple): Y-axis limits (min, max)
            exclude_mass (bool): Whether to exclude mass region
            include_legend (bool): Whether to include legend
            legend_config (dict, optional): Legend configuration
            label_positions (dict, optional): Custom label positions.
                Format: {'E_unc': (x, y), 'mass_exclusion': (x, y)}
                Use None for auto-positioning
        """
        self.xlims = xlims
        self.ylims = ylims
        self.exclude_mass = exclude_mass
        self.include_legend = include_legend
        self.legend_config = {**self.DEFAULT_LEGEND_CONFIG, **(legend_config or {})}
        self.label_positions = {**self.DEFAULT_LABEL_POSITIONS, **(label_positions or {})}
        self.handles = []
        self.labels = []

def exponentlabel(x, pos):
    return str("{:.0f}".format(np.log10(x)))

def set_matplotlib_style(math_fontset='cm', fontsize=35, font_family='STIXGeneral', hatch_color='lightgray'):
    plt.rcParams.update({
        'mathtext.fontset': math_fontset,
        'font.size': fontsize,
        'font.family': font_family,
        'hatch.color': hatch_color
    })
    
def setup_axes(ax, xlims, ylims):
    """ Set up axes for subplot (i, j) """
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    formatter = FuncFormatter(exponentlabel) 
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    log_min = int(np.ceil(np.log10(xlims[0])))
    log_max = int(np.floor(np.log10(xlims[1])))
    step = max(1, round((log_max - log_min) / 7))
    ax.set_xticks([10**i for i in range(log_min, log_max + 1, step)])
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    
    ax.tick_params(direction="in")
    plt.subplots_adjust(wspace = 0, hspace = 0)

def setup_axis_labels(fig, coupling_order, coupling_type, fontsize = 45, padding = 50):
    shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
    shadowaxes.set_ylabel(COUPLING_LABELS[coupling_order][coupling_type], labelpad = padding, fontsize = fontsize)
    shadowaxes.set_xlabel(r'$\log_{10}(\omega/\rm{eV})$', labelpad = padding, fontsize= fontsize)
    
def setup_title(ax, title, padding = 10, fontsize = 35):
    ax.set_title(title, pad = padding, fontsize = fontsize)

def setup_time_label(ax, time_label, padding = 20, fontsize = 35):
    ax.set_ylabel(time_label, labelpad = padding, rotation = 270, fontsize = fontsize)
    ax.yaxis.set_label_position("right")

def add_boxed_label(ax, x, y, txt, rotation = 0, fontsize = 25, 
                    color = 'black', facecolor = 'white', 
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
                        edgecolor = color,
                        boxstyle = boxstyle
                        )
            )
    
def plot_MICROSCOPE(ax, microscope_val):
    """ Plot MICROSCOPE EP violation limits """
    ymin, ymax = ax.get_ylim()
    if microscope_val < ymin:
        return

    ax.axhline(microscope_val, color='gray', linewidth=2)
    ax.axhspan(microscope_val, 1e100, color='gray', alpha=0.1)

    pos_y = microscope_val * 1.3
    if ymin <= pos_y <= ymax:
        _, xmax = ax.get_xlim()
        pos_x = xmax / 10
        ax.text(pos_x, pos_y, r'${\rm MICROSCOPE}$', color='k', ha='right')

def plot_FifthForce(ax, range_x, fifthForce_m):
    """ Plot fifth-force limits """
    ax.plot(range_x, fifthForce_m)

def plot_coupling_from_time_delay(ax, range_x, coupling, color, label=None):
    """ Plot the detection delays of 1 day and 1 year relative to a light-speed signal """
    ax.plot(range_x, coupling, color = color, linewidth = 2, linestyle = '--', label = label)
    
def plot_fill_coupling_from_time_delay(ax, range_x, dday1, dday30, dyear1, dyear30):
    """ Fill in region between dilatonic coupling curves"""
    ax.fill_between(range_x, dday1, dday30, color = 'tab:purple', alpha = 0.1)
    ax.fill_between(range_x, dyear1, dyear30, color = 'tab:red', alpha = 0.1)

def label_coupling_from_time_delay(ax, pos_x, pos_y, time_label, color, fontsize=25, rotation=90):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not (xmin <= pos_x <= xmax and ymin <= pos_y <= ymax):
        return
    label = rf'$\delta t\, \gtrsim \, 1~{{\rm {time_label}}}~\uparrow$'
    add_boxed_label(ax, pos_x, pos_y, label, rotation = rotation, fontsize = fontsize, color = color)
    
def plot_fill_region(ax, fillregion_x, fillregion_y, coupling):
    """ Shade in the viable parameter space """
    ax.fill_between(fillregion_x, coupling, fillregion_y, where = coupling < fillregion_y, color = 'tab:green', alpha = 0.3)

def plot_coupling(ax, Elist, coupling, label=None):
    """ Plot the projected sensitivity of future experiments """
    ax.plot(Elist, coupling, c = 'k', linewidth = 2, alpha = 1, label = label)
    
def plot_crit_couplings(ax, range_x, d_earth, d_exp, d_atm):
    """ Plot the critical screening from the Earth, atmosphere, and experimental apparatus """
    ax.plot(range_x, d_earth, color = 'tab:blue', linewidth = 3)
    ax.plot(range_x, d_atm, color = 'tab:blue', linestyle = 'dashed')
    ax.plot(range_x, d_exp, color = 'tab:blue', linestyle = 'dotted')
    
    # Shade in the region above the critical coupling at Earth
    ax.fill_between(range_x, d_earth, 1e100, color = 'tab:blue', alpha = .05)

def plot_E_unc(ax, E_unc, color='chocolate'):
    """ Plot the scalar energy calculated from the uncertainty principle"""
    ax.axvline(E_unc, color = color, linestyle = '--')
    
    # Shade in the region up to E_unc
    ax.axvspan(1e-100, E_unc, color = color, alpha = 0.1)

def label_E_unc(ax, E_unc=None, pos_x=None, pos_y=None, fontsize=35, color='tab:brown'):
    """Label the energy uncertainty exclusion region

    Args:
        ax: Matplotlib axis
        E_unc (float, optional): Energy uncertainty value. Used for auto-positioning near the exclusion line
        pos_x (float, optional): X position. If None, auto-position near E_unc line
        pos_y (float, optional): Y position. If None, auto-position at bottom
        fontsize (int): Font size
        color (str): Text color
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if pos_x is None:
        pos_x = E_unc / 3 if E_unc is not None else xmin * 5
    if pos_y is None:
        pos_y = ymin * 10

    if not (xmin <= pos_x <= xmax and ymin <= pos_y <= ymax):
        return
    add_boxed_label(ax, pos_x, pos_y, r'$\omega\,t_{*} \lesssim \, 2\pi$', fontsize=fontsize, color=color)
    
def plot_constraint(ax, constraint, coupling_type, pos_x=None, pos_y=None, fontsize=35, color='black'):
    """ Plot the astrophysical constraint"""
    ymin, ymax = ax.get_ylim()
    if constraint < ymin:
        return

    ax.axhline(constraint, color = 'gray', linewidth = 3)

    # Shade in the region up to the constraint
    ax.axhspan(constraint, 1e100, color = 'gray', alpha = 0.1)

    # Label constraint
    if pos_x is None:
        xmin, xmax = ax.get_xlim()
        pos_x = xmax / 10
    if pos_y is None:
        pos_y = constraint * 1.25
    if ymin <= pos_y <= ymax:
        ax.text(pos_x, pos_y, ALP_EXCLUSION_LABELS[coupling_type], fontsize=fontsize, color=color, ha='right')

def plot_mass_exclusion(ax, m, alpha=1):
    """ Plot region excluded due to the scalar energy being less than its mass """
    ax.axvline(m, c = 'k', linestyle = '--')
    ax.axvspan(1e-100, m, facecolor = 'none', hatch = '/', edgecolor = 'k', alpha = alpha)
    
def label_mass_exclusion(ax, pos_x=None, pos_y=None, fontsize=35, color='black', facecolor='white'):
    """ Label region in parameter space where omega < scalar field mass """
    txt = r'$\omega<m_{\phi}$'
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if pos_x is None:
        pos_x = xmin * 3
    if pos_y is None:
        pos_y = (ymin * ymax) ** 0.5  # geometric mean for log-scale axes

    if not (xmin <= pos_x <= xmax and ymin <= pos_y <= ymax):
        return
    add_boxed_label(ax, pos_x, pos_y, txt, fontsize=fontsize, color=color, facecolor=facecolor)

def plot_supernova(ax, Elist, constraint, coupling_type, x_label=None, y_label=None):
    if coupling_type not in ['photon', 'electron', 'gluon']:
        raise ValueError(f"{coupling_type} is not a valid coupling type for plot_supernova.")
    
    label = {
        'photon': r'${\rm Supernova}~\gamma \gamma \rightarrow \phi \phi$',
        'electron': r'${\rm Supernova}~e^+ e^- \rightarrow \phi \phi$',
        'gluon': r'${\rm Supernova}~N N \rightarrow N N \phi \phi$'
    }[coupling_type]
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not (ymin <= constraint <= ymax):
        return

    ax.axhline(constraint, color='gray', linewidth=3)
    ax.axhspan(constraint, 1e100, color='gray', alpha=0.1)

    if x_label is None:
        x_label = xmin * 3
    if y_label is None:
        y_label = constraint * 3
    if xmin <= x_label <= xmax and ymin <= y_label <= ymax:
        ax.text(x_label, y_label, label, fontsize=30, color='black')
        
def label_critical_screening(ax, coupling_type, w, d_screen_earth, d_screen_atm, d_screen_exp,
                             label_positions=None):
    """Label critical screening lines near their respective curves.

    Labels are placed at staggered x fractions of each curve's visible range so they
    don't overlap. Y position tracks the actual curve value.

    Args:
        label_positions (dict, optional): Override positions per key.
            Format: {'eth': (x, y), 'atm': (x, y), 'exp': (x, y)}.
            Either value may be None to keep the auto-computed coordinate.
    """
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

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lbl = crit_screening_labels[coupling_type]
    label_positions = label_positions or {}

    # Stagger labels at 20%, 50%, 80% of each curve's visible x range
    x_fracs = {'eth': 0.2, 'atm': 0.5, 'exp': 0.8}
    curves  = {'eth': d_screen_earth, 'atm': d_screen_atm, 'exp': d_screen_exp}

    for key, frac in x_fracs.items():
        curve = curves[key]
        vis = (w >= xmin) & (w <= xmax) & np.isfinite(curve) & (curve >= ymin) & (curve <= ymax)
        if not np.any(vis):
            continue
        x_vis = w[vis]
        c_vis = curve[vis]
        idx = int(frac * (len(x_vis) - 1))
        px_auto = x_vis[idx] * 0.3
        py_auto = c_vis[idx] * 4

        override = label_positions.get(key)
        px = (override[0] if override[0] is not None else px_auto) if override else px_auto
        py = (override[1] if override[1] is not None else py_auto) if override else py_auto

        if xmin <= px <= xmax and ymin <= py <= ymax:
            ax.text(px, py, lbl[key], fontsize=35, color='tab:blue')
            
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
        if prefix in filename:
            if coupling_order == "linear":
                pos = (5e-12, 4e-9) if coupling_type == "photon" else (2e-11, 4e-9)
            else:
                pos = parameter_list_positions[coupling_type][prefix]
            txt = parameter_list[coupling_type][prefix] + '\n' + eta_DM[coupling_type]
            ax.text(*pos, txt, bbox = bbox_style)