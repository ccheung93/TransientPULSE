import os
import matplotlib.pyplot as plt
import numpy as np
from inputs.source import Source
from inputs.spectrum import SignalModel
from utils.constants import SOLAR_TO_EV
from .plots import *

class OutputHandler:
    def __init__(self, plot: Plot = None, figsize=None):
        """Initializes the OutputHandler.

        Args:
            plot (Plot, optional): Plot configuration containing axis limits. Defaults to None.
            figsize (tuple, optional): Size of the figure in inches (width, height).
                                      If None, automatically calculates based on grid size. Defaults to None.
        """
        self.plot = plot
        self.figsize = figsize
        self.xlims = plot.xlims if plot else None
        self.ylims = plot.ylims if plot else None

    def _init_figure(self, sources):
        """Initalizes a matplotlib figure and subplots based on whether we're plotting a grid or a single panel

        Args:
            sources (Source or list[list[Source]]): Either a single Source object or a 2D list of Source objects
        """
        set_matplotlib_style()
        if isinstance(sources, list) and isinstance(sources[0], list):
            self.nrows = len(sources)
            self.ncols = len(sources[0])

            # Calculate dynamic figure size if not provided
            if self.figsize is None:
                panel_width = 15   # Width per panel in inches
                panel_height = 10.5  # Height per panel in inches
                figsize = (self.ncols * panel_width, self.nrows * panel_height)
            else:
                figsize = self.figsize

            self.fig, self.axs = plt.subplots(self.nrows, self.ncols, figsize=figsize, sharex=False, sharey=True)
            self.axs = np.atleast_2d(self.axs)
        else:
            # For single panel, use standard size if not provided
            if self.figsize is None:
                figsize = (30, 21)
            else:
                figsize = self.figsize
            self.fig, self.axs = plt.subplots(figsize=figsize)
    
    def plot_parameter_space(self, sources, spectra, plot, save_path=None):
        """Main entry point to generate a plot.

        Args:
            sources (Source or list[list[Source]]): Either a single Source object or a 2D list of Source objects
            spectra (SignalModel or list[list[SignalModel]]): Either a single SignalModel object or a 2D list of SignalModel objects
            plot (Plot): Plot object containing plot configurations.
            save_path (str, optional): If provided, saves the figure to this path. Defaults to None.
        """
        self._init_figure(sources)
        if isinstance(sources, list) and isinstance(sources[0], list):
            self._plot_grid(sources, spectra, plot)
        else:
            self._plot_single_panel(self.axs, sources, spectra, plot)
            setup_title(self.axs, rf'$\log_{{10}}(m_{{\phi}}/\mathrm{{eV}}) = {np.log10(sources.mass):.0f}$')
            setup_time_label(self.axs, rf'$t_*$ = {sources.tstar:g} s', padding=40)
            self._plot_source_params(self.axs, sources, spectra, plot)
        
        if plot.include_legend:
            self._add_shared_legend(plot)
            
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.fig.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
    
    def _add_shared_legend(self, plot):
        if isinstance(self.axs, np.ndarray):
            # Get handles and labels from the first subplot with valid data
            for ax in self.axs.flat:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
        else:
            handles, labels = self.axs.get_legend_handles_labels()

        # Only add the legend if there are items to show
        if handles:
            self.fig.legend(handles, labels, **plot.legend_config)
        
    def _plot_grid(self, sources, spectra, plot):
        """Plot a grid of parameter space panels.

        Args:
            sources (list[list[Source]]): 2D list of Source objects
            spectra (list[list[SignalModel]]): 2D list of SignalModel objects
            plot (Plot): Plot object containing plot configurations.
        """
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[i, j]
                source = sources[i][j]
                spectrum = spectra[i][j]
                self._plot_single_panel(ax, source, spectrum, plot)
                
                if i == 0:
                    setup_title(ax, rf'$\log_{{10}}(m_{{\phi}}/\mathrm{{eV}}) = {np.log10(source.mass):.0f}$')
                if j == self.ncols - 1:
                    setup_time_label(ax, rf'$t_*$ = {source.tstar:g} s', padding=40)
                if i == self.nrows - 1 and j == self.ncols - 1:
                    self._plot_source_params(ax, source, spectrum, plot)
                    
                ax.tick_params(axis='both', which='both', labelsize=25)
        
    def _plot_single_panel(self, ax, source: Source, spectrum: SignalModel, plot: Plot):
        """Plot a single panel of the parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            source (Source): Source object containing source parameters.
            spectrum (SignalModel): SignalModel object containing computed quantities.
            plot (Plot): Plot object containing plot configurations.
        """
        x = spectrum.w
        self._setup_axes(ax, source, plot)
        self._plot_exclusions(ax, source, spectrum, plot)
        self._plot_couplings(ax, x, spectrum, plot)
        self._plot_viable_region(ax, spectrum)
    
    def _setup_axes(self, ax, source, plot):
        """Set up the axis scale, limits and labels.

        Args:
            ax (matplotlib.axes.Axes): Axis to configure.
            source (Source): Source object containing source parameters for labeling.
            plot (Plot): Plot object containing plot configurations.
        """
        if plot.exclude_mass:
            xlims = self.xlims
            self.xlims = (source.mass, xlims[1])
        setup_axes(ax, self.xlims, self.ylims)
        setup_axis_labels(self.fig, source.coupling_order, source.coupling_type)
        
    def _plot_exclusions(self, ax, source: Source, spectrum: SignalModel, plot: Plot):
        """Plot exclusion contraints on the parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            source (Source): Source object containing source parameters.
            spectrum (SignalModel): SignalModel object containing constraints and limits.
            plot (Plot): Plot object containing plot configurations.
        """
        plot_E_unc(ax, spectrum.E_unc)
        e_unc_pos = plot.label_positions.get('E_unc')
        if e_unc_pos is not None:
            label_E_unc(ax, E_unc=spectrum.E_unc, pos_x=e_unc_pos[0], pos_y=e_unc_pos[1])
        else:
            label_E_unc(ax, E_unc=spectrum.E_unc)  # Auto-position near E_unc line

        if self.xlims[0] < source.mass:
            plot_mass_exclusion(ax, source.mass)
            mass_pos = plot.label_positions.get('mass_exclusion')
            if mass_pos is not None:
                label_mass_exclusion(ax, pos_x=mass_pos[0], pos_y=mass_pos[1])
            else:
                label_mass_exclusion(ax)

        if source.ULB_type == 'ALP':
            plot_constraint(ax, spectrum.constraint, source.coupling_type)
        elif source.coupling_order == 'quad':
            plot_supernova(ax, spectrum.w, spectrum.constraint, source.coupling_type)
            if spectrum.d_screen_earth is not None:
                plot_crit_couplings(ax, spectrum.w, spectrum.d_screen_earth, spectrum.d_screen_exp, spectrum.d_screen_atm)
                crit_label_pos = plot.label_positions.get('crit_screening')
                label_critical_screening(ax, source.coupling_type, spectrum.w,
                                         spectrum.d_screen_earth, spectrum.d_screen_atm,
                                         spectrum.d_screen_exp, label_positions=crit_label_pos)
    
    def _plot_couplings(self, ax, x, spectrum: SignalModel, plot: Plot):
        """Plot the coupling curve and time delay lines.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            x (np.ndarray): Energy array.
            spectrum (SignalModel): SignalModel object containing computed couplings.
            plot (Plot): Plot object containing plot configurations.
        """
        plot_coupling(ax, x, spectrum.coupling_probe, label='coupling')
        colors = ['tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        for i, (label, coupling) in enumerate(spectrum.coupling_time_delays.items()):
            plot_coupling_from_time_delay(ax, x, coupling, colors[i], label=rf'coupling_from_dt{label}')
            if label in spectrum.coupling_time_delays_secondary:
                coupling_secondary = spectrum.coupling_time_delays_secondary[label]
                plot_coupling_from_time_delay(ax, x, coupling_secondary, colors[i])
                ax.fill_between(x, coupling, coupling_secondary, color=colors[i], alpha=0.1)
            if not plot.include_legend:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                y_mid = (ymin * ymax) ** 0.5
                vis = (x >= xmin) & (x <= xmax) & np.isfinite(coupling) & (coupling > 0)
                if np.any(vis):
                    x_vis, c_vis = x[vis], coupling[vis]
                    idx = np.argmin(np.abs(np.log10(np.clip(c_vis, 1e-300, None)) - np.log10(y_mid)))
                    x_default = x_vis[idx] * 0.25
                    if spectrum.E_unc is not None and x_default < spectrum.E_unc:
                        x_default = spectrum.E_unc * 2
                else:
                    x_default = (xmin * xmax) ** 0.5
                override = plot.label_positions.get(f'dt_{label}')
                x_label = (override[0] if override[0] is not None else x_default) if override else x_default
                y_label = (override[1] if override[1] is not None else y_mid) if override else y_mid
                label_coupling_from_time_delay(ax, x_label, y_label, label, color=colors[i])
        
    def _plot_source_params(self, ax, source: Source, spectrum: SignalModel, plot: Plot):
        """Show R, E_total, and sensitivity as a text label on the plot.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            source (Source): Source object containing source parameters.
            spectrum (SignalModel): SignalModel containing sensitivity.
            plot (Plot): Plot object; checks label_positions['source_params'] for override.
        """
        etot_msun = source.Etot / SOLAR_TO_EV
        log_etot = np.log10(etot_msun)
        if abs(log_etot - round(log_etot)) < 0.01:
            exp = int(round(log_etot))
            etot_str = (r'$E_{\rm tot} = M_{\odot}$' if exp == 0
                        else rf'$E_{{\rm tot}} = 10^{{{exp}}}\ M_{{\odot}}$')
        else:
            etot_str = rf'$E_{{\rm tot}} = {etot_msun:.2g}\ M_{{\odot}}$'

        R_pc = source.R
        if R_pc >= 1e6:
            R_str = rf'$R = {R_pc/1e6:.4g}\ {{\rm Mpc}}$'
        else:
            R_str = rf'$R = {R_pc/1e3:.4g}\ {{\rm kpc}}$'

        txt = R_str + '\n' + etot_str
        if spectrum.sensitivity is not None:
            log_eta = np.log10(spectrum.sensitivity)
            if abs(log_eta - round(log_eta)) < 0.01:
                eta_str = rf'$\eta_{{\rm DM}} = 10^{{{int(round(log_eta))}}}$'
            else:
                eta_str = rf'$\eta_{{\rm DM}} = {spectrum.sensitivity:.2g}$'
            txt += '\n' + eta_str

        override = plot.label_positions.get('source_params')
        if override:
            pos_x, pos_y = override
        else:
            _, xmax = ax.get_xlim()
            ymin, _ = ax.get_ylim()
            pos_x = xmax / 10
            pos_y = ymin * 5

        bbox_style = dict(facecolor='white', alpha=0.0, edgecolor='white', boxstyle='round,pad=0.2')
        ax.text(pos_x, pos_y, txt, bbox=bbox_style, fontsize=25, ha='right')

    def _plot_viable_region(self, ax, spectrum):
        """Fill the viable region of parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            spectrum (SignalModel): SignalModel object containing energies and computed couplings.
        """
        mask = spectrum.w > spectrum.E_unc
        fill_x = spectrum.w[mask]
        # Upper bound must be below ALL dt coupling curves (most conservative)
        dt_upper = np.minimum.reduce(list(spectrum.coupling_time_delays.values()))[mask]
        candidates = [
            np.full_like(fill_x, spectrum.constraint),
            dt_upper
        ]
        if spectrum.d_screen_exp is not None:
            candidates.append(spectrum.d_screen_exp[mask])
        fill_y = np.minimum.reduce(candidates)
        plot_fill_region(ax, fill_x, fill_y, spectrum.coupling_probe[mask])
        
    def export_quantity(self, quantity, file):
        """Export quantity to CSV file.

        Args:
            quantity (_type_): _description_
            file (str): file path
        """
        if isinstance(quantity, np.ndarray):
            if quantity.ndim != 1:
                raise NotImplementedError('Currently only supports exporting 1D numpy arrays.')
            np.savetxt(file, quantity)
            print(f'{file} has been saved.')
        else:
            raise NotImplementedError('Currently only supports exporting 1D numpy arrays.')
            