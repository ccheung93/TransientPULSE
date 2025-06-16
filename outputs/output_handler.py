import matplotlib.pyplot as plt
import numpy as np
from inputs.source import Source
from inputs.spectrum import Spectrum
from plots import *

class OutputHandler:
    def __init__(self, plot: Plot = None, figsize=(30, 21)):
        """Initializes the OutputHandler.

        Args:
            plot (Plot, optional): Plot configuration containing axis limits. Defaults to None.
            figsize (tuple, optional): Size of the figure in inches (width, height). Defaults to (30, 21).
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
            self.fig, self.axs = plt.subplots(self.nrows, self.ncols, figsize=self.figsize, sharex=False, sharey=True)
            self.axs = np.atleast_2d(self.axs)
        else:
            self.fig, self.axs = plt.subplots(figsize=self.figsize)
    
    def plot_parameter_space(self, sources, spectra, plot, save_path=None):
        """Main entry point to generate a plot.

        Args:
            sources (Source or list[list[Source]]): Either a single Source object or a 2D list of Source objects
            spectra (Spectrum or list[list[Spectrum]]): Either a single Spectrum object or a 2D list of Spectrum objects
            plot (Plot): Plot object containing plot configurations.
            save_path (str, optional): If provided, saves the figure to this path. Defaults to None.
        """
        self._init_figure(sources)
        if isinstance(sources, list) and isinstance(sources[0], list):
            self._plot_grid(sources, spectra, plot)
        else:
            self._plot_single_panel(self.axs, sources, spectra, plot)
            setup_title(self.axs, rf'$\log_{{10}}(m_{{\phi}}/\mathrm{{eV}}) = {np.log10(sources.mass):.0f}$')
            setup_time_label(self.axs, rf'$t_*$ = {sources.tstar} s', padding=40)
        
        if plot.include_legend:
            self._add_shared_legend(plot)
            
        if save_path:
            self.fig.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
    
    def _add_shared_legend(self, plot):
        if isinstance(self.axs, np.ndarray):
            # Get handles and labels from the first subplot with valid data
            for ax in self.axs.flat:
                handles, labels = self.axs[0, 0].get_legend_handles_labels()
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
            spectra (list[list[Spectrum]]): 2D list of Spectrum objects
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
                    setup_time_label(ax, rf'$t_*$ = {source.tstar} s', padding=40)
                    
                ax.tick_params(axis='both', which='both', labelsize=25)
        
    def _plot_single_panel(self, ax, source: Source, spectrum: Spectrum, plot: Plot):
        """Plot a single panel of the parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            source (Source): Source object containing source parameters.
            spectrum (Spectrum): Spectrum object containing computed quantities.
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
        
    def _plot_exclusions(self, ax, source: Source, spectrum: Spectrum, plot: Plot):
        """Plot exclusion contraints on the parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            source (Source): Source object containing source parameters.
            spectrum (Spectrum): Spectrum object containing constraints and limits.
            plot (Plot): Plot object containing plot configurations.
        """
        plot_E_unc(ax, spectrum.E_unc)
        if not plot.include_legend:
            label_E_unc(ax, 5e-17, 1e-8)
        if self.xlims[0] < source.mass:
            plot_mass_exclusion(ax, source.mass)
            label_mass_exclusion(ax)
        if source.ULB_type == 'ALP':
            plot_constraint(ax, spectrum.constraint, source.coupling_type)
        else:
            # TODO plot_MICROSCOPE(ax, spectrum.w)
            pass
    
    def _plot_couplings(self, ax, x, spectrum: Spectrum, plot: Plot):
        """Plot the coupling curve and time delay lines.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            x (np.ndarray): Energy array.
            spectrum (Spectrum): Spectrum object containing computed couplings.
            plot (Plot): Plot object containing plot configurations.
        """
        plot_coupling(ax, x, spectrum.coupling_probe, label='coupling')
        colors = ['tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        for i, (label, coupling) in enumerate(spectrum.coupling_time_delays.items()):
            plot_coupling_from_time_delay(ax, x, coupling, colors[i], label=rf'coupling_from_dt{label}')
            if not plot.include_legend:
                x_label = x[np.argmax(coupling)]*0.15
                y_label = spectrum.constraint*1.35
                label_coupling_from_time_delay(ax, x_label, y_label, label, color=colors[i])
        
    def _plot_viable_region(self, ax, spectrum):
        """Fill the viable region of parameter space.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            spectrum (Spectrum): Spectrum object containing energies and computed couplings.
        """
        mask = spectrum.w > spectrum.E_unc
        fill_x = spectrum.w[mask]
        fill_y = np.minimum(
            np.full_like(fill_x, spectrum.constraint), 
            np.maximum.reduce(list(spectrum.coupling_time_delays.values()))[mask]
        )
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
            