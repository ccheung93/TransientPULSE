"""
Waveform propagation package for scalar field simulations.

This package contains the core implementation for time-dependent waveform
propagation through galactic density profiles, including spectrogram generation.
"""

from waveform.collection import WaveformCollection
from waveform.propagation import propagation, plot_spectrogram, plot_waveform

__all__ = ['WaveformCollection', 'propagation', 'plot_spectrogram', 'plot_waveform']
