# TransientPULSE

Transient PULSE: Propagation of UltraLight (pseudo)Scalars to Earth

Transient PULSE is a software package written in Python that characterizes how transient signals of ultralight bosonic (ULB) fields change during their propagation to Earth. The code allows users to input an emission energy spectrum and a density profile for the medium of Standard Model matter through which the ULB fields propagate. The emission spectrum can be given as a CSV file or an analytical expression. Then, the code propagates the momentum modes within the emission spectrum from the source to the Earth, with the final result giving a frequency versus time decomposition of the signal, known as a spectrogram.

Furthermore, there are a variety of parameters that users can adjust to explore different types of events and particle physics models. A user can also produce experimental reach plots, with simplified rescaling of bounds and projections of dark matter searches to transient signals of the type simulated in the code package.

Transient PULSE is developed and maintained at the University of Delaware.

## Installation

```bash
# Clone repository
git clone https://github.com/ccheung93/TransientPULSE.git
cd TransientPULSE

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The virtual environment isolates package versions to ensure reproducibility. The `requirements.txt` file specifies the tested version ranges for each dependency. Requires Python 3.8 or later.

## Usage

Transient PULSE provides two complementary workflows:

- **Waveform propagation** -- numerically propagates a source emission spectrum through a density profile to build the frequency-time spectrogram of the signal at Earth. See `examples_propagation.ipynb` for a walkthrough with a time-independent and a time-dependent Gaussian source, or `examples_propagation.py` for the equivalent Python script.
- **Constraint plotting** -- analytically derives detection-threshold, time-delay, and screening couplings from source and experiment parameters to produce exclusion/sensitivity plots. See `examples_constraint_plots_manual.ipynb` for single-source and grid-plot examples, or `examples_constraint_plots.py` for the equivalent Python script.

`examples_complete_workflow.ipynb` demonstrates the full pipeline end to end, from an input spectrum, through propagation, to a constraint plot.

## License

Transient PULSE is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use Transient PULSE in your research, please cite:

J. Arakawa, C. Cheung, M. H. Zaheer, V. Takhistov, and M. S. Safronova, "Transient PULSE: Code for the Transient Propagation of UltraLight (pseudo)Scalars to Earth," submitted to Computer Physics Communications.

(Full citation details, including DOI, will be added once the preprint/paper is published.)

## Contributing

All contributions to the Transient PULSE package are welcome. Please open an issue or pull request on GitHub.
