"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 2000
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.2
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.2
    target_peak_width: Optional[float] = 2e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(name="03a_Qubit_Spectroscopy", parameters=Parameters(qubits='q0'))


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
quam = QuAM.load()
# Generate the OPX and Octave configurations


# Open Communication with the QOP
qmm = quam.connect()  # need to wrap this with a hack function that loads then sets upconverters to None

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = quam.active_qubits
else:
    qubits = [quam.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
for q in quam.qubits.values():
    q.xy.opx_output.full_scale_power_dbm = 4
    q.xy.operations.saturation.amplitude = 0.25
from utils import generate_and_fix_config
config = generate_and_fix_config(quam)

operation = node.parameters.operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
amps = np.logspace(-1, 0, 11) * operation_amp
qubit_freqs = {q.name: q.f_01 for q in qubits}  # for opx

target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        # the desired width of the response to the saturation pulse (including saturation amp), in Hz
        3e6
    )

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    amp_ = declare(fixed)

    for i, qubit in enumerate(qubits):
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(amp_, amps)):
                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    qubit.xy.update_frequency(df + config['elements'][qubit.xy.name]['intermediate_frequency'])
                    qubit.align()
                    qubit.wait(125_000)  # rethermalize time

                    # Play the saturation pulse
                    qubit.xy.play(
                        operation,
                        amplitude_scale=amp_,
                        duration=(operation_len *
                                u.ns if operation_len is not None else None),
                    )
                    qubit.align()

                    # readout the resonator
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # Wait for the qubit to decay to the ground state
                    qubit.resonator.wait(quam.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(
        duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = quam
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "amp": amps})
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits)
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
    ds['IQ'] = ds.I * 1j* ds.Q
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([dfs + qubit_freqs[q.name] for q in qubits]),
            ),
            "amps_full": (
                ["qubit", "amp"],
                np.array([20*np.log10(amps * q.xy.operations['saturation'].amplitude*operation_amp) + \
                          q.xy.opx_output.full_scale_power_dbm
                           for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"
    ds.amps_full.attrs['long_name'] = 'Output Power'
    ds.amps_full.attrs['units'] = 'dBm'
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    ds['IQ_displacement'] = np.abs(ds.IQ - np.median(ds.IQ))
    # %%
    ds.I.isel(amp=slice(1, None)).plot(col='qubit', x='freq_full', y='amps_full', cmap='jet')
    plt.gca().set_xlabel('qubit drive freq [Hz]')
    plt.gca().set_ylabel('saturation power [dBm]')

# %%
    result = peaks_dips(ds.I, dim="freq", prominence_factor=5)
    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                result.sel(qubit=q.name).position.values + qubit_freqs[q.name],
            )
            for q in qubits
        ]
    )
