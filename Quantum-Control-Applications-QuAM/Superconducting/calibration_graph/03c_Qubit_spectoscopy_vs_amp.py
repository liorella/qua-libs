"""
        QUBIT SPECTROSCOPY VS AMP
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
    operation_amplitude_factor: Optional[float] = 0.1
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 800
    frequency_step_in_mhz: float = 1
    target_peak_width: Optional[float] = 2e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(name="03a_Qubit_Spectroscopy", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
quam = QuAM.load('/usr/local/google/home/ellior/quam_state/lsp04_v01_glacier/')
# Generate the OPX and Octave configurations


# Open Communication with the QOP
qmm = quam.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = quam.active_qubits
else:
    qubits = [quam.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
for q in quam.qubits.values():
    q.xy.opx_output.full_scale_power_dbm = 4

config = quam.generate_config()
config['controllers']['con1']['fems'][2]['analog_inputs'][1]['gain_db'] = 30
for q in quam.qubits:
    config['elements'][q+'.xy']['thread'] = q
    config['elements'][q+'.resonator']['thread'] = q

config['controllers']['con1']['fems'][2]['analog_outputs'][2]['upconverters'] = {1: {'frequency': 4.2e9},
                                                                                 2: {'frequency': 4.6e9}
                                                                                 }
config['controllers']['con1']['fems'][2]['analog_outputs'][2]['band'] = 1

config['controllers']['con1']['fems'][2]['analog_outputs'][2].pop(
    'upconverter_frequency')
config['controllers']['con1']['fems'][2]['analog_outputs'][4]['upconverters'] = {1: {'frequency': 5.3e9},
                                                                                 2: {'frequency': 5.9e9},
                                                                                 }

config['controllers']['con1']['fems'][2]['analog_outputs'][4].pop(
    'upconverter_frequency')

config['elements']['q0.xy']['MWInput']['upconverter'] = 1
config['elements']['q0.xy']['intermediate_frequency'] = quam.qubits['q0'].f_01 - 4.2e9
config['elements']['q1.xy']['MWInput']['upconverter'] = 1
config['elements']['q1.xy']['intermediate_frequency'] = quam.qubits['q1'].f_01 - 5.3e9
config['elements']['q2.xy']['MWInput']['upconverter'] = 1
config['elements']['q2.xy']['intermediate_frequency'] = quam.qubits['q2'].f_01 - 4.2e9
config['elements']['q3.xy']['MWInput']['upconverter'] = 1
config['elements']['q3.xy']['intermediate_frequency'] = quam.qubits['q3'].f_01 - 5.3e9
config['elements']['q4.xy']['MWInput']['upconverter'] = 2
config['elements']['q4.xy']['intermediate_frequency'] = quam.qubits['q4'].f_01 - 4.6e9
config['elements']['q5.xy']['MWInput']['upconverter'] = 2
config['elements']['q5.xy']['intermediate_frequency'] = quam.qubits['q5'].f_01 - 5.9e9

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
amps = np.arange(0, 0.3, 0.1)
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

    for i, qubit in enumerate(qubits):
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):

                # Update the qubit frequency
                qubit.xy.update_frequency(df + config['elements'][qubit.xy.name]['intermediate_frequency'])
                qubit.align()

                # Play the saturation pulse
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
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

        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits)
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([dfs + qubit_freqs[q.name] for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # search for frequency for which the amplitude the farthest from the mean to indicate the approximate location of the peak
    shifts = np.abs((ds.IQ_abs - ds.IQ_abs.mean(dim="freq"))
                    ).idxmax(dim="freq")
    # Find the rotation angle to align the separation along the 'I' axis
    angle = np.arctan2(
        ds.sel(freq=shifts).Q - ds.Q.mean(dim="freq"),
        ds.sel(freq=shifts).I - ds.I.mean(dim="freq"),
    )
    # rotate the data to the new I axis
    ds = ds.assign(
        {"I_rot": np.real(ds.IQ_abs * np.exp(1j * (ds.phase - angle)))})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.I_rot, dim="freq", prominence_factor=5)
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

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            Pi_length = q.xy.operations["x180"].length
            used_amp = q.xy.operations["saturation"].amplitude * operation_amp
            print(
                f"Drive frequency for {q.name} is "
                f"{(result.sel(qubit = q.name).position.values + q.xy.RF_frequency) / 1e9:.6f} GHz"
            )
            fit_results[q.name]["drive_freq"] = result.sel(
                qubit=q.name).position.values + q.xy.RF_frequency
            print(
                f"(shift of {result.sel(qubit = q.name).position.values/1e6:.3f} MHz)")
            factor_cw = float(target_peak_width /
                              result.sel(qubit=q.name).width.values)
            factor_pi = np.pi / \
                (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
            print(
                f"Found a peak width of {result.sel(qubit = q.name).width.values/1e6:.2f} MHz")
            print(
                f"To obtain a peak width of {target_peak_width/1e6:.1f} MHz the cw amplitude is modified "
                f"by {factor_cw:.2f} to {factor_cw * used_amp / operation_amp * 1e3:.0f} mV"
            )
            print(
                f"To obtain a Pi pulse at {Pi_length} ns the Rabi amplitude is modified by {factor_pi:.2f} "
                f"to {factor_pi*used_amp*1e3:.0f} mV"
            )
            print(
                f"readout angle for qubit {q.name}: {angle.sel(qubit = q.name).values:.4}")
            print()
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}")
            print()
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    approx_peak = result.base_line + result.amplitude * \
        (1 / (1 + ((ds.freq - result.position) / result.width) ** 2))
    for ax, qubit in grid_iter(grid):
        # Plot the line
        (ds.assign_coords(freq_GHz=ds.freq_full /
         1e9).loc[qubit].I_rot * 1e3).plot(ax=ax, x="freq_GHz")
        # Identify the resonance peak
        ax.plot(
            abs_freqs[qubit["qubit"]] / 1e9,
            ds.loc[qubit].sel(
                freq=result.loc[qubit].position.values, method="nearest").I_rot * 1e3,
            ".r",
        )
        # Identify the width
        (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit] * 1e3).plot(
            ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
        )
        ax.set_xlabel("Qubit freq [GHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy (amplitude)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    with node.record_state_updates():
        for q in qubits:
            if not np.isnan(result.sel(qubit=q.name).position.values):

                q.xy.intermediate_frequency += float(
                    result.sel(qubit=q.name).position.values)
                prev_angle = q.resonator.operations["readout"].integration_weights_angle
                if not prev_angle:
                    prev_angle = 0.0
                q.resonator.operations["readout"].integration_weights_angle = (
                    prev_angle + angle.sel(qubit=q.name).values
                ) % (2 * np.pi)
                Pi_length = q.xy.operations["x180"].length
                used_amp = q.xy.operations["saturation"].amplitude * \
                    operation_amp
                factor_cw = float(target_peak_width /
                                  result.sel(qubit=q.name).width.values)
                factor_pi = np.pi / \
                    (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
                if factor_cw * used_amp / operation_amp < 0.5:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["saturation"].amplitude = factor_cw * \
                        used_amp / operation_amp
                else:
                    # TODO: 1 for OPX1000 MW
                    q.xy.operations["saturation"].amplitude = 0.5

                if factor_pi * used_amp < 0.3:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["x180"].amplitude = factor_pi * used_amp
                elif factor_pi * used_amp >= 0.3:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["x180"].amplitude = 0.3
    node.results["ds"] = ds

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = quam
    node.save()
    quam.save()

# %%
