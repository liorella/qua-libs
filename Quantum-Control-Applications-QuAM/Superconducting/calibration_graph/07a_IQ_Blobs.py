"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 2000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    operation_name: str = "readout"  # or "readout_QND"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    measure_thermal_population: bool = False


node = QualibrationNode(name="07a_IQ_Blobs", parameters=Parameters(
    num_runs=10000,
    reset_type_thermal_or_active='active',
    # qubits=[
    #     'q0', 'q1', 'q2',
    #         #  'q3',
    #            'q4'
    #            ]
))


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
quam = QuAM.load()
# Open Communication with the QOP
qmm = quam.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = quam.active_qubits
else:
    qubits = [quam.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# for q in quam.qubits.values():
#     q.resonator.operations['readout'].amplitude = 0.075
#     q.resonator.operations['readout'].length = 1000
#     q.active_reset_available = True
# quam.qubits['q3'].resonator.intermediate_frequency = quam.qubits['q3'].resonator.intermediate_frequency + 2.5e6
# quam.qubits['q3'].active_reset_available = False
# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation_name = node.parameters.operation_name

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for i, qubit in enumerate(qubits):
        align()
        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
            qubit.reset(reset_type)
            qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
            # save data
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])

            qubit.align()
            # excited iq blobs for all qubits
            qubit.reset(reset_type)
            if not node.parameters.measure_thermal_population:
                qubit.xy.play("x180")
            align()
            qubit.resonator.measure(operation_name, qua_vars=(I_e[i], Q_e[i]))
            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
            if not node.parameters.measure_thermal_population:
                qubit.xy.play("x180")
            # save data
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")


# %% {Simulate_or_execute}
from utils import generate_and_fix_config, print_qubit_params
config = generate_and_fix_config(quam)
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = quam
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_runs, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"N": np.linspace(1, n_runs, n_runs)})

    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])
    # %% {Data_analysis}
    n_bins = 100
    hist_I_g = {}
    hist_Q_g = {}
    hist_I_e = {}
    hist_Q_e = {}
    
    node.results = {"ds": ds, "figs": {}, "results": {}}
    ds['IQ_g'] = ds.I_g + 1j * ds.Q_g
    ds['IQ_e'] = ds.I_e + 1j * ds.Q_e
    
    plot_individual = False
    for q in qubits:
        # Perform two state discrimination
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            ds.I_g.sel(qubit=q.name),
            ds.Q_g.sel(qubit=q.name),
            ds.I_e.sel(qubit=q.name),
            ds.Q_e.sel(qubit=q.name),
            True,
            b_plot=plot_individual,
        )
        # TODO: check the difference between the above and the below
        # Get the rotated 'I' quadrature
        IQ_g_rot = ds.IQ_g.sel(qubit=q.name) * np.exp(1j * angle)
        IQ_e_rot = ds.IQ_e.sel(qubit=q.name) * np.exp(1j * angle)
        # Get the blobs histogram along the rotated axis
        hist_I_g[q.name] = np.histogram(np.real(IQ_g_rot), bins=n_bins)
        hist_Q_g[q.name] = np.histogram(np.imag(IQ_g_rot), bins=n_bins)
        hist_I_e[q.name] = np.histogram(np.real(IQ_e_rot), bins=n_bins)
        hist_Q_e[q.name] = np.histogram(np.imag(IQ_e_rot), bins=n_bins)

        # Get the discriminating threshold along the rotated axis
        RUS_threshold = hist_I_g[q.name][1][1:][np.argmax(hist_I_g[q.name][0])]
        RUS_pi_threshold = hist_I_e[q.name][1][1:][np.argmax(hist_I_e[q.name][0])]
        # Save the individual figures if requested
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name] = {}
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)
        node.results["results"][q.name]["rus_pi_threshold"] = float(RUS_pi_threshold)

    # %% {Plotting}
    n_avg = n_runs // 2
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qn = qubit["qubit"]
        # TODO: maybe wrap it up in a function plot_IQ_blobs?
        ang = node.results["results"][qn]["angle"]
        IQ_g_rot = ds.IQ_g.sel(qubit=qn) * np.exp(1j * ang)
        IQ_e_rot = ds.IQ_e.sel(qubit=qn) * np.exp(1j * ang)
        ax.plot(1e3 * np.real(IQ_g_rot), 1e3 * np.imag(IQ_g_rot), ".", alpha=0.2, label="Ground", markersize=1)
        ax.plot(1e3 * np.real(IQ_e_rot), 1e3 * np.imag(IQ_e_rot), ".", alpha=0.2, label="Excited", markersize=1)
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_threshold"],
            color="k",
            linestyle="--",
            lw=0.5,
            label="RUS Threshold",
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_pi_threshold"],
            color="b",
            linestyle="--",
            lw=0.5,
            label="RUS Pi Threshold",
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["threshold"],
            color="r",
            linestyle="--",
            lw=0.5,
            label="Threshold",
        )
        ax.axis("equal")
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.set_title(qubit["qubit"])

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    plt.tight_layout()
    node.results["figure_IQ_blobs"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        confusion = node.results["results"][qubit["qubit"]]["confusion_matrix"]
        ax.imshow(confusion, vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels=["|g>", "|e>"])
        ax.set_yticklabels(labels=["|g>", "|e>"])
        ax.set_ylabel("Prepared")
        ax.set_xlabel("Measured")
        ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
        ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
        ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
        ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
        ax.set_title(qubit["qubit"])

    grid.fig.suptitle("g.s. and e.s. fidelity")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelity"] = grid.fig

    # %%
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qn = qubit['qubit']
        ax.semilogy(1e3*hist_I_g[qn][1][1:], hist_I_g[qn][0] / n_runs, label="Ground")
        ax.semilogy(1e3*hist_I_e[qn][1][1:], hist_I_e[qn][0] / n_runs, label='Excited')
        ax.set_title(qn)
        ax.set_xlabel('I [mV]')
        ax.grid(which='both', axis='y')
    
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_threshold"],
            color="k",
            linestyle="--",
            lw=0.5,
            label="RUS Threshold",
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_pi_threshold"],
            color="b",
            linestyle="--",
            lw=0.5,
            label="RUS Pi Threshold",
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["threshold"],
            color="r",
            linestyle="--",
            lw=0.5,
            label="Threshold",
        )
    grid.fig.suptitle('I quad histogram')
    grid.fig.tight_layout()
    node.results['hist_I'] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qn = qubit['qubit']
        ax.semilogy(1e3*hist_Q_g[qn][1][1:], hist_Q_g[qn][0] / n_runs, label="Ground")
        ax.semilogy(1e3*hist_Q_e[qn][1][1:], hist_Q_e[qn][0] / n_runs, label='Excited')
        ax.set_title(qn)
        ax.set_xlabel('Q [mV]')
        ax.grid(which='both', axis='y')


    grid.fig.suptitle('Q quad histogram')
    grid.fig.tight_layout()
    node.results['hist_Q'] = grid.fig

    print_qubit_params(quam, [['resonator', 'operations', 'readout', 'amplitude'],
                              ['resonator', 'operations', 'readout', 'length'],
                              ['resonator', 'opx_output', 'full_scale_power_dbm']])

    # %% {Update_state}
    with node.record_state_updates():
        for qubit in qubits:
            qubit.resonator.operations[operation_name].integration_weights_angle -= float(
                node.results["results"][qubit.name]["angle"]
            )
            # Convert the thresholds back in demod units
            qubit.resonator.operations[operation_name].threshold = (
                float(node.results["results"][qubit.name]["threshold"])
                * qubit.resonator.operations[operation_name].length
                / 2**12
            )
            # todo: add conf matrix to the readout operation rather than the resonator
            qubit.resonator.operations[operation_name].rus_exit_threshold = (
                float(node.results["results"][qubit.name]["rus_threshold"])
                * qubit.resonator.operations[operation_name].length
                / 2**12
            )
            qubit.resonator.rus_pi_threshold = (
                float(node.results["results"][qubit.name]["rus_pi_threshold"])
                * qubit.resonator.operations[operation_name].length
                / 2**12
            )
            if operation_name == "readout":
                qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

    # %% {Save_results}
    node.results['ds'] = ds.drop_vars(['IQ_g', 'IQ_e'])
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = quam
    node.save()

# %%
