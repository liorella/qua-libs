"""
find the frequency and amplitude for playing an X gate between 1 and 2 states
"""


# %% {Imports}
from utils import generate_and_fix_config, print_qubit_params
import pyle.samples.readout_helpers as readout_helpers
import pandas as pd
import pyle.dataking.datataker_v2.datataker_base as datataker_base
import pyle.datavault.public as pdv
import pyle.datavault.client as dv_client
import getpass
from pyle.dataking.datataker_v2.ac_stark_spectroscopy import CombinedACStarkChiAnalysis
import pyle.samples.keys as keys
from tunits import GHz, Hz, dBm, MHz
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam.components import pulses

# %% {Node_parameters}


class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    min_amp_factor: float = 0.0
    max_amp_factor: float = 1.5
    amp_factor_step: float = 0.02
    freq_offset_start_hz: float = 20e6
    freq_offset_stop_hz: float = -700e6
    freq_offset_step_hz: float = -1e6
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    transitions_from_zero_state: bool = False
    wideband_single_qubit_mode: bool = True


node = QualibrationNode(name="08c_Pi_12", parameters=Parameters(
    qubits=[
        # 'q0',
        # 'q1',
        'q2',
        # 'q3',
        # 'q4'
    ]
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

# %% add pulses

for qn, q in quam.qubits.items():
    q.xy.operations['x180_12'] = pulses.DragCosinePulse(
        amplitude=1,
        alpha=0.0,
        anharmonicity=0,
        length=140,
        axis_angle=0,
        detuning=0,
        digital_marker="ON")
# qubits['q0'].resonator.intermediate_frequency =  -197180861.435524 - 2e6
# qubits['q1'].resonator.intermediate_frequency =  -100829215.06058693 - 2e6
# qubits['q2'].resonator.intermediate_frequency =  -166334403.52182102 - 2e6
# qubits['q3'].resonator.intermediate_frequency =  -82862742.7987318 - 2e6
# qubits['q4'].resonator.intermediate_frequency =  -142087366.62756538 - 2e6
# qubits['q5'].resonator.intermediate_frequency =  -50440263.15789032 - 2e6

for q in quam.qubits.values():
    min_if = q.xy.intermediate_frequency + node.parameters.freq_offset_stop_hz
    if min_if < -400e6:
        print(
            f'WARNING: trying to work with {min_if=}<-400e6 for qubit {q.name}')
# %% {QUA_program}

if len(qubits) == 1 and node.parameters.wideband_single_qubit_mode:
    qubit = qubits[0]
    upconv = qubit.xy.upconverter
    qubit.xy.opx_output.upconverters[upconv] = qubit.f_01 - 300e6
    qubit.xy.intermediate_frequency = 300e6
    print(f'setting {qubit.name} IF to 300MHz')

n_avg = node.parameters.num_averages  # The number of averages
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

freqs = np.arange(
    node.parameters.freq_offset_start_hz,
    node.parameters.freq_offset_stop_hz,
    node.parameters.freq_offset_step_hz,
)

with program() as pi_12:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(bool) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    freq_offset = declare(int)
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(a, amps)):
                with for_(*from_array(freq_offset, freqs)):
                    # Initialize the qubits
                    qubit.reset(reset_type)
                    align()
                    if not node.parameters.transitions_from_zero_state:
                        qubit.xy.play('x180')
                    qubit.xy.update_frequency(
                        freq_offset + qubit.xy.intermediate_frequency)
                    qubit.xy.play('x180_12', amplitude_scale=a)
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    if not node.parameters.transitions_from_zero_state:
                        qubit.xy.play('x180')
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], I[i] >
                           qubit.resonator.operations["readout"].threshold)
                    save(state[i], state_stream[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].boolean_to_int().buffer(len(freqs)).buffer(len(amps)).average().save(
                f"state{i + 1}"
            )


# %% {Simulate_or_execute}

config = generate_and_fix_config(quam)

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(
        duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, pi_12, simulation_config)
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
        job = qm.execute(pi_12)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {
                                 "amp": amps, "freq_offset": freqs}, dims_tuple=('amp', 'freq_offset'))
    # Add the qubit pulse absolute amplitude to the dataset
    ds = ds.assign_coords(
        {
            "abs_amp": (
                ["qubit", "amp"],
                np.array(
                    [q.xy.operations['x180_12'].amplitude * amps for q in qubits]),
            ),

            "abs_freq": (
                ["qubit", "freq_offset"],
                np.array([q.xy.intermediate_frequency + freqs for q in qubits]),
            )
        }
    )
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # %% {Plotting}
    ds.state.plot(col='qubit', x='freq_offset', y='amp')
    # grid_names = [q.grid_location for q in qubits]
    # grid = QubitGrid(ds, grid_names)
    # for ax, qubit in grid_iter(grid):
    #     ds.sel(qubit=qubit).plot(hue='amp', ax=ax)
    # grid.fig.suptitle("Rabi : I vs. amplitude")
    # plt.tight_layout()
    # plt.show()
    node.results["figure"] = plt.gcf()

    # %% {Update_state}

    # %% {Save_results}
    # node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    # node.machine = quam
    node.save()
    # quam.save()
# %%
