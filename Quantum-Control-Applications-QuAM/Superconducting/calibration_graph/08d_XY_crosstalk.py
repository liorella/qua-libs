"""
here we measure all other qubits to get a sense of XY crosstalk
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation
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

    qubits: Optional[List[str]] = ['q0', 'q1', 'q2']
    num_averages: int = 5000
    max_number_rabi_pulses_per_sweep: int = 50
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(
    name="08d_XY_crosstalk", parameters=Parameters())


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


# %% {QUA_program}
config = quam.generate_config()
config['controllers']['con1']['fems'][1]['analog_inputs'][1]['gain_db'] = 30
for q in quam.qubits:
    config['elements'][q+'.xy']['thread'] = q
    config['elements'][q+'.resonator']['thread'] = q
n_avg = node.parameters.num_averages  # The number of averages
# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_rabi_pulses_per_sweep
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation = 'x180'
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)

N_pi_vec = np.arange(1, N_pi, 2).astype("int")

with program() as xy_crosstalk:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(bool) for _ in range(num_qubits)]
    state_stream = [[declare_stream() for _ in range(num_qubits)]
                    for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                # Initialize the qubits
                if not node.parameters.simulate:
                    for qubit_meas in qubits:
                        if reset_type == "active":
                            active_reset(qubit_meas, "readout",
                                        readout_pulse_name='readout')
                        else:
                            wait(qubit_meas.thermalization_time * u.ns)

                align()
                # Loop for error amplification (perform many qubit pulses)
                with for_(count, 0, count < npi, count + 1):
                    qubit.xy.play(operation)
                align()
                for j, qubit_meas in enumerate(qubits):
                    align()
                    qubit_meas.resonator.measure(
                        "readout", qua_vars=(I[j], Q[j]))
                    assign(
                        state[j], I[j] > qubit_meas.resonator.operations["readout"].threshold)
                    save(state[j], state_stream[i][j])
        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            for j, meas_qubit in enumerate(qubits):
                state_stream[i][j].boolean_to_int().buffer(
                    np.ceil(N_pi / 2)).average().save(f"dm_{qubit.name[1:]}{meas_qubit.name[1:]}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(
        duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, xy_crosstalk, simulation_config)
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
        job = qm.execute(xy_crosstalk)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %%


for k, v in job.result_handles.items():
    if 'dm' in k:
        plt.plot(N_pi_vec, v.fetch_all(), label=k)
plt.legend()
plt.xlabel('Number of 2pi rotations on drive')
plt.ylabel('population of others')
plt.title('XY crosstalk raw data')
plt.ylim([0, 1])
# %%
res_array = []
for qd in qubits:
    res_array.append([])
    for qn in qubits:
        res_array[-1].append(job.result_handles[f'dm_{qd.name[1:]}{qn.name[1:]}'].fetch_all())

res_array = np.array(res_array)

