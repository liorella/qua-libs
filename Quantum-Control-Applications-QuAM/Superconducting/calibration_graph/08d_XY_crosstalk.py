"""
here we measure all other qubits to get a sense of XY crosstalk
"""


# %% {Imports}
from datetime import datetime
import numpy
import os
from utils import generate_and_fix_config
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
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
    num_averages: int = 1000
    max_number_x_gates_per_sweep: int = 200
    x_gate_step_size: int = 2
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    parallel: bool = False
    other_qubits_on_equator: bool = True  # measure AC stark shift pull
    detune_adverserial_drive: int = int(25e6) # this is needed to avoid ZZ affecting this measzurement


node = QualibrationNode(
    name="08d_XY_crosstalk", parameters=Parameters(
        qubits=['q0', 'q1',
                'q2', 'q3', 'q4'
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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Number of applied Rabi pulses sweep
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation = 'x180'
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)

max_x_gates = node.parameters.max_number_x_gates_per_sweep
x_gates_rep = np.arange(1,
                     max_x_gates,
                     node.parameters.x_gate_step_size).astype("int")

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
            with for_(*from_array(npi, x_gates_rep)):
                if node.parameters.parallel:
                    # Initialize the qubits

                    align()
                    for j, qubit_meas in enumerate(qubits):
                        qubit_meas.reset(reset_type)
                        if node.parameters.other_qubits_on_equator:
                            if i != j:
                                qubit_meas.xy.play('y90')
                    align()
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency + node.parameters.detune_adverserial_drive)
                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        qubit.xy.play(operation)
                    # we want to keep duration between reset and meas fixed
                    with for_(count, npi, count < max_x_gates, count + 1):
                        qubit.xy.play(operation, amplitude_scale=0)
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    align()
                    for j, qubit_meas in enumerate(qubits):
                        align()
                        if node.parameters.other_qubits_on_equator:
                            if i != j:
                                qubit_meas.xy.play('-y90')
                        qubit_meas.resonator.measure(
                            "readout", qua_vars=(I[j], Q[j]))
                        assign(
                            state[j], I[j] > qubit_meas.resonator.operations["readout"].threshold)
                        save(state[j], state_stream[i][j])
                else:
                    for j, qubit_meas in enumerate(qubits):
                        align()
                        qubit_meas.reset(reset_type)
                        if node.parameters.other_qubits_on_equator:
                            if i != j:
                                qubit_meas.xy.play('y90')
                        align()
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + node.parameters.detune_adverserial_drive)
                        with for_(count, 0, count < npi, count + 1):
                            qubit.xy.play(operation)
                        # we want to keep duration between reset and meas fixed
                        with for_(count, npi, count < max_x_gates, count + 1):
                            qubit.xy.play(operation, amplitude_scale=0)
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        align()
                        if node.parameters.other_qubits_on_equator:
                            if i != j:
                                qubit_meas.xy.play('x90')
                        qubit_meas.resonator.measure(
                            "readout", qua_vars=(I[j], Q[j]))
                        assign(
                            state[j], I[j] > qubit_meas.resonator.operations["readout"].threshold)
                        if i == j:  # in this case we are most likely in 1 at end of cycle
                            qubit.xy.play(operation)
                        save(state[j], state_stream[i][j])

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            for j, meas_qubit in enumerate(qubits):
                state_stream[i][j].boolean_to_int().buffer(len(x_gates_rep)).average().save(
                    f"dm_{qubit.name[1:]}{meas_qubit.name[1:]}")


# %% {Simulate_or_execute}
config = generate_and_fix_config(quam)

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

res = {}
for k, v in job.result_handles.items():
    if 'dm' in k:
        res[k] = v.fetch_all()

res_array = np.array([[res[f'dm_{q.id[1:]}{qm.id[1:]}'] for qm in qubits] for q in qubits])
q_index_array = np.array([[[q.id[1:], qm.id[1:]] for qm in qubits] for q in qubits])

import xarray as xr
da = xr.DataArray(res_array,
                  coords=[('qd', [q.id for q in qubits]),
                          ('qm', [qm.id for qm in qubits]),
                          ('x_gates', x_gates_rep)
                          ],
                  dims=['qd', 'qm', 'x_gates']
                  )

node.results['res'] = res
plt.figure()
for k in res:
    plt.plot(x_gates_rep, res[k], '.-', label=k)
plt.legend()
plt.xlabel('Number of 2pi rotations on drive')
plt.ylabel('population of others')
plt.title('XY crosstalk raw data')
plt.ylim([0, 1])
node.results['fig_all'] = plt.gcf()
plt.figure()
for k in res:
    if k[-1] != k[-2]:
        plt.plot(x_gates_rep, res[k], '.-', label=f'q{k[-2]}->q{k[-1]}')
plt.legend(title='drive -> meas', loc='best')
plt.xlabel('Number of 2pi rotations on drive')
plt.ylabel('population')
plt.title('XY crosstalk raw data')
node.results['fig_adverse'] = plt.gcf()

node.save()

# %%
import xrft
da.plot(col='qd', row='qm')
# %%
da_ft = xrft.power_spectrum(da, 'x_gates', 'x_gates', detrend='constant')
# da_ft /= da_ft.max(dim='freq_x_gates')
da_ft.plot(col='qd', row='qm')
# %%
from quam_libs.lib.fit import peaks_dips

peaks_dips(da_ft, 'freq_x_gates', prominence_factor=0.25).position.transpose().plot.imshow(yincrease=False, vmax=0.05)
# %%