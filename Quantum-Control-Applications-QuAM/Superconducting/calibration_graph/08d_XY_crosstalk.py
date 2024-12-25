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
    num_averages: int = 5000
    max_number_rabi_pulses_per_sweep: int = 200
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    parallel: bool = False


node = QualibrationNode(
    name="08d_XY_crosstalk", parameters=Parameters(
        qubits=['q0', 'q1', 'q2']
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
N_pi = node.parameters.max_number_rabi_pulses_per_sweep
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation = 'x180'
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)

N_pi_vec = np.arange(1, N_pi, 12).astype("int")

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
                if node.parameters.parallel:
                    # Initialize the qubits

                    align()
                    for j, qubit_meas in enumerate(qubits):
                        qubit_meas.reset(reset_type)
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
                else:
                    for j, qubit_meas in enumerate(qubits):
                        align()
                        qubit_meas.reset(reset_type)
                        align()
                        with for_(count, 0, count < npi, count + 1):
                            qubit.xy.play(operation)
                        # we want to keep duration between reset and meas fixed
                        with for_(count, npi, count < N_pi, count + 1):
                            qubit.xy.play(operation, amplitude_scale=0)
                        align()
                        qubit_meas.resonator.measure(
                            "readout", qua_vars=(I[j], Q[j]))
                        assign(
                            state[j], I[j] > qubit_meas.resonator.operations["readout"].threshold)
                        save(state[j], state_stream[i][j])

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            for j, meas_qubit in enumerate(qubits):
                state_stream[i][j].boolean_to_int().buffer(len(N_pi_vec)).average().save(
                    f"dm_{qubit.name[1:]}{meas_qubit.name[1:]}")


# %% {Simulate_or_execute}
config = generate_and_fix_config(quam)

for _ in range(200):
    node = QualibrationNode(
        name="08d_XY_crosstalk", parameters=Parameters(
            qubits=['q0', 'q1', 'q2']
        ))
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

    node.results['res'] = res
    plt.figure()
    for k in res:
        plt.plot(N_pi_vec, res[k], '.-', label=k)
    plt.legend()
    plt.xlabel('Number of 2pi rotations on drive')
    plt.ylabel('population of others')
    plt.title('XY crosstalk raw data')
    plt.ylim([0, 1])
    node.results['fig_all'] = plt.gcf()
    plt.figure()
    for k in res:
        if k[-1] != k[-2]:
            plt.plot(N_pi_vec, res[k], '.-', label=f'q{k[-2]}->q{k[-1]}')
    plt.legend(title='drive -> meas', loc='best')
    plt.xlabel('Number of 2pi rotations on drive')
    plt.ylabel('population')
    plt.title('XY crosstalk raw data')
    node.results['fig_adverse'] = plt.gcf()

    node.save()
# %%
res_array = []
for qd in qubits:
    res_array.append([])
    for qn in qubits:
        res_array[-1].append(
            job.result_handles[f'dm_{qd.name[1:]}{qn.name[1:]}'].fetch_all())

res_array = np.array(res_array)


# %%
date = '2024-12-21'
data_storage_path = '/usr/local/google/home/ellior/.qualibrate/user_storage/LSP04_v01_GLACIER/' + date
# go over all folders in this path, and for each folder load the npz file and put it in a dictionary

timestamp_format = "%Y-%m-%d-%H%M%S"

data = {}
timestamps = {}
for folder in os.listdir(data_storage_path):
    folder_path = os.path.join(data_storage_path, folder)
    if os.path.isdir(folder_path):
        folder_spec = folder.split('_')
        index = int(folder_spec[0][1:])

        timestamps[index] = datetime.strptime(
            date + '-' + folder_spec[-1], timestamp_format)
        data[index] = np.load(os.path.join(folder_path, 'arrays.npz'))

import xarray as xr

dat_ds = {}
for index in data:
    if index > 263:
        dat_ds[index] = xr.Dataset(data_vars={k: ('num_pi', v) for k, v in data[index].items()}, coords={'num_pi': ('num_pi', N_pi_vec), 'timestamp': timestamps[index]})

ds = xr.concat([dat_ds[index] for index in dat_ds], dim='timestamp').sortby('timestamp')
xr.Dataset.to_dataarray()
# %%
from quam_libs.lib.fit import fit_oscillation, oscillation
da_exp = ds.to_dataarray(dim='experiment')
osc = fit_oscillation(da_exp, dim='num_pi')

# %%
osc.sel(fit_vals='f', experiment=['res.dm_00', 'res.dm_11', 'res.dm_22']).plot(hue='experiment')
plt.xlabel('time [UTC]')
plt.ylabel('Pi rotation error')
plt.title('Rotation errors over time')
plt.legend(['q0->q0', 'q1->q1', 'q2->q2'], title='drive->meas')

# %%
ds.to_dataarray(dim='experiment').sel(num_pi=1+12*5).plot(hue='experiment')

# %%
pi_coeff = 5
da_exp.sel(num_pi=1+12*pi_coeff).\
    sel(experiment=list(set(da_exp.experiment.values).difference(['res.dm_00', 'res.dm_11', 'res.dm_22']))).\
        plot(hue='experiment')
# %%
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(4, 8))
da_exp.sel(num_pi=1+12*pi_coeff).\
    sel(experiment=['res.dm_01', 'res.dm_02']).\
        plot(hue='experiment', ax=ax[0])
da_exp.sel(num_pi=1+12*pi_coeff).\
    sel(experiment=['res.dm_10', 'res.dm_12']).\
        plot(hue='experiment', ax=ax[1])
da_exp.sel(num_pi=1+12*pi_coeff).\
    sel(experiment=['res.dm_20', 'res.dm_21']).\
        plot(hue='experiment', ax=ax[2])

fig.tight_layout()
# %%
