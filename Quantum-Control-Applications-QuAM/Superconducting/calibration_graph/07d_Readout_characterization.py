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
import pyle.dataking.datataker_v2.datataker_base as datataker_base
import pyle.datavault.public as pdv
import pyle.datavault.client as dv_client
import getpass
from pyle.dataking.datataker_v2.ac_stark_spectroscopy import CombinedACStarkChiAnalysis
from utils import generate_and_fix_config
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter, fetching_tool
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.macros import qua_declaration
from quam_libs.components import QuAM
from qualibrate import QualibrationNode, NodeParameters
import pyle.samples.keys as keys
from labrad.units import Hz, dBm, MHz
%cd / usr/local/google/home/ellior/qua-libs/Quantum-Control-Applications-QuAM/Superconducting


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    qubit_operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.2
    operation_len_in_ns: Optional[int] = None
    q_frequency_span_in_mhz: float = 100
    q_frequency_step_in_mhz: float = 0.5
    r_frequency_span_in_mhz: float = 30
    r_frequency_step_in_mhz: float = 0.5
    target_peak_width: Optional[float] = 2e6
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(name="07d_Readout_characterization",
                        parameters=Parameters(
                            qubits=['q0', 'q1', 'q2', 'q4']
                        ))


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
quam = QuAM.load('/usr/local/google/home/ellior/quam_state/lsp04_v01_glacier')
# Generate the OPX and Octave configurations


# Open Communication with the QOP
# need to wrap this with a hack function that loads then sets upconverters to None
qmm = quam.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = quam.active_qubits
else:
    qubits = [quam.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

dvw = dv_client.default_datavault().cd(
    ["", "Users", getpass.getuser(), "onboarding", "test_datavault"]
)
# %% {QUA_program}
for q in quam.qubits.values():
    q.xy.operations.saturation.amplitude = 0.25
config = generate_and_fix_config(quam, use_threads=False)

operation = node.parameters.qubit_operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
q_span = node.parameters.q_frequency_span_in_mhz * u.MHz
q_step = node.parameters.q_frequency_step_in_mhz * u.MHz
q_dfs = np.arange(-q_span // 2, +0.5 * q_span // 2, q_step, dtype=np.int32)
r_span = node.parameters.r_frequency_span_in_mhz * u.MHz
r_step = node.parameters.r_frequency_step_in_mhz * u.MHz
r_dfs = np.arange(-r_span // 2, +r_span // 2, r_step, dtype=np.int32)
qubit_freqs = {q.name: q.f_01 for q in qubits}  # for opx

target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        # the desired width of the response to the saturation pulse (including saturation amp), in Hz
        3e6
    )

state_readout = True
with program() as readout_characterization:

    if state_readout:
        I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
        state = [declare(bool) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]
    else:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    
    q_df = declare(int)  # QUA variable for the qubit frequency
    r_df = declare(int)  # QUA variable for the qubit frequency
    prep_state = declare(fixed)
    amp_ = declare(fixed)

    for i, qubit in enumerate(qubits):
        q_if = config['elements'][qubit.xy.name]['intermediate_frequency']
        res_if = config['elements'][qubit.resonator.name]['intermediate_frequency']
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(prep_state, [0.0, 1.0]):
                with for_(*from_array(r_df, r_dfs)):
                    with for_(*from_array(q_df, q_dfs)):
                        qubit.reset()
                        qubit.xy.play('x180', amplitude_scale=prep_state)

                        qubit.align()

                        # Play the saturation pulse
                        qubit.resonator.update_frequency(r_df + res_if)
                        qubit.xy.update_frequency(q_df + q_if)
                        qubit.align()
                        play('readout' * amp(1), qubit.resonator.name,
                             duration=145 + 60*8//4)

                        qubit.xy.wait(125)
                        play(
                            'x90' * amp(1/2), qubit.xy.name,
                            duration=60*2 // 4,
                        )
                        qubit.xy.update_frequency(q_if)
                        qubit.resonator.update_frequency(res_if)
                        qubit.align()

                        # readout the resonator
                        # we should wait for it to deplete
                        qubit.resonator.wait(250)
                        # qubit.xy.play('x180', amplitude_scale=state)  # flip to get consistent results
                        if state_readout:
                            quam.measure(qubit, state[i])
                            save(state[i], state_stream[i])
                        else:
                            qubit.resonator.measure(
                            "readout", qua_vars=(I[i], Q[i]))

                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        # Wait for the qubit to decay to the ground state
                        qubit.resonator.wait(quam.depletion_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if state_readout:
                state_stream[i].boolean_to_int().buffer(len(q_dfs)).buffer(
                    len(r_dfs)).buffer(2).average().save(f"state{i + 1}")

            else:
                I_st[i].buffer(len(q_dfs)).buffer(
                    len(r_dfs)).buffer(2).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(q_dfs)).buffer(
                    len(r_dfs)).buffer(2).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(
        duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, readout_characterization, simulation_config)
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
        job = qm.execute(readout_characterization)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {'prep_state': [
                                 0, 1], "r_freq": r_dfs, "q_freq": q_dfs}, ('prep_state', 'r_freq', 'q_freq'))

    if not state_readout:
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
        ds['IQ'] = ds.I * 1j * ds.Q
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "q_freq_full": (
                ["qubit", "q_freq"],
                np.array([q_dfs + qubit_freqs[q.name] for q in qubits]),
            ),
            "r_freq_full": (
                ["qubit", "r_freq"],
                np.array([r_dfs + q.resonator.RF_frequency for q in qubits]),
            ),
        }
    )
    # Add the dataset to the node
    node.results = {"ds": ds}
    # ds.state.loc[dict(prep_state=1)] = 1 - ds.state.loc[dict(prep_state=1)]
    # %% {Data_analysis}
    ds.state.plot(col='qubit', row='prep_state', x='r_freq',
              y='q_freq', add_colorbar=False)

    plt.gcf().tight_layout()


# %%

ds['one'] = ds.state
ds['zero'] = 1- ds.state

data = {q: {s: ds[['zero', 'one']].
        rename({'r_freq_full': 'readoutFrequency'}).
        sel(qubit=q, prep_state=s).stack(z=('r_freq', 'q_freq')).
        to_dataframe()[['readoutFrequency', 'q_freq', 'zero', 'one']].
        to_numpy()
        for s in [0, 1]}
        for q in ds.qubit.values}



ro_attenuation = -20
ro_power = quam.qubits['q0'].resonator.get_output_power('readout') + ro_attenuation
ro_power = ro_power * dBm
# ro_power = quam.qubits['q0'].resonator.opx_output.full_scale_power_dbm + \
#       u.volts2dBm( quam.qubits['q0'].resonator.operations['readout'].amplitude) + \
#       ro_attenuation


results = {}
analyses = {}
ro_attenuation = -20
for q in ds.qubit.values:
    ro_power = quam.qubits[q].resonator.get_output_power('readout') + ro_attenuation
    ro_power = ro_power * dBm
    dataset = [dvw.create_dataset_oneshot_data(
        data=data[q][s],
        name="ac_stark_spec_test",
        # describes an independent (or indep) axis, with the label and time
        independents=[pdv.IndependentAxis('readoutFrequency', 'Hz'),
                    pdv.IndependentAxis('q_freq_full', 'Hz')],
        # describes the dependent axes, with the label, legend, and unit (empty in this case)
        dependents=[
            pdv.DependentAxis("|0>", "|0>", ""),
            pdv.DependentAxis("|1>", "|1>", ""),
        ],
        params={"state": s,
                keys.FREQUENCY_1_0: quam.qubits[q].f_01 * Hz,
                keys.READOUT_POWER: ro_power,
                'anharmonicity': -200 * MHz,
                keys.BARE_RESONATOR_FREQUENCY: quam.qubits[q].resonator.RF_frequency * Hz},
    ) for s in [0, 1]]

    result = datataker_base.DataTakerResult(dataset)
    analysis = CombinedACStarkChiAnalysis.from_data(result)
    results[q] = result
    analyses[q] = analysis


# %%
fig, ax = plt.subplots(1, len(ds.qubit), figsize=[12, 4], sharey=True)
for i, q in enumerate(ds.qubit.values):
    analyses[q].plot(ax=ax[i])
fig.tight_layout()

    # %%
dvw['00054 - ac_stark_spec_test'].quick_plot()

from quam_libs.components.readout_resonator import ReadoutResonator
ReadoutResonator.measure