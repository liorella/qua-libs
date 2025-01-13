"""
        ACSTARKCHI
This experiment recreates the experiment done in https://arxiv.org/pdf/2402.00413.

Things to note:
1. `operation_amplitde_factor` is 0.8 in the paper and we're keeping it this way here
2. the pi pulse duration seems to matter a lot and it needs to be smaller than 1/kappa for the
experiment to make sense. This is contained in the factor `operation_len_factor`.

"""

# %% {Imports}
import pyle.samples.readout_helpers as readout_helpers
import pandas as pd
import pyle.dataking.datataker_v2.datataker_base as datataker_base
import pyle.datavault.public as pdv
import pyle.datavault.client as dv_client
import getpass
from pyle.dataking.datataker_v2.ac_stark_spectroscopy import CombinedACStarkChiAnalysis
import pyle.samples.keys as keys
from tunits import GHz, Hz, dBm, MHz
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
# %cd / usr/local/google/home/ellior/qua-libs/Quantum-Control-Applications-QuAM/Superconducting


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    qubit_operation: str = "x180"
    operation_amplitude_factor: float = 0.8
    operation_len_factor: float = 1.25
    q_frequency_span_in_mhz: float = 50
    q_frequency_step_in_mhz: float = 0.5
    r_frequency_span_in_mhz: float = 30
    r_frequency_step_in_mhz: float = 0.5
    target_peak_width: Optional[float] = 2e6
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(name="07d_Readout_characterization",
                        parameters=Parameters(
                            qubits=[
                                'q0', 'q1', 'q2', 'q4',
                                # 'q3'
                            ]
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
config = generate_and_fix_config(quam, use_threads=False)

operation = node.parameters.qubit_operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Qubit detuning sweep with respect to their resonance frequencies
q_span = node.parameters.q_frequency_span_in_mhz * u.MHz
q_step = node.parameters.q_frequency_step_in_mhz * u.MHz
q_dfs = np.arange(-q_span // 2, +0.5 * q_span // 2, q_step, dtype=np.int32)
r_span = node.parameters.r_frequency_span_in_mhz * u.MHz
r_step = node.parameters.r_frequency_step_in_mhz * u.MHz
r_dfs = np.arange(-r_span // 2, +r_span // 2, r_step, dtype=np.int32)
qubit_freqs = {q.name: q.f_01 for q in qubits}  # for opx
operation_duration_clk = {q.name:
                          int(node.parameters.operation_len_factor *
                              (q.xy.operations[operation].length // 4))
                          for q in qubits}

rabi_rate_duration_factor = {q.name: (q.xy.operations[operation].length // 4) /
                             operation_duration_clk[q.name]
                             for q in qubits}
print('rabi_rate_duration_factor = ')
print(rabi_rate_duration_factor)
print('operation_duration_clk = ')
print(operation_duration_clk)

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
                        qubit.reset()                         # TEMP REMOVE
                        qubit.xy.play('x180', amplitude_scale=prep_state)

                        qubit.align()

                        # Play the saturation pulse
                        qubit.resonator.update_frequency(r_df + res_if)
                        qubit.xy.update_frequency(q_df + q_if)
                        qubit.align()
                        play('readout' * amp(0.5), qubit.resonator.name,
                             duration=145 + operation_duration_clk[qubit.name])

                        qubit.xy.wait(125)
                        play(
                            operation * amp(rabi_rate_duration_factor[qubit.name]
                                            * node.parameters.operation_amplitude_factor),
                            qubit.xy.name,
                            duration=operation_duration_clk[qubit.name],
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
    job.plot_waveform_report_with_simulated_samples()

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
    # for fig, ax in plt.subplots(2, len(qubits)):
    # for i, q in enumerate(qubits):
    # ds.state.plot(x='r_freq_full', y='q_freq')

    ds.state.plot(col='qubit', row='prep_state', x='r_freq',
                  y='q_freq', add_colorbar=False)

    plt.gcf().tight_layout()
    node.results['fig'] = plt.gcf()
    node.save()

# %%

ds['one'] = ds.state
ds['zero'] = 1 - ds.state

data = {q: {s: ds[['zero', 'one']].
        rename({'r_freq_full': 'readoutFrequency'}).
        sel(qubit=q, prep_state=s).stack(z=('r_freq', 'q_freq')).
        to_dataframe()[['readoutFrequency', 'q_freq', 'zero', 'one']].
        to_numpy()
        for s in [0, 1]}
        for q in ds.qubit.values}


# ro_power = quam.qubits['q0'].resonator.opx_output.full_scale_power_dbm + \
#       u.volts2dBm( quam.qubits['q0'].resonator.operations['readout'].amplitude) + \
#       ro_attenuation

df_summary = pd.DataFrame(columns=['q',
                                   'chi [MHz]',
                                   'kappa [MHz]',
                                   '2chi/kappa',
                                   'on res photon number',
                                   'shift at res [MHz]',
                                   'photon no. on drive',
                                   'effective attenuation [dB]'])

# I inferred this value so that I will get the correct amplitude when using readout_helpers.power2amp
ro_attenuation = - 16.95
datasets = {}
results = {}
analyses = {}
for q in ds.qubit.values:
    fsp = quam.qubits[q].resonator.opx_output.full_scale_power_dbm
    ro_power = quam.qubits[q].resonator.get_output_power(
        'readout') + ro_attenuation + (fsp + 11)
    ro_power = ro_power * dBm
    dataset = [dvw.create_dataset_oneshot_data(
        data=data[q][s],
        name="ac_stark_chi",
        # describes an independent (or indep) axis, with the label and time
        independents=[pdv.IndependentAxis('readoutFrequency', 'Hz'),
                      pdv.IndependentAxis('q_freq_full', 'Hz')],
        # describes the dependent axes, with the label, legend, and unit (empty in this case)
        dependents=[
            pdv.DependentAxis("|0>", "|0>", ""),
            pdv.DependentAxis("|1>", "|1>", ""),
        ],
        params={
            'target': str(q),
            "state": s,
            keys.FREQUENCY_1_0: quam.qubits[q].f_01 * Hz,
            keys.READOUT_POWER: ro_power,
            'anharmonicity': -200 * MHz,
            keys.BARE_RESONATOR_FREQUENCY: quam.qubits[q].resonator.RF_frequency * Hz},
    ) for s in [0, 1]]
    datasets[q] = dataset

    result = datataker_base.DataTakerResult(dataset)
    analysis = CombinedACStarkChiAnalysis.from_data(result)
    results[q] = result
    analyses[q] = analysis
    dac_amp = quam.qubits[q].resonator.operations['readout'].amplitude
    drive_freq_ghz = quam.qubits[q].resonator.RF_frequency * 1e-9
    # print('actual readout amp = ', dac_amp)
    row_to_add = [
        q,
        analyses[q].combined_model.chi['MHz'],
        analyses[q].combined_model.average_linewidth['MHz'],
        2*abs(analyses[q].combined_model.chi_GHz /
              analyses[q].combined_model.average_linewidth_GHz),
        analyses[q].combined_model.compute_average_on_resonance_photon_number(
            dac_amp),
        analyses[q].combined_model.average_shift_GHz_per_dac_amp_squared_on_res * dac_amp**2*1e3,
        analyses[q].combined_model.compute_photon_numbers(
            drive_freq_ghz, dac_amp),
        analyses[q].combined_model.effective_readout_attenuation
    ]
    df_summary.loc[len(df_summary.index)] = row_to_add

df_summary

# %%
fig, ax = plt.subplots(1, len(ds.qubit), figsize=[12, 4], squeeze=False)
for i, q in enumerate(ds.qubit.values):
    drive_freq_ghz = quam.qubits[q].resonator.RF_frequency * 1e-9
    analyses[q].plot(ax=ax[0][i])
    ax[0][i].axvline(drive_freq_ghz, color='r', linestyle='--')
    ax[0][i].set_aspect(1)
    ax[0][i].set_title(q)
fig.suptitle('ACStarkChi model results')
fig.tight_layout()
# %%
for q in ds.qubit.values:
    x = 2*abs(analyses[q].combined_model.chi_GHz /
              analyses[q].combined_model.average_linewidth_GHz)
    print(q,
          analyses[q].combined_model.average_linewidth*(1+x**2),
          np.sqrt(analyses[q].combined_model.average_linewidth*(1+x**2)))
# %%
save_to_quam = False
if save_to_quam:
    for q in ds.qubit.values:
        print(q)
        print(analyses[q].combined_model.average_resonance_freq_GHz -
            quam.qubits[q].resonator.LO_frequency * 1e-9)
        quam.qubits[q].resonator.intermediate_frequency = analyses[q].combined_model.average_resonance_freq_GHz * \
            1e9 - quam.qubits[q].resonator.LO_frequency
        print(quam.qubits[q].resonator.intermediate_frequency * 1e-9)

    quam.save()
# %% calibrating the difference between power2amp and get_output_power for a -11 full_scale_power_dbm on readout
print('expected readout amp = ', readout_helpers.power2amp(quam.qubits['q0'].resonator.get_output_power(
    'readout') * dBm - 16.95 * dBm))
print('actual readout amp = ',
      quam.qubits['q0'].resonator.operations['readout'].amplitude)
# %% prepare plot for presenting

rabi_rates = {q.name: (2/(4*operation_duration_clk[q.name])) * GHz for q in qubits}  # the factor of 2 is because it's a half-cosine waveform
assert len(set(rabi_rates.values())) == 1
col_row_map = {0: 2, 1: 3}
fig, ax = plt.subplots(2, len(qubits), sharey=True)
for i, ax_row in enumerate(ax):
    for j, ax_col in enumerate(ax_row):
        dataset_ghz = datasets[qubits[j].name][i]
        dataset_ghz[:, [0, 1]] = dataset_ghz[:, [0, 1]] * 1e-9
        dataset_ghz.quick_plot(column=col_row_map[i],
                                               ax=ax_col, title=qubits[j].name,
                                               colorbar=False,
                                               label_axes=False)
        ax_col.set_aspect(1)
        if j == 0:
            ax_col.set_ylabel('qubit offset freq [GHz]')
        if i == 1:
            ax_col.set_xlabel('readout freq [GHz]')

fig.suptitle(f'Rabi rate = {list(rabi_rates.values())[0]['MHz']:.1f} MHz, '
             f'pi duration = {list(operation_duration_clk.values())[0] * 4} nsec')
fig.tight_layout()

for q in qubits:
    for i in [0, 1]:
        print(q.name, i, datasets[q.name][i].url())

for q in qubits:
    for i in [0, 1]:
        print(q.name, i, datasets[q.name][i].parameters[keys.READOUT_POWER])
# %%
