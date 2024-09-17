# %%
from pathlib import Path

from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec, lf_fem_spec, opx_spec, octave_spec
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.machine import build_quam_wiring

# Define static parameters
host_ip = "172.16.33.101"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_opx_plus(controllers = [1, 2])
instruments.add_octave(indices = 1)
# instruments.add_mw_fem(controller=1, slots=[1, 2])
# instruments.add_lf_fem(controller=1, slots=[3, 4])

# Define any custom/hardcoded channel addresses
q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)
q1_drive_ch = octave_spec(index=1, rf_out=2, rf_in=None)
q1_flux_con = opx_spec(con=2, in_port=None, out_port=None)

# Define which quantum elements are present in the system
qubits = [1, 2]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()

# Single feed-line for reading the resonators & individual qubit drive lines
# connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
# connectivity.add_qubit_flux_lines(qubits=qubits)
# connectivity.add_qubit_drive_lines(qubits=qubits)
# allocate_wiring(connectivity, instruments)

# Single feed-line for reading the resonators & driving the qubits + flux on specific fem slot
connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
connectivity.add_qubit_flux_lines(qubits=qubits, constraints=q1_flux_con)
for qubit in qubits:
    connectivity.add_qubit_drive_lines(qubits=qubit, constraints=q1_drive_ch)
    allocate_wiring(connectivity, instruments, block_used_channels=False)

connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2)])

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

# %%
