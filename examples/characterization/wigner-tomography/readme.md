---
id: index
title: Wigner tomography
sidebar_label: Wigner tomography
slug: ./
---

Wigner tomography is the process of extracting the Wigner function $W(\alpha)$ 
(a quasi-probabilistic distribution) of the quantum state of a cavity.
The Wigner function is related to the density matrix of a state, therefore the
tomography procedure will allow for the construction of the density matrix and a full 
reconstruction of the quantum state. Through a standard procedure one can encode the state of a qubit
in a superposition of coherent states in a cavity. Therefore, using Wigner
tomography it'll be possible to extract the full density matrix of the qubit.

The Wigner function is defined as follows $$W(\alpha) = 2/\pi \langle P\rangle_\alpha$$,
where $P$ is the photon parity operator and $\alpha$ is the same parameter in the coherent states
and represents a complex vector in the IQ plane.

Using a qubit coupled to the cavity it's straightforward to extract the photon parity of the cavity form
a repeated measurement of the qubit through an additional readout resonator. 
The parity is related to the qubit state as such: $\langle P\rangle \propto P_e - P_g$m where $P_e$ and $P_g$ are the
excited and ground state probabilities of the qubit, which can be extracted with repeated measurement.

Notice: The example describes the tomography process assuming the cavity was encoded prior.

## Config

The configuration consists of  4 quantum elements:
* `cavity_I` and `cavity_Q` define single input elements and are the I and Q components of the cavity
 that we'll perform the tomography on.
* `qubit` is the qubit that's coupled to the cavity
* `rr` is the readout resonator that's coupled to the qubit and used to read its state

Each element has its own IF and LO frequencies, and connection ports. Next, for each element we define the relevant
operation and pulse:
* For `cavity_I` and `cavity_Q` we define the `displace_pulse`, which will be the real and imaginary parts of the displace 
pulse. These were separated due to a needed 2d parameter sweep over the amplitudes of the pulses for the tomography.
* For the `qubit` we define the `x_pi/2_pulse` which is simply a $\pi/2$ rotation around the x axis
* For the `rr` we define the `readout_pulse` - the pulse used for measuring the resonator.

The wavefroms used for the `displace_pulse` and `x_pi/2_pulse` are Gaussians with different parameters.
Generally to displace a cavity one need to apply a pulse such that it integrates to the desired $\alpha$.


## Program

We first calculate the revival time of the qubit coupled to the cavity. Then, we decide of the $\alpha$ range 
we want to sample for constructing the Wigner function, and the spacing. Once we defined the require parameters
we proceed to the Qua program.

We first define the Qua fixed variable for the amplitude scaling required to shift the cavity by the desired $\alpha$
We than create 2 Qua `for_` loops to iterate over the points of the IQ grid. The inner most `for_` loops is for repeated
measurement of the same point in the IQ plane.

Then, in each cycle we perform the tomography procedure:
* We align both the cavity components so they play simultaneously. We displace the I and Q components by the real and 
imaginary parts of $\alpha$, respectively, this is done using realtime amplitude modulation, by multiplying the pulse
with the function `amp(x)`, where `x` is the scaling parameter.
* Next, we align the cavity with the qubit to ensure the pulses to the qubit wait for the cavity to get to the desired state.
On the qubit we apply and `x_pi/2` operation to bring it to the equator. We wait for the revival time, and then apply
a second `x_pi/2` operation to project the qubit to the excited or ground state.
* Finally, we measure the state using the readout resonator and demodulated the reflected signals to get the
qubits state on the IQ plane which can then determine its state.

## Post processing

Having the I,Q results of repeated measurment of the qubit for different $\alpha$ we can extract the parity of the cavity
at each point by counting the excited and ground state measurements. We can display the results using the a heatmap
which represents the IQ plane, with the axes being the real and imaginary parts of $\alpha$.    



## Sample output


## Script
[download script](wigner_tomography.py)