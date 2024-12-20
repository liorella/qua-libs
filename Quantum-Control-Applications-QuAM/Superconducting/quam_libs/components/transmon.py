from quam.core import quam_dataclass
from quam.components.channels import IQChannel, Pulse
from quam import QuamComponent
from .flux_line import FluxLine
from .readout_resonator import ReadoutResonator
from qualang_tools.octave_tools import octave_calibration_tool
from qm import QuantumMachine, logger
from typing import Dict, Any, Literal, Optional, Union, List, Tuple
from qm.qua import align, wait, declare, fixed, assign, while_, save, StreamType
import numpy as np
from dataclasses import field

__all__ = ["Transmon"]


@quam_dataclass
class Transmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        xy (IQChannel): The xy drive component.
        z (FluxLine): The z drive component.
        resonator (ReadoutResonator): The readout resonator component.
        T1 (float): The transmon T1 in s.
        T2ramsey (float): The transmon T2* in s.
        T2echo (float): The transmon T2 in s.
        thermalization_time_factor (int): thermalization time in units of T1.
        anharmonicity (int, float): the transmon anharmonicity in Hz.
        freq_vs_flux_01_quad_term (float):
        arbitrary_intermediate_frequency (float):
        sigma_time_factor:
        phi0_current (float):
        phi0_voltage (float):
        GEF_frequency_shift (int):
        chi (float):
        grid_location (str): qubit location in the plot grid as "(column, row)"
    """

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None
    resonator: ReadoutResonator = None

    f_01: float = None
    f_12: float = None
    anharmonicity: int = 150e6
    freq_vs_flux_01_quad_term: float = 0.0
    arbitrary_intermediate_frequency: float = 0.0

    T1: float = 10e-6
    T2ramsey: float = None
    T2echo: float = None
    thermalization_time_factor: int = 5
    sigma_time_factor: int = 5
    phi0_current: float = 0.0
    phi0_voltage: float = 0.0

    GEF_frequency_shift: int = 10
    chi: float = 0.0
    grid_location: str = None
    extras: Dict[str, Any] = field(default_factory=dict)

    active_reset_available: bool = False

    def get_output_power(self, operation, Z=50) -> float:
        power = self.xy.opx_output.full_scale_power_dbm
        amplitude = self.xy.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

    @property
    def inferred_f_12(self) -> float:
        """The 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity"""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.f_01=} is not a number")
        if not isinstance(self.anharmonicity, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.anharmonicity=} is not a number")
        return self.f_01 + self.anharmonicity

    @property
    def inferred_anharmonicity(self) -> float:
        """The transmon anharmonicity in Hz, derived from f_01 and f_12."""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_01=} is not a number")
        if not isinstance(self.f_12, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_12=} is not a number")
        return self.f_12 - self.f_01

    @property
    def sigma(self, operation: Pulse):
        return operation.length / self.sigma_time_factor

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns."""
        return int(self.thermalization_time_factor * self.T1 * 1e9 / 4) * 4

    def calibrate_octave(
        self, QM: QuantumMachine, calibrate_drive: bool = True, calibrate_resonator: bool = True
    ) -> None:
        """Calibrate the Octave channels (xy and resonator) linked to this transmon for the LO frequency, intermediate
        frequency and Octave gain as defined in the state.

        Args:
            QM (QuantumMachine): the running quantum machine.
            calibrate_drive (bool): flag to calibrate xy line.
            calibrate_resonator (bool): flag to calibrate the resonator line.
        """
        if calibrate_resonator and self.resonator is not None:
            logger.info(f"Calibrating {self.resonator.name}")
            octave_calibration_tool(
                QM,
                self.resonator.name,
                lo_frequencies=self.resonator.frequency_converter_up.LO_frequency,
                intermediate_frequencies=self.resonator.intermediate_frequency,
            )

        if calibrate_drive and self.xy is not None:
            logger.info(f"Calibrating {self.xy.name}")
            octave_calibration_tool(
                QM,
                self.xy.name,
                lo_frequencies=self.xy.frequency_converter_up.LO_frequency,
                intermediate_frequencies=self.xy.intermediate_frequency,
            )

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            self.xy.operations[gate] = f"#./{gate}_{gate_shape}"

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def __matmul__(self, other):
        if not isinstance(other, Transmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, " f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError(
                "Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}")

        for qubit_pair in self._root.qubit_pairs:
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError(
                "Qubit pair not found: qubit_control={self.name}, " "qubit_target={other.name}")

    def align(self):
        align(self.xy.name, self.resonator.name)

    def wait(self, duration):
        wait(duration, self.xy.name, self.resonator.name)

    def reset(self, override_default: Optional[Literal['thermal', 'active']] = None):
        reset_type = self.default_reset if override_default is None else override_default
        if not self.active_reset_available:
            reset_type = 'thermal'
        if reset_type == "active":
            self.active_reset("readout", readout_pulse_name='readout')
        elif reset_type == "thermal":
            self.wait(self.thermalization_time // 4)
        else:
            raise ValueError(f"Unrecognized reset type {reset_type}.")
        self.align()

    
    def active_reset(
        self,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout_QND",
        max_attempts: int = 15,
    ):
        pulse = self.resonator.operations[readout_pulse_name]

        I = declare(fixed)
        Q = declare(fixed)
        state = declare(bool)
        attempts = declare(int, value=1)
        pi_threshold = pulse.threshold#self.resonator.rus_pi_threshold if self.resonator.rus_pi_threshold is not None else pulse.threshold
        assign(attempts, 1)
        self.align()
        self.resonator.measure(readout_pulse_name, qua_vars=(I, Q))
        assign(state, I > pi_threshold)
        wait(self.resonator.depletion_time // 2, self.resonator.name)
        self.xy.play(pi_pulse_name, condition=state)
        self.align()
        with while_((I > pulse.rus_exit_threshold) & (attempts < max_attempts)):
            self.align()
            self.resonator.measure(readout_pulse_name, qua_vars=(I, Q))
            assign(state, I > pi_threshold)
            wait(self.resonator.depletion_time // 2, self.resonator.name)
            self.xy.play(pi_pulse_name, condition=state)
            self.align()
            assign(attempts, attempts + 1)
        wait(500, self.xy.name)
        self.align()
        if save_qua_var is not None:
            save(attempts, save_qua_var)