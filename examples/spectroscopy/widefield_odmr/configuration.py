import numpy as np

pulse_len = 100
readout_len = 400
qubit_IF = 50e6
rr_IF = 50e6
qubit_LO = 6.345e9
qb_LO = 4.755e9


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


gauss_pulse = gauss(0.2, 0, 15, pulse_len)

config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # I
                2: {'offset': +0.0},  # Q
                3: {'offset': +0.0},  # Fluorescence AOM
            },
            'digital_outputs': {
                1: {},
            },
            'analog_inputs': {
                1: {'offset': +0.0},
            }
        }
    },

    'elements': {

        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                'lo_frequency': qb_LO,
            },
            'outputs': {
                'output1': ('con1', 1)
            },
            'intermediate_frequency': 80e6,
            'operations': {
                'SAT': "SAT_PULSE",
            },
            'time_of_flight': 180,
            'smearing': 0
        },
        "readout_el": {
            "singleInput": {
                "port": ("con1", 3)
            },
            'intermediate_frequency': 80e6,
            'operations': {
                'readout': "readoutPulse",
            },
            'digitalInputs': {
                'digital_input1': {
                    'port': ("con1", 1),
                    'delay': 0,
                    'buffer': 0,
                },
            }
        },
    },

    "pulses": {
        "constPulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'gauss_wf'
            }
        },
        "readoutPulse": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'single': 'const_wf'
            },
            'digital_marker': 'ON'
        },
        "SAT_PULSE": {
            'operation': 'control',
            'length': pulse_len,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            }
        },
        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
            },
            'digital_marker': 'ON'
        },

    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },
    },

    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss_pulse
        },
        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.2, 0, 12, pulse_len)
        },
        '-pi/2_wf': {
            'type': 'arbitrary',
            'samples': gauss(-0.1, 0, 12, pulse_len)
        },
        'pi/2_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.1, 0, 12, pulse_len)
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0
        },
        'readout_wf': {
            'type': 'constant',
            'sample': 0.3
        }
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        },
    },
    'integration_weights': {

        'integW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
        },

    },
    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(0.0, 0.0)}
        ],
        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': qb_LO,
             'correction': IQ_imbalance(0.0, 0.0)}
        ],
    }
}