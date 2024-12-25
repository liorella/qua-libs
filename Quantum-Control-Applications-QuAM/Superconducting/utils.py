from quam_libs.components import QuAM

def generate_and_fix_config(quam: QuAM, use_threads=True) -> dict:
    config = quam.generate_config()
    config['controllers']['con1']['fems'][2]['analog_inputs'][1]['gain_db'] = 30
    if use_threads:
        for q in quam.qubits:
            config['elements'][q+'.xy']['thread'] = q
            config['elements'][q+'.resonator']['thread'] = q

    upcon21 = quam.ports.mw_outputs.con1[2][2].upconverters[1]
    upcon22 = quam.ports.mw_outputs.con1[2][2].upconverters[2]
    upcon41 = quam.ports.mw_outputs.con1[2][4].upconverters[1]
    upcon42 = quam.ports.mw_outputs.con1[2][4].upconverters[2]

    config['controllers']['con1']['fems'][2]['analog_outputs'][2]['upconverters'] = {1: {'frequency': upcon21
    },
                                                                                    2: {'frequency': upcon22}
                                                                                    }

    config['controllers']['con1']['fems'][2]['analog_outputs'][4]['upconverters'] = {1: {'frequency': upcon41},
                                                                                    2: {'frequency': upcon42},
                                                                                    }

    return config

import pandas as pd


def print_qubit_params(quam, params: list[list[str]]):
    cols = ['qubit']
    for param_path in params:
        cols.append('/'.join(param_path))
    rows = []
    for qn in quam.qubits:
        rows.append([])
        rows[-1].append(qn)
        for param_path in params:
            q = quam.qubits[qn]
            for p in param_path:
                q = getattr(q, p)
            rows[-1].append(q)
    return pd.DataFrame(rows, columns=cols)
            