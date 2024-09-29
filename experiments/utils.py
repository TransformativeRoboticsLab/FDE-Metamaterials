import random

from metatop import V_DICT

PARAMETER_SPECS = {
    'E_max': {
        'type': 'float',
        'range': (1., 1.),
    },
    'E_min': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(10)),
    },
    'nu': {
        'type': 'float',
        'range': (0.1, 0.499),
    },
    'start_beta': {
        'type': 'int',
        'range': (1, 8),
    },
    'n_betas': {
        'type': 'int',
        'range': (1, 8),
    },
    'n_epochs': {
        'type': 'int',
        'range': (1, 8),
    },
    'epoch_duration':{
        'type': 'int',
        'range': (1, 10_000),
    },
    'extremal_mode': {
        'type': 'int',
        'values': (1, 2),
    },
    'basis_v': {
        'type': 'cat',
        'values': ['BULK', 'VERT', 'HSA', 'SHEAR']
    },
    'objective_type': {
        'type': 'cat',
        'values': ('ray', 'ray_sq')
    },
    'nelx': {
        'type': 'int',
        'range': (1, 1_000),
    },
    'nely': {
        'type': 'int',
        'range': (1, 1_000),
    },
    'norm_filter_radius': {
        'type': 'float',
        'range': (0.01, 0.5),
    },
    'g_vec_eps': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(6)),
    },
    'weight_scaling_factor': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(4)),
    },
}

def get_random_value(param):
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        raise ValueError(f"Parameter '{param}' not found in PARAMETER_SPECS")

    s_type = spec.get('type', None)
    if s_type== 'cat':
        return random.choice(spec['values'])
    elif s_type  == 'float':
        if 'range' in spec:
            return random.uniform(*spec['range'])
        elif 'values' in spec:
            return random.choice(spec['values'])
    elif s_type == 'int':
        if 'range' in spec:
            return random.randint(*spec['range'])
        elif 'values' in spec:
            return random.choice(spec['values'])
        
def validate_param_value(param, value, verbose=True):
    print(f"Validating parameter '{param}' with value: {value}")
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        print(f"Parameter '{param}' not found in PARAMETER_SPECS")
        return False

    s_type = spec.get('type', None)
    if s_type == 'cat':
        print(value, spec['values'])
        print(value in spec['values'])
        if value not in spec['values']:
            raise ValueError(f"Invalid categorical value for parameter {param}: {value}. Values must be one of {spec['values']}")
    elif s_type == 'float':
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}. Values must be one of {spec['values']}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")
    elif s_type == 'int':
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")
    
    if verbose:
        print(f"Parameter '{param}' validated successfully with value: {value}")

if __name__ == "__main__":
    for _ in range(10_000):
        for param in PARAMETER_SPECS.keys():
            v = get_random_value(param)
            validate_param_value(param, v, verbose=False)