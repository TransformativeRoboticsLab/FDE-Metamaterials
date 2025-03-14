import numpy as np
from nlopt import ForcedStop


def stop_on_nan(x):
    if np.isnan(x).any():
        print("NaN value detected in objective function. Terminating optimization run.")
        raise ForcedStop
