import numpy as np

ISQR2 = 1. / np.sqrt(2.)
ISQR3 = 1. / np.sqrt(3.)
V_DICT = {
    "BULK": np.array([[ISQR2, -ISQR2, 0.],
                      [ISQR2,  ISQR2, 0.],
                      [0.,     0.,    1.]]),
    "IBULK": np.array([[-ISQR2, ISQR2, 0.],
                       [ISQR2, ISQR2, 0.],
                       [0.,    0.,    1.]]),
    "VERT": np.array([[0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]]),
    "VERT2": np.array([[0., ISQR2, -ISQR2],
                       [1., 0.,     0.],
                       [0., ISQR2,  ISQR2]]),
    "SHEAR": np.array([[0., -ISQR2, ISQR2],
                       [0.,  ISQR2, ISQR2],
                       [1.,  0.,    0.]]),
    "SHEARXY": np.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]]),
    "HSA": np.array([[0.,     0.,    1.],
                    [ISQR2, -ISQR2, 0.],
                    [ISQR2,  ISQR2, 0.]]),
    "HSA2": np.array([[0.,     0.,    1.],
                     [ISQR2,  ISQR2, 0.],
                      [-ISQR2, ISQR2, 0.]]),
    "IHSA": np.array([[1., 0., 0.],
                      [0., ISQR2, -ISQR2],
                      [0., ISQR2,  ISQR2]]),
    "EYE": np.eye(3),
    "HSA3": np.array([[1, -1/2,  3/2],
                      [1, -1/2, -3/2],
                      [1,  1,    0]]),
}

for k, v in V_DICT.items():
    for column in v.T:
        column /= np.linalg.norm(column)
    assert np.allclose(v.T @ v, np.eye(3)), f'{k} does not have orthonormal columns'