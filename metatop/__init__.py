import numpy as np
from loguru import logger

ISQR2 = 1. / np.sqrt(2.)
ISQR3 = 1. / np.sqrt(3.)
V_DICT = {
    "BULK": np.array([[ISQR2, -ISQR2, 0.],
                      [ISQR2, ISQR2, 0.],
                      [0.,     0.,    1.]]),
    # (P)ure Shear
    "PSHEAR": np.array([[ISQR2, ISQR2, 0.],
                        [-ISQR2, ISQR2, 0.],
                        [0.,    0.,    1.]]),
    "VERT": np.array([[0., 1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]]),
    "VERT2": np.array([[0., ISQR2, -ISQR2],
                       [1., 0.,     0.],
                       [0., ISQR2,  ISQR2]]),
    # (S)imple Shear
    "SSHEAR": np.array([[0., -ISQR2, ISQR2],
                       [0.,  ISQR2, ISQR2],
                       [1.,  0.,    0.]]),
    # alternative definition with different orthogonal vectors
    "SSHEARXY": np.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]]),
    # NSC: Normal-shear coupling
    "NSC": np.array([[0.,     0.,    1.],
                    [ISQR2, -ISQR2, 0.],
                    [ISQR2,  ISQR2, 0.]]),
    # alternative definitions with different orthogonal vectors
    "NSC2": np.array([[0.,     0.,    1.],
                     [ISQR2,  ISQR2, 0.],
                      [-ISQR2, ISQR2, 0.]]),
    "INSC": np.array([[1., 0., 0.],
                      [0., ISQR2, -ISQR2],
                      [0., ISQR2,  ISQR2]]),
    "NSC3": np.array([[1, -1/2,  3/2],
                      [1, -1/2, -3/2],
                      [1,  1,    0]]),
}


def is_orthonormalizable(matrix):
    """Check if a matrix can be orthonormalized by checking linear independence."""
    return np.linalg.matrix_rank(matrix) == matrix.shape[1]


bad_keys = []
for k, v in list(V_DICT.items()):
    try:
        if not is_orthonormalizable(v):
            logger.error(
                f"Matrix {k} cannot be orthonormalized - removing from V_DICT")
            bad_keys.append(k)
            continue

        # Normalize columns
        for column in v.T:
            norm = np.linalg.norm(column)
            if norm < 1e-10:
                raise ValueError(f"Zero or near-zero column in matrix {k}")
            column /= norm

        # Verify orthonormality
        if not np.allclose(v.T @ v, np.eye(3)):
            raise ValueError(
                f"Matrix {k} could not be properly orthonormalized")

    except Exception as e:
        logger.error(f"Error processing matrix {k}: {str(e)}")
        bad_keys.append(k)

# Remove invalid matrices
for k in bad_keys:
    V_DICT.pop(k)
