import warnings as wn

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd


def isotropic_elasticity_matrix(E, nu, plane='stress', output_style='mandel'):
    """
    Compute the elasticity matrix for isotropic materials under plane stress or strain conditions.

    This function calculates the elasticity matrix based on the provided Young's modulus,
    Poisson's ratio, and the given condition of plane stress or strain. The output matrix
    can be in either 'mandel' or 'standard' format.

    Parameters
    ----------
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    plane : str, optional
        Condition of the material. It can be either 'stress' or 'strain'.
        Default is 'stress'.
    output_style : str, optional
        Output format of the elasticity matrix. It can be either 'mandel' or 'standard'.
        Default is 'mandel'.

    Returns
    -------
    C : numpy.ndarray
        The calculated elasticity matrix of the material under given conditions.

    Raises
    ------
    ValueError
        If the output_style is neither 'mandel' nor 'standard'.

    References
    ----------
    - https://www.wikiwand.com/en/Hooke%27s_law#Linear_elasticity_theory_for_continuous_media
    - https://www.wikiwand.com/en/Plane_stress
    - https://www.wikiwand.com/en/Hooke%27s_law#Isotropic_materials
    """
    if (plane == 'stress'):
        alpha = E / (1 - nu**2)
        C = np.array([[1, nu, 0],
                      [nu, 1, 0],
                      [0, 0, (1 - nu)/2]])
        C *= alpha
    elif (plane == 'strain'):
        alpha = E / (1 + nu) / (1 - 2 * nu)
        C = np.array([[1 - nu, nu, 0],
                      [nu, 1 - nu, 0],
                      [0, 0, (1-2*nu)/2]])
        C *= alpha

    if output_style == 'mandel':
        s = np.diag(np.array([1, 1, np.sqrt(2)]))
        C = s@C@s
    elif output_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect output style')

    return C


def orthotropic_elasticity_matrix(E1, E2, G12, nu12, output_style='mandel'):
    '''
    Assembles a 3x3 elasticity matrix in Voigt notation
    Assumes plane stress for orthotropic material:
    https://www.wikiwand.com/en/Hooke%27s_law#Linear_elasticity_theory_for_continuous_media
    https://www.wikiwand.com/en/Hooke%27s_law#Anisotropic_materials
    but with constitutive equation of sqrt(2)*strain/stress, not 2*strain/stress.
    See "Which Elasticity Tensors are Realizable?", Milton eq 2.1.

    in: E1, Young's modulus in direction 1
        E2, Young's modulus in direction 2
        G12, Shear modulus of face 1 in direction 2
        nu12, Poisson ratio in direction 2 by pulling on face 1

    out: C, 3x3 elasticity matrix in Voigt notation
    '''
    nu21 = nu12 * E2 / E1
    alpha = 1. / (1. - nu12*nu21)
    C = np.array([[E1,        nu21 * E1,           0.],
                  [nu12 * E2, E2,                  0.],
                  [0.,        0.,      G12*(1. - nu12*nu21)]], dtype=np.float64)

    C = C * alpha

    if output_style == 'mandel':
        a = np.array([1, 1, np.sqrt(2)], dtype=np.double)
        I = np.eye(3)
        C = a*I@C@I*a
    elif output_style == 'standard':
        pass
    else:
        raise ValueError(
            'Incorrect output style. Acceptable values are "mandel" and "standard"')

    return C


def assemble_elasticity_matrix(values, input_style='ansys', output_style='mandel'):
    '''
    Reads in a list of values that correspond to either the upper right or bottom
    left triangle of the symmetric elastic matrix in standard Voigt notation.
    Outputs a 6x6 matrix following varying styles.

    in: values, a list of 21 values, must be 21 values long and should be ordered as:
        [C11, C21, C22, C31, C32, C33, C41, C42 ... C66]
        input_style, the structure of the matrix values being input. Can be 'ansys' or 'standard'
        output_style, the style which we output. Can be 'standard' or 'mandel'

    out: C, a symmetric 6x6 stiffness matrix in standard ordering

    Input Orders:
      Standard: Standard notation which follows strain vector ordering [11, 22, 33, 23, 13, 12]^T
      ANSYS: ANSYS specific matrix ordering following strain vector ordering [11, 22, 33, 12, 23, 13]^T

      Note:  Ansys structures their equations differently than standard notation when using material designer.
        See the Material Designer Users Guide pg 71, eq 3.1. Specifically the stress and strain vector ordering.
        Specifically this means that D4x corresponds to gamma12, which is different from standard notation.


    Output Styles:
      Standard: Standard engineering notation found everywhere on the internet
      Mandel: Applies a factor of sqrt(2) or 2 to values to change to an orthonormal
              basis for the tensor. Note this also changes stress and strain tensors.
              See Milton "The Theory of Composites" pg. 23 for more info or the
              reference below

    References:
      * https://csmbrannon.net/tag/mandel-notation/
    '''

    if (input_style == 'ansys'):
        C = make_symmetric_matrix(values, 6, order='F')
        C[[5, 3, 4], :] = C[[3, 4, 5], :]
        C[:, [5, 3, 4]] = C[:, [3, 4, 5]]
    else:
        C = make_symmetric_matrix(values, 6, order='C')

    if (output_style == 'mandel'):
        s = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2),
                     np.sqrt(2)], dtype=np.double)
        I = np.eye(6)
        C = s*I@C@I*s
    elif output_style == 'standard' or output_style == 'voigt':
        pass
    else:
        raise ValueError('Incorrect output style')

    # check mechanical stability criteria
    if np.linalg.det(C) < 0:
        wn.warn("Warning: The determinant of the elasticity matrix is negative. This is a non-physical result and should be analyzed carefully.")

    return C


def reduce_elasticity_matrix(C, plane='stress'):
    '''
    Used to reduce a 3D elasticity matrix in standard notation to a 2D elasticity matrix

    in: C, a 6x6 stiffness matrix
    out: Cp, a 3x3 stiffness matrix
    '''

    if (np.shape(C) != (6, 6)):
        raise ValueError('Input matrix size is not 6x6')

    P = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1]])

    if plane == 'stress':
        S = np.linalg.inv(C)
        Sp = P@S@P.T
        Cp = np.linalg.inv(Sp)
    elif plane == 'strain':
        Cp = P@C@P.T
    else:
        raise ValueError(
            'Incorrect plane state. Valid values are "stress" or "strain".')

    return Cp


def make_symmetric_matrix(values, N, order='C'):
    '''
    Takes in a vector of length L and produces a symmetric matrix that is NxN.

    inputs: values, list of values to plug into matrix
            N, size of matrix to produces as NxN matrix
            order, the ordering of the values. Follows standard numpy notation of
                   'F' for Fortran style (column major) and 'C' for C-style (row major)

    Reference: https://codereview.stackexchange.com/questions/107094/create-symmetrical-matrix-from-list-of-values
    '''
    if (len(values) != N*(N+1)/2):
        raise ValueError('Length of values and desired matrix size mismatch')
    m = np.zeros((N, N), dtype=np.float64)
    if order == 'F':
        rs, cs = np.triu_indices(N)
    elif order == 'C':
        rs, cs = np.tril_indices(N)
    else:
        raise ValueError('Incorrect input style')

    m[rs, cs] = values
    m[cs, rs] = values
    return m


def matrix2tensor(V, input_style='mandel'):
    '''
    Converts a 3x3 elasticity matrix into a 2x2x2x2 elasticity tensor

    in: V, a 3x3 elasticity matrix
        input_style, proper scaling for different matrix notation styles, e.g. Mandel or Voigt

    out: C, a 2x2x2x2 elasticity tensor

    Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
    '''
    lookup = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 2}
    # lookup3D = {(0,0): 0, (1,1): 1, (2,2): 2, (1,2): 3, (2,1): 3, (2,0): 4, (0,2): 4, (0,1): 5, (1,0): 5}
    C = np.zeros((2, 2, 2, 2), dtype=np.float64)

    if input_style == 'mandel':
        a = np.diag(np.array([1, 1, 1/np.sqrt(2)], dtype=np.double))
        V = a@V@a
    elif input_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect input style')

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(C.shape[2]):
                for l in range(C.shape[3]):
                    p = lookup[(i, j)]
                    q = lookup[(k, l)]
                    C[i, j, k, l] = V[p, q]

    return C


def tensor2matrix(C, output_style='mandel'):
    '''
    Converts a 2x2x2x2 elasticity tensor into a 3x3 elasticity matrix

    in: C, a 2x2x2x2 elasticity tensor
    out: V, a 3x3 elasticity matrix

    Reference for indexing: https://www.wikiwand.com/en/Linear_elasticity#Anisotropic_homogeneous_media
    '''
    lookup = {0: (0, 0), 1: (1, 1), 2: (0, 1)}
    V = np.zeros((3, 3), dtype=np.float64)
    for p in range(V.shape[0]):
        for q in range(V.shape[1]):
            i, j = lookup[p]
            k, l = lookup[q]
            V[p, q] = C[i, j, k, l]

    if output_style == 'mandel':
        a = np.diag(np.array([1, 1, np.sqrt(2)], dtype=np.double))
        V = a@V@a
    elif output_style == 'standard':
        pass
    else:
        raise ValueError('Incorrect output style')

    return V


def transform_elasticity_tensor(C, Q):
    """
    Transforms elasticity tensor `C` by the transformation matrix `Q`

    input: C, 2x2x2x2 elasticity tensor
           Q, 2x2 transformation matrix

    output: C', 2x2x2x2 transformed elasticity tensor
    """
    return np.einsum('mi,nj,ok,pl,ijkl->mnop', Q, Q, Q, Q, C)


def rotate_elasticity_matrix(M, theta, input_style='mandel', output_style='mandel'):
    '''
    Rotates a 3x3 elasticity matrix

    in: M, 3x3 matrix to rotate
        theta, angle to rotate by in radians
        input_style, matrix style, e.g. mandel or standard

    out: M_prime, rotated 3x3 elasticity matrix
    '''
    C = matrix2tensor(M, input_style)
    Q = rotmat(theta)
    C_prime = transform_elasticity_tensor(C, Q)
    M_prime = tensor2matrix(C_prime, output_style)

    return M_prime


def reflect_elasticity_matrix(M, dir='x', input_style='mandel', output_style='mandel'):
    """
    Reflects a 3x3 elasticity matrix about a direction `dir`

    in: M, 3x3 elasticity matrix C
        dir, direction about which we reflect, can be 'x' or 'y' only
        input_style, matrix style, e.g., mandel or standard (voigt)

    out: M_prime, reflected 3x3 elasticity matrix
    """

    C = matrix2tensor(M, input_style)
    Q = refmat(dir='y')
    C_prime = transform_elasticity_tensor(C, Q)
    M_prime = tensor2matrix(C_prime, output_style)

    return M_prime


def rotate_3d_elasticity_matrix(M, theta, axis='z', input_style='mandel', output_style='mandel'):
    '''
    Rotates a 6x6 elasticity matrix

    in: M, 6x6 matrix to be rotated
        theta, angle to rotate by in radians
        input_style, matrix style, e.g. mandel or standard

    out: M_prime, rotated 6x6 about the axis specified
    '''
    pass


def rotmat(theta):
    '''
    Generates a 2D coordinate rotation matrix


    in: theta, angle to rotate about the 3-axis
    out: Q, transformation matrix

    Note: This rotates the coordinate system, not the tensor
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def refmat(dir="x"):
    """
    Generates a 2D coordinate reflection matrix about the `dir` direction

    in: dir, can be 'x' or 'y'
    out: Q, transformation matrix

    """
    Q = np.eye(2)
    if dir == 'x':
        Q[1, 1] = -1
    elif dir == 'y':
        Q[0, 0] = -1
    else:
        raise ValueError("Reflection direction must be 'x' or 'y'")

    return Q


def calculate_elastic_properties(M, input_style='mandel'):
    '''
    Calculate the 2D elastic material properties of a 3x3 elasticity matrix

    in: M, a 3x3 elastic matrix
        input_style, input style of matrix, 'mandel' or 'standard'

    out: E1, Young's modulus in the direction of strain 1
         E2, Young's modulus in the direction of strain 2
         G12, Shear modulus of face 1 in the direction of strain 2
         nu12, Poisson ratio in strain 2 direction when applying normal strain 1
         eta121, shear strain when applying normal strain 1

    Reference:
      1. "Extreme values of Young’s modulus and Poisson’s ratio of hexagonal crystals" by Gorodstov
      2. "Fundamentals of crystal physics" by Sirotin, p654

    Note: Ref 1 is where I found the equation, but they cite Ref 2 for the equations
          I also don't have validation for my equation for shear modulus,
              it was just based on intuition and confirming agains the 2*22 case,
              but it may be wrong for an oblique matrix???
    '''
    S = matrix2tensor(np.linalg.inv(M), input_style=input_style)

    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    E1 = 1/np.einsum('ijkl,i,j,k,l', S, e1, e1, e1, e1)
    E2 = 1/np.einsum('ijkl,i,j,k,l', S, e2, e2, e2, e2)
    G12 = 1/np.einsum('ijkl,i,j,k,l', S, e1, e2, e1, e2)
    if input_style == 'mandel':
        G12 /= 4
    nu12 = -E1 * np.einsum('ijkl,i,j,k,l', S, e1, e1, e2, e2)
    nu21 = -E2 * np.einsum('ijkl,i,j,k,l', S, e2, e2, e1, e1)
    eta121 = E1 * np.einsum('ijkl,i,j,k,l', S, e1, e2, e1, e1)
    eta122 = E2 * np.einsum('ijkl,i,j,k,l', S, e1, e2, e2, e2)

    return E1, E2, G12, nu12, eta121, nu21, eta122


def anisotropy_index(C, input_style):
    '''
    Calculate the 2D anisotropy index of a given 2D 3x3 elasticity matrix

    in: C, a 3x3 elasticity matrix in standard (Voigt) notation
        input_style, can only be "standard" but is a check on the user that C is indeed in standard notation

    out: Kr, the Reuss estimate for bulk modulus
        Kv, the Voigt estimate for bulk modulus
        Gr, the Reuss estimate for shear modulus
        Gv, the Voigt estimate for shear modulus
        ASU, the anisotropy index

    references:
      * "Elastic anisotropy measure for two-dimensional crystals" by Li et al.
    '''

    if input_style != 'standard':
        raise ValueError('Only valid input style is "standard"')

    S = np.linalg.inv(C)
    Kr = 1/(S[0, 0]+S[1, 1]+2*S[0, 1])
    Gr = 2/(S[0, 0]+S[1, 1]-2*S[0, 1]+S[2, 2])
    Kv = (C[0, 0]+C[1, 1]+2*C[0, 1])/4
    Gv = (C[0, 0]+C[1, 1]-2*C[0, 1]+4*C[2, 2])/8
    ASU = np.sqrt((Kv/Kr - 1)**2 + 2*(Gv/Gr - 1)**2)

    return Kr, Kv, Gr, Gv, ASU


def polar_plot(M, input_style='mandel', orientation='horizontal', size=(1.5, 1.5)):
    '''
    Rotate the elasticity matrix around the 2D plane to generate a polar plot

    in: M, 3x3 elasticity matrix
        input_style, style of elasticity matrix. can be "mandel" or "standard"
        orientation, stack subplots horizontally or vertically
        size: base fig size

    out: ax, the axes object of the polar plot
         Es, the Young's modulus values for the rotated matrix
         Gs, the shear modulus values for the rotated matrix
         nus, the Poisson ratio values for the rotated matrix
    '''

    thetas, Es, Gs, nus = generate_planar_values(M, input_style)

    if orientation == 'horizontal':
        sub = (1, 2)
    elif orientation == 'vertical':
        sub = (2, 1)
    else:
        raise ValueError('orientation must be "horizontal" or "vertical"')

    fig, (ax1, ax2) = plt.subplots(sub[0], sub[1], figsize=(
        size[0], size[1]), subplot_kw={'projection': 'polar'})
    Emax = np.max(Es)
    Gmax = np.max(Gs)
    lwE, lwG = 1, 1
    if Emax > Gmax:
        lwE = 3
        lwG = 3
    else:
        lwE = 2
        lwG = 4

    lwE = 2
    lwG = 2
    leg_pos = (1.5, 1.2)
    ax1.plot(thetas, Es/Emax, label=r'$E_r$', linewidth=lwE)
    ax1.plot(thetas, Gs/Emax, label=r'$G_r$', linewidth=lwG)
    ax1.legend(loc='upper right', bbox_to_anchor=leg_pos)
    ax1.set_rmax(np.max([Es, Gs])/Emax)
    ax1.set_rticks([np.max(Gs)/Emax, 1])

    ax2.plot(thetas, nus, label=r'$\nu$', color='tab:green', linewidth=2)
    ax2.legend(loc='upper right', bbox_to_anchor=leg_pos)
    ax2.set_rticks([-1, 0, 0.5, 1])

    # fig.tight_layout(pad=0.5)

    return fig, ax1, ax2, (Es, Gs, nus)


def generate_planar_values(M, input_style):
    thetas = np.linspace(0, 2*np.pi, num=101, dtype=np.float64)
    Es, Gs, nus = [], [], []

    for theta in thetas:
        M_prime = rotate_elasticity_matrix(
            M, theta, input_style=input_style, output_style='mandel')
        E, _, G, nu, _, _, _ = calculate_elastic_properties(
            M_prime, input_style='mandel')
        Es.append(E)
        Gs.append(G)
        nus.append(nu)

    Es = np.asarray(Es)
    Gs = np.asarray(Gs)
    nus = np.asarray(nus)
    return thetas, Es, Gs, nus


def construct_S(E1, E2, G12, nu21, eta122, nu12, eta121):
    S11 = 1/E1
    S22 = 1/E2
    S44 = 1/G12

    S12 = -nu12/E1
    S21 = -nu21/E2
    S21 = 0.5 * (S21 + S12)

    # I would like to average S14 and S41, but this would require getting optical data from shear testing.
    # While doable, I think it's overkill
    S41 = eta121 / E1
    S42 = eta122 / E2

    S = np.array([[S11, S21, S41],
                  [S21, S22, S42],
                  [S41, S42, S44]])

    return S


def nearest_psd(A, min_eigval=5e-8):
    """
    Find the nearest positive semidefinite matrix to the given symmetric matrix A.

    Parameters
    ----------
    A : numpy.ndarray, shape (n, n)
        Input symmetric matrix.
    min_eigval : float, optional, default: 1e-6
        Minimum eigenvalue threshold. All eigenvalues smaller than this value
        will be set to this value before reconstructing the matrix.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        The nearest positive semidefinite matrix to A.
    """
    # Ensure the input matrix A is symmetric by averaging A with its transpose
    # Does nothing if the matrix is already symmetric
    B = (A + A.T) / 2

    # Compute the eigenvalues (D) and eigenvectors (Q) of the symmetric matrix B
    D, Q = np.linalg.eigh(B)

    # Set all eigenvalues smaller than min_eigval to min_eigval
    D = np.maximum(D, min_eigval)

    # Reconstruct the positive semidefinite matrix using the updated eigenvalues and the original eigenvectors
    return Q @ np.diag(D) @ Q.T


def normal_sample(theta, meas_mean, meas_std):
    """
    Generate a single artificial sample for a given theta.

    This function generates an artificial sample from a normal distribution,
    using the provided means and standard deviations for each column in the
    original DataFrame (except 'theta'). The generated sample is returned as
    a pandas Series where the columns are the same as the meas_mean columns.
    The 'theta' value is also included in the returned series.

    This is helpful to do a Monte Carlo simulation around measured values and establish
    a kind of distribution to see how error in the eigenvalue plots the measured
    error introduces.

    Parameters:
    theta (float): The theta value for which to generate the sample. Should be the index of the dataframes.
    meas_mean (DataFrame): The DataFrame of mean values for each 'theta'.
    meas_std  (DataFrame): The DataFrame of standard deviation values for each 'theta'.

    Returns:
    dict: A dictionary containing the generated sample.

    Note:
    This function assumes that each column in the provided DataFrames is independent.
    If columns are correlated, this function will not respect those correlations. In
    that case, consider using a multivariate normal distribution. This function also
    does not handle cases where the standard deviation is NaN.
    """
    means = meas_mean.loc[theta]
    stds = meas_std.loc[theta]

    # Generate a sample for each column at once
    sample_values = np.random.normal(loc=means, scale=stds)

    # Convert the sample values to a dictionary and add theta to the sample
    sample = {col: val for col, val in zip(meas_mean.columns, sample_values)}
    sample['theta'] = theta

    return pd.Series(sample)


def bootstrap_sample(df: pd.DataFrame, theta: float) -> pd.Series:
    """
    Generate a single bootstrap sample for a given theta by independently sampling a value from each column.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to generate the sample.
    theta (float): The theta value for which to generate the sample.

    Returns:
    pd.Series: A pandas Series containing the generated bootstrap sample.
    """
    # Check if the dataframe contains any rows with the specified theta value
    if theta not in df['theta'].unique():
        raise ValueError(f"No rows with theta={theta} found in the dataframe.")

    theta_df = df[df['theta'] == theta]

    # Check if the dataframe contains any columns
    if theta_df.empty:
        raise ValueError("The dataframe does not contain any columns.")

    # Drop rows with missing values before sampling
    theta_df = theta_df.dropna()

    sample_values = theta_df.apply(lambda col: col.sample(1).item())

    return sample_values


def sample_S(theta, meas_mean, meas_std):
    sample = normal_sample(theta, meas_mean, meas_std)
    return construct_S(sample.E1, sample.E2, sample.G12, sample.nu21, sample.eta122, sample.nu12, sample.eta121)


def vecang(v1, v2):
    num = la.norm(np.cross(v1, v2))
    den = la.norm(v1) * la.norm(v2) + np.dot(v1, v2)
    return 2.0*np.arctan2(num, den)
