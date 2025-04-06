import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.testing as npt
from loguru import logger
from PIL import Image, ImageOps

from metatop.image import img2func
from metatop.mechanics import (calculate_elastic_constants,
                               convert_isotropic_properties, linear_strain,
                               linear_stress, macro_strain, mandelize)
from metatop.Metamaterial import setup_metamaterial

np.set_printoptions(precision=3)

# --- Helper Function for Annotations ---


def add_point_annotations(ax, func_to_evaluate, p1, p2, fmt=".2e", **kwargs):
    """
    Adds text annotations for function values at two points to a matplotlib axis.

    Args:
        ax: The matplotlib axis object.
        func_to_evaluate: The scalar FEniCS Function to evaluate.
        p1: The first FEniCS Point object.
        p2: The second FEniCS Point object.
        fmt: String format for the value display (default: scientific notation).
        **kwargs: Additional keyword arguments passed to ax.text (e.g., fontsize, color).
    """
    # Default annotation style
    style = dict(color='white', ha='center', va='center', fontsize=7,
                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    # Update style with any user-provided kwargs
    style.update(kwargs)
    try:
        val1 = func_to_evaluate(p1)
        val2 = func_to_evaluate(p2)
        text1 = f"{val1:{fmt}}"
        text2 = f"{val2:{fmt}}"
        ax.text(p1.x(), p1.y(), text1, **style)
        ax.text(p2.x(), p2.y(), text2, **style)
    except RuntimeError as err:
        print(
            f"Warning: Could not evaluate function for annotation on axis '{ax.get_title()}': {err}")

# --- Strain Plotting Function ---


def plot_strain_fields(mesh, strain_tensors, load_names, point1, point2, title="Total Strain Component Plots"):
    """
    Generates and shows plots for strain tensor components (total strain).

    Args:
        mesh: The FEniCS Mesh object.
        strain_tensors: List of total strain TensorFunctions [exx_total, eyy_total, exy_total].
        load_names: List of strings corresponding to the load cases.
        point1: First FEniCS Point for annotation.
        point2: Second FEniCS Point for annotation.
        title: Overall title for the figure.
    """
    DG0_scalar = fe.FunctionSpace(mesh, 'DG', 0)
    strain_names = ['Exx', 'Eyy', 'Exy', 'Eyx']
    strain_indices = [[0, 0], [1, 1], [0, 1], [1, 0]]
    nrows = len(strain_tensors)
    ncols = len(strain_indices)

    fig, axs = plt.subplots(nrows, ncols, figsize=(
        ncols*4 + 1, nrows*3 + 1), squeeze=False)  # Ensure axs is 2D

    for i, (e_tensor, row_axs) in enumerate(zip(strain_tensors, axs)):
        for j, (idx, ax) in enumerate(zip(strain_indices, row_axs)):
            plt.sca(ax)  # Set current axis
            e_component_func = fe.project(e_tensor[idx[0], idx[1]], DG0_scalar)
            # Use component name
            plot_title = f"{load_names[i]} | {strain_names[j]}"
            p = fe.plot(e_component_func, title=plot_title, cmap='viridis')
            cbar = plt.colorbar(p, ax=ax)
            cbar.ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter('%.1e'))  # Format colorbar

            if j == 0:
                ax.set_ylabel(load_names[i])  # Set y label on the axis
            if i == nrows - 1:
                ax.set_xlabel(strain_names[j])  # Set x label on the axis

            # Add annotations using the helper function
            add_point_annotations(ax, e_component_func,
                                  point1, point2, fmt=".2e")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

# --- Stress Plotting Function ---


def plot_stress_fields(mesh, stress_tensors, load_names, point1, point2, plot_syx=True, title="Total Stress Component Plots"):
    """
    Generates and shows plots for stress tensor components (total stress).

    Args:
        mesh: The FEniCS Mesh object.
        stress_tensors: List of total stress TensorFunctions [s_exx_load, s_eyy_load, s_exy_load].
        load_names: List of strings corresponding to the load cases.
        point1: First FEniCS Point for annotation.
        point2: Second FEniCS Point for annotation.
        plot_syx: Boolean, whether to include the Syx component plot.
        title: Overall title for the figure.
    """
    DG0_scalar = fe.FunctionSpace(mesh, 'DG', 0)
    if plot_syx:
        stress_names = ['Sxx', 'Syy', 'Sxy', 'Syx']
        stress_indices = [[0, 0], [1, 1], [0, 1], [1, 0]]
        ncols = 4
    else:
        stress_names = ['Sxx', 'Syy', 'Sxy']
        stress_indices = [[0, 0], [1, 1], [0, 1]]
        ncols = 3

    nrows = len(stress_tensors)
    figsize_width = ncols * 4 + 1  # Add some space for colorbars etc.
    figsize_height = nrows * 3 + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(
        figsize_width, figsize_height), squeeze=False)  # Ensure axs is 2D

    for i, (s_tensor, row_axs) in enumerate(zip(stress_tensors, axs)):
        for j, (idx, ax) in enumerate(zip(stress_indices, row_axs)):
            plt.sca(ax)  # Set current axis
            s_component_func = fe.project(s_tensor[idx[0], idx[1]], DG0_scalar)
            # Use component name
            plot_title = f"{load_names[i]} | {stress_names[j]}"
            p = fe.plot(s_component_func, title=plot_title, cmap='viridis')
            cbar = plt.colorbar(p, ax=ax)
            cbar.ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter('%.1e'))  # Format colorbar

            if j == 0:
                ax.set_ylabel(load_names[i])  # Set y label
            if i == nrows - 1:
                ax.set_xlabel(stress_names[j])  # Set x label

            # Add annotations using the helper function
            add_point_annotations(ax, s_component_func,
                                  point1, point2, fmt=".2e")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout


# These are the base parameters we used for the optimization process
base_config = dict(
    E_max=1.,
    E_min=1/30.,
    nu=0.4,
    nelx=50,
    nely=50,
)


def make_prop_dict(E1, E2, G12, nu12, nu21, eta122=0., eta121=0.):
    return dict(
        E1=E1,
        E2=E2,
        G12=G12,
        nu12=nu12,
        nu21=nu21,
        eta121=eta121,
        eta122=eta122,
    )


def check_props(simulated, expected):
    out = True
    for key in expected:
        if key not in simulated:
            logger.error(f"Key {key} not found in actual properties.")
            out = False
        if not np.isclose(simulated[key], expected[key]):
            logger.error(
                f"Property {key} does not match: simulated={simulated[key]:.2e}, expected={expected[key]:.2e}")
            out = False
    return out


def solid_black():
    logger.info("Testing solid black material...")

    metamate = setup_metamaterial(**base_config)
    metamate.x.vector()[:] = 1.
    Chom = mandelize(metamate.solve()[1])
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    p = convert_isotropic_properties({'E': base_config['E_max'],
                                      'nu': base_config['nu']})

    real_props = make_prop_dict(E1=p['E'],
                                E2=p['E'],
                                G12=p['G'],
                                nu12=p['nu'],
                                nu21=p['nu'])

    if not check_props(sim_props, real_props):
        logger.error("FAIL: Solid Black Material")
    else:
        logger.success("PASS: Solid Black Material")


def solid_white():
    logger.info('Testing solid white material...')

    metamate = setup_metamaterial(**base_config)
    metamate.x.vector()[:] = 0.  # solid white
    Chom = mandelize(metamate.solve()[1])
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    p = convert_isotropic_properties({'E': base_config['E_min'],
                                      'nu': base_config['nu']})

    real_props = make_prop_dict(E1=p['E'],
                                E2=p['E'],
                                G12=p['G'],
                                nu12=p['nu'],
                                nu21=p['nu'])

    if not check_props(sim_props, real_props):
        logger.error("FAIL: Solid White Material")
    else:
        logger.success("PASS: Solid White Material")


def voigt_prediction(x1, x2, f):
    # isostrain model
    # volume fraction `f` is for x1
    return f*x1 + (1-f)*x2


def reuss_prediction(x1, x2, f):
    # isostress model
    # volume fraction `f` is for x1
    return x1*x2 / (f*x1 + (1-f)*x2)


def pinstripe_horz(vol_frac=0.5, plot_strain=False, plot_stress=False):
    """ 
    Note: When running the pinstripe version materials we compare against the Voigt and Reuss mixture models. These are simple 1D models and technically operate as bounds on the material. Because they are 1D and our PBC enforce 2D boundary conditions we actually won't line up with the Reuss estimates; however, it still operates as a lower bound so as long as the calculate values is > Reuss estimate then we are all good.

    Extensive validation was done by looking at the stress and strain fields to understand why the Voigt model is exact, but the Reuss model is inexact with the homogenized values.
    """
    logger.info('Testing horizontal pinstripe material')

    config = base_config.copy()
    metamate = setup_metamaterial(**config)
    dof_coords = metamate.R.tabulate_dof_coordinates()
    high_density_dofs = np.where(dof_coords[:, 1] <= vol_frac)[0]
    metamate.x.vector()[:] = 0.
    metamate.x.vector()[high_density_dofs] = 1.

    if not np.allclose(metamate.volume_fraction, vol_frac):
        logger.error(
            "Metamaterial self-tabulated volume fraction not equal to deired volume fraction. All other answers are suspect")

    sols, Chom = metamate.solve()
    Chom = mandelize(Chom)
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    if plot_strain or plot_stress:
        # --- Calculations (only if plotting is needed) ---
        T = fe.TensorFunctionSpace(metamate.mesh, 'DG', 0)

        # Calculate TOTAL strain tensors
        exx_total_tensor = fe.project(
            linear_strain(sols[0]) + macro_strain(0), T)
        eyy_total_tensor = fe.project(
            linear_strain(sols[1]) + macro_strain(1), T)
        exy_total_tensor = fe.project(
            linear_strain(sols[2]) + macro_strain(2), T)

        # Calculate TOTAL stress tensors
        sxx_load_stress_tensor = fe.project(linear_stress(
            exx_total_tensor, metamate.E, metamate.prop.nu), T)
        syy_load_stress_tensor = fe.project(linear_stress(
            eyy_total_tensor, metamate.E, metamate.prop.nu), T)
        sxy_load_stress_tensor = fe.project(linear_stress(
            exy_total_tensor, metamate.E, metamate.prop.nu), T)

        # --- Setup for plotting functions ---
        load_names = ['Exx Load', 'Eyy Load', 'Exy Load']
        point1 = fe.Point(0.25, 0.5)  # Verify location in phase 1
        point2 = fe.Point(0.75, 0.5)  # Verify location in phase 2

        # --- Plotting Strain ---
        if plot_strain:
            strain_tensors = [exx_total_tensor,
                              eyy_total_tensor, exy_total_tensor]
            plot_strain_fields(metamate.mesh, strain_tensors,
                               load_names, point1, point2)

        # --- Plotting Stress ---
        if plot_stress:
            stress_tensors = [sxx_load_stress_tensor,
                              syy_load_stress_tensor, sxy_load_stress_tensor]
            plot_stress_fields(metamate.mesh, stress_tensors,
                               load_names, point1, point2)

        plt.show()

    # values for a composite based on theory
    E_f = config['E_max']
    E_m = config['E_min']
    nu_f = config['nu']
    nu_m = nu_f
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    E_x = voigt_prediction(E_f, E_m, vol_frac)
    E_y = reuss_prediction(E_f, E_m, vol_frac)
    G_xy = reuss_prediction(G_f, G_m, vol_frac)
    nu_xy = voigt_prediction(nu_f, nu_m, vol_frac)
    nu_yx = nu_xy * E_y / E_x

    model_props = make_prop_dict(E1=E_x,
                                 E2=E_y,
                                 G12=G_xy,
                                 nu12=nu_xy,
                                 nu21=nu_yx)

    if not check_props(sim_props, model_props):
        logger.error(
            "FAIL: Pinstripe horizontal, or is it? Double check the bound values. See note in code.")
    else:
        logger.success("PASS: Pinstripe horizontal")


def pinstripe_vert(vol_frac=0.5, plot_strain=False, plot_stress=False):
    """ 
    Note: When running the pinstripe version materials we compare against the Voigt and Reuss mixture models. These are simple 1D models and technically operate as bounds on the material. Because they are 1D and our PBC enforce 2D boundary conditions we actually won't line up with the Reuss estimates; however, it still operates as a lower bound so as long as the calculate values is > Reuss estimate then we are all good.

    Extensive validation was done by looking at the stress and strain fields to understand why the Voigt model is exact, but the Reuss model is inexact with the homogenized values.
    """
    logger.info('Testing vertical pinstripe material')

    config = base_config.copy()
    metamate = setup_metamaterial(**config)
    dof_coords = metamate.R.tabulate_dof_coordinates()
    high_density_dofs = np.where(dof_coords[:, 0] <= vol_frac)[0]
    metamate.x.vector()[:] = 0.
    metamate.x.vector()[high_density_dofs] = 1.

    if not np.allclose(metamate.volume_fraction, vol_frac):
        logger.error(
            "Metamaterial self-tabulated volume fraction not equal to deired volume fraction. All other answers are suspect")

    # Chom = mandelize(metamate.solve()[1])
    sols, Chom = metamate.solve()
    Chom = mandelize(Chom)
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    if plot_strain or plot_stress:
        # --- Calculations (only if plotting is needed) ---
        T = fe.TensorFunctionSpace(metamate.mesh, 'DG', 0)

        # Calculate TOTAL strain tensors
        exx_total_tensor = fe.project(
            linear_strain(sols[0]) + macro_strain(0), T)
        eyy_total_tensor = fe.project(
            linear_strain(sols[1]) + macro_strain(1), T)
        exy_total_tensor = fe.project(
            linear_strain(sols[2]) + macro_strain(2), T)

        # Calculate TOTAL stress tensors
        sxx_load_stress_tensor = fe.project(linear_stress(
            exx_total_tensor, metamate.E, metamate.prop.nu), T)
        syy_load_stress_tensor = fe.project(linear_stress(
            eyy_total_tensor, metamate.E, metamate.prop.nu), T)
        sxy_load_stress_tensor = fe.project(linear_stress(
            exy_total_tensor, metamate.E, metamate.prop.nu), T)

        # --- Setup for plotting functions ---
        load_names = ['Exx Load', 'Eyy Load', 'Exy Load']
        point1 = fe.Point(0.25, 0.5)  # Verify location in phase 1
        point2 = fe.Point(0.75, 0.5)  # Verify location in phase 2

        # --- Plotting Strain ---
        if plot_strain:
            strain_tensors = [exx_total_tensor,
                              eyy_total_tensor, exy_total_tensor]
            plot_strain_fields(metamate.mesh, strain_tensors,
                               load_names, point1, point2)

        # --- Plotting Stress ---
        if plot_stress:
            stress_tensors = [sxx_load_stress_tensor,
                              syy_load_stress_tensor, sxy_load_stress_tensor]
            plot_stress_fields(metamate.mesh, stress_tensors,
                               load_names, point1, point2)

        plt.show()

    # values for a composite based on theory, sub-f is fiber, sub-m is matrix
    E_f = config['E_max']
    E_m = config['E_min']
    nu_f = config['nu']
    nu_m = nu_f
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    E_x = reuss_prediction(E_f, E_m, vol_frac)
    E_y = voigt_prediction(E_f, E_m, vol_frac)
    G_xy = reuss_prediction(G_f, G_m, vol_frac)
    nu_yx = voigt_prediction(nu_f, nu_m, vol_frac)
    nu_xy = nu_yx * E_x / E_y

    model_props = make_prop_dict(E1=E_x,
                                 E2=E_y,
                                 G12=G_xy,
                                 nu12=nu_xy,
                                 nu21=nu_yx)

    if not check_props(sim_props, model_props):
        logger.error(
            "FAIL: Pinstripe vertical, or is it? Double check the bound values. See note in code.")
    else:
        logger.success("PASS: Pinstripe vertical")


def andreassen_fig_2a():
    """
    The image of the double arrowhead was screen captured from Fig. 2a, cropped, and here gets imported and loaded in as the density function. Because I don't have access to the direct density function, instead relying on the error-prone screen-grab/projection process I cannot exactly reproduce the results from the paper; however, we pass the test up to three decimal places in the homogenization process which I'll consider a win.

    Ref:
    Andreassen, E., Lazarov, B. S. & Sigmund, O. Design of manufacturable 3D extremal elastic microstructure. Mech. Mater. 69, 1-10 (2014).
    """
    logger.info('Testing Andreassen Fig 2a auxetic')
    # load the image, invert it because black in the image is zero value, but should be 1 in the density, then binarize it to remove any blurring/anti-aliasing effects from the inaccurate screengrab process
    img = ImageOps.invert(Image.open(
        "./tests/assets/Andreassen_Fig2a.png").convert("L"))
    # thresh = 128
    # img = img.point(lambda x: 255 if x > thresh else 0, mode='1')

    config = base_config.copy()
    # paper used 100x100 grid, but I double it to make sure we sample the image as finely as possible
    config['nelx'] = 200
    config['nely'] = 200
    # use quad mesh to try to stay true to paper's original implementation
    config['mesh_cell_type'] = 'quad'
    config['E_min'] = 1e-9
    config['nu'] = 0.3
    metamate = setup_metamaterial(**config)
    metamate.x = img2func(img, metamate.mesh, metamate.R)
    # metamate.plot_density()

    published_volume_fraction = 0.35  # should be ~35 %
    try:
        npt.assert_approx_equal(metamate.volume_fraction,
                                published_volume_fraction,
                                significant=2)
    except AssertionError as e:
        logger.error(
            f"Calculated volume fraction for Andreassen double arrowhead is incorrect: {e}")
    Chom = metamate.solve()[1]

    published_Chom = 1e-2 * np.array([[2.81, -2.53, 0],
                                      [-2.53, 2.81, 0],
                                      [0, 0, 2.67]])

    try:
        npt.assert_array_almost_equal(Chom, published_Chom, decimal=2)
        logger.success("PASS: Andreassen Fig. 2a double arrowhead")
    except AssertionError as e:
        logger.error(
            f"FAIL: Non-equivalence between homogenized C with Andreassen published C: {e}")


def wang_table_2():
    """"
    Same idea as the Andreassen test, except here I did the thresholding in GIMP so we don't need to threshold here.

    Ref:
    Wang, X., Chen, S. & Zuo, L. On Design of Mechanical Metamaterials Using Level-Set Based Topology Optimization. ASME 2015 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference V02BT03A012 (2016) doi:10.1115/DETC2015-47518.

    """
    logger.info('Testing Wang Table 2 material')

    img = ImageOps.invert(Image.open(
        "./tests/assets/Wang_Table2.png").convert("L"))

    config = base_config.copy()
    config['E_min'] = 1e-6
    metamate = setup_metamaterial(**config)
    metamate.x = img2func(img, metamate.mesh, metamate.R)
    # metamate.plot_density()
    Chom = metamate.solve()[1]
    # Modulus in paper has a 0.91 scale factor so we take that out to compare
    published_Chom = 1/0.91 * np.array([[0.147, -0.075, 0.],
                                        [-0.075, 0.141, 0.],
                                        [0., 0., 0.012]])

    try:
        npt.assert_almost_equal(Chom, published_Chom, decimal=2)
        logger.success("PASS: Wang Table 2")
    except AssertionError as e:
        logger.error(
            f"Equivalencve failed: Homogenized C with Wang Table 2 published C: {e}")


if __name__ == "__main__":
    solid_black()
    solid_white()
    pinstripe_horz()
    pinstripe_vert()
    andreassen_fig_2a()
    wang_table_2()
    # plt.show()
