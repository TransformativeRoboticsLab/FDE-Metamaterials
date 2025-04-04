import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from loguru import logger
from PIL import Image, ImageOps

from metatop.image import img2func
from metatop.mechanics import (calculate_elastic_constants,
                               convert_isotropic_properties, mandelize)
from metatop.Metamaterial import setup_metamaterial

np.set_printoptions(precision=3)

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


def pinstripe_horz(vol_frac=0.5):
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

    Chom = mandelize(metamate.solve()[1])
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    # values for a composite based on theory
    E_f = config['E_max']
    E_m = config['E_min']
    nu_f = config['nu']
    nu_m = nu_f
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    E_x = vol_frac * E_f + (1 - vol_frac) * E_m
    E_y = (vol_frac / E_f + (1 - vol_frac) / E_m)**-1
    G_xy = (vol_frac / G_f + (1 - vol_frac) / G_m)**-1
    nu_xy = vol_frac * nu_f + (1 - vol_frac) * nu_m
    nu_yx = nu_xy * E_y / E_x

    model_props = make_prop_dict(E1=E_x,
                                 E2=E_y,
                                 G12=G_xy,
                                 nu12=nu_xy,
                                 nu21=nu_yx)
    print(model_props)
    print(sim_props)

    if not check_props(sim_props, model_props):
        logger.error("FAIL: Pinstripe horizontal")
    else:
        logger.success("PASS: Pinstripe horizontal")


def pinstripe_vert(vol_frac=0.5):
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

    Chom = mandelize(metamate.solve()[1])
    sim_props = calculate_elastic_constants(Chom, input_style='mandel')

    # values for a composite based on theory, sub-f is fiber, sub-m is matrix
    E_f = config['E_max']
    E_m = config['E_min']
    nu_f = config['nu']
    nu_m = nu_f
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    E_y = vol_frac * E_f + (1 - vol_frac) * E_m
    E_x = (vol_frac / E_f + (1 - vol_frac) / E_m)**-1
    G_xy = (vol_frac / G_f + (1 - vol_frac) / G_m)**-1
    nu_yx = vol_frac * nu_f + (1 - vol_frac) * nu_m
    nu_xy = nu_yx * E_x / E_y

    model_props = make_prop_dict(E1=E_x,
                                 E2=E_y,
                                 G12=G_xy,
                                 nu12=nu_xy,
                                 nu21=nu_yx)
    print(model_props)
    print(sim_props)

    if not check_props(sim_props, model_props):
        logger.error("FAIL: Pinstripe vertical")
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
