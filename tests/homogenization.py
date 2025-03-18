import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from metatop.image import img2func
from metatop.mechanics import (calculate_elastic_constants,
                               convert_isotropic_properties)
from metatop.metamaterial import setup_metamaterial

np.set_printoptions(precision=2)

base_config = dict(
    E_max=1.,
    E_min=1e-9,
    nu=0.3,
    nelx=100,
    nely=100,
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


def check_props(actual, desired):
    for key in desired:
        if key not in actual:
            logger.error(f"Key {key} not found in actual properties.")
            return False
        if not np.isclose(actual[key], desired[key]):
            logger.error(
                f"Property {key} does not match: actual={actual[key]}, desired={desired[key]}")
            return False
    return True


def solid_black(metamate):

    metamate.x.vector()[:] = 1.
    Chom = metamate.solve()[1]
    sim_props = calculate_elastic_constants(Chom, input_style='standard')

    E = 1
    nu = 0.3
    p = convert_isotropic_properties({'E': E, 'nu': nu})

    real_props = make_prop_dict(E1=p['E'],
                                E2=p['E'],
                                G12=p['G'],
                                nu12=p['nu'],
                                nu21=p['nu'])

    if not check_props(real_props, sim_props):
        raise AssertionError()


def andreassen():
    """
    The image of the double arrowhead was screen capture from the paper, cropped, and here gets imported and loaded in as the density function. There is some error in between all of this so a few percent error in calculations I am happy with. At the time of the writing we see max ~3% error in the matrix.

    Ref:
    Andreassen, E., Lazarov, B. S. & Sigmund, O. Design of manufacturable 3D extremal elastic microstructure. Mech. Mater. 69, 1-10 (2014).  
    """
    img = ImageOps.invert(Image.open(
        "./tests/assets/andreassen_auxetic.png").convert("L"))
    thresh = 128
    def fn(x): return 255 if x > thresh else 0
    img = img.convert('L').point(fn, mode='1')

    config = base_config.copy()
    config['nelx'] = 200
    config['nely'] = 200
    config['mesh_cell_type'] = 'quad'
    metamate = setup_metamaterial(**config)
    metamate.x = img2func(img, metamate.mesh, metamate.R)
    # metamate.plot_density()
    # scale our matrix to match the paper's scaling
    Chom = metamate.solve()[1] / 1e-2

    published_C = np.array([[2.81, -2.53, 0],
                            [-2.53, 2.81, 0],
                            [0, 0, 2.67]])

    # truncate to the same level of precision as published and smooth out any very small values
    zero_threshold = np.linalg.norm(Chom) / 100  # 1% of the norm
    Chom_truncated = np.round(Chom, 2)
    Chom_truncated[np.abs(Chom_truncated) < zero_threshold] = 0.

    eps = 1e-6  # Small constant to avoid division by zero
    err = np.abs((published_C - Chom_truncated) / (published_C + eps))
    logger.info(f"Simulated Chom:\n{Chom_truncated}")
    logger.info(f"Published Chom:\n{published_C}")
    logger.info(f"Per element error (%):\n{100.*err}")
    logger.info(f"Frobenius norm of error: {np.linalg.norm(err)}")


if __name__ == "__main__":
    metamate = setup_metamaterial(**base_config)
    # solid_black(metamate)
    andreassen()
    plt.show()
