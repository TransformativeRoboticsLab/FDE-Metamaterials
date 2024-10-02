import fenics as fe
import numpy as np
from scipy.spatial import KDTree


# when an epoch changes or we change beta the constraint values can jump
# and because the constraints can also be clamped by t we need to make sure
# that we start the epoch in a feasible state.
# Basically t could be too low for the constraints to be satisfied and the
# optimizer will spend cycles trying to get t up to a feasible value.
# We avoid this by jumping t to a feasible value at the start of each epoch
def update_t(x, gs):
    print(f"Updating t...\nOld t value {x[-1]:.3e}")
    new_t = -np.inf
    x[-1] = 0.
    for g in gs:
        results = np.zeros(g.n_constraints)
        g(results, x, np.array([]), dummy_run=True)
        new_t = max(new_t, *(results))
    x[-1] = new_t
    print(f"New t value: {x[-1]:.3e}")

def init_density(density_seed_type, vol_frac, dim):
    if density_seed_type == 'uniform':
        return np.random.uniform(0., 1., dim)
    elif density_seed_type == 'beta':
        return beta_function(vol_frac, dim)
    elif density_seed_type == 'binomial':
        return np.random.binomial(1, vol_frac, dim)
    else:
        raise ValueError(f"Invalid density_seed_type: {density_seed_type}")

def beta_function(vol_frac, size):
    # we use a reduced volume frac to ensure that the mean is actually below the desired frac. We want to ensure we start in a feasible region of the problem if we have an upper bound on the volume fraction.
    reduced_vol_frac = 0.95 * (vol_frac * (1. - vol_frac) / 0.1 - 1.)
    a = vol_frac * reduced_vol_frac
    b = (1. - vol_frac) * reduced_vol_frac
    return np.random.beta(a, b, size)

def mirror_density(density, fn_space, type='x'):
    if type == 'x':
        ref_angles = [np.pi/2]
        domain = lambda x: x[0] > 0.
    elif type == 'y':
        ref_angles = [np.pi]
        domain = lambda x: x[1] > 0.
    elif type == 'xy':
        ref_angles = [np.pi/2, np.pi, 3*np.pi/2]
        domain = lambda x: x[0] > 0. and x[1] > 0.
    elif type == 'xyd':
        ref_angles = [np.pi/4, np.pi/2, np.pi*3/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        domain = lambda x: x[0] > 0. and x[1] > 0. and x[1]/x[0] < 1.
    elif type == 'hex':
        ref_angles = [np.pi/6, np.pi/2, 5*np.pi/6, np.pi, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]
        domain = lambda x: x[0] > 0. and x[1] > 0. and x[1]/x[0] < 1./np.sqrt(3)
    else:
        raise ValueError(f"Invalid mirror type: {type}. Must be one of 'x', 'y', 'xy', 'xyd', 'hex'")

    mirror_density = density.copy()
    dofs = fn_space.tabulate_dof_coordinates()

    mesh = fn_space.mesh()
    rad = mesh.hmin()/2
    x_min, y_min = mesh.coordinates().min(axis=0)
    x_max, y_max = mesh.coordinates().max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    shifted_dofs = dofs - np.array([width, height])/2
    
    tree = KDTree(shifted_dofs)
    
    def ref(p, th):
        return np.array([[np.cos(2*th), np.sin(2*th)],
                         [np.sin(2*th), -np.cos(2*th)]]) @ p


    # We do this as a series of reflections, propagating the density around the domain
    # NOTE: We aren't checking the domain, because there isn't an exact mapping when using the hex. Instead we do this for all points and it still works out, even if it is inefficient and slower.
    mirror_source = []
    mirror_target = []
    for n, c in enumerate(shifted_dofs):
        if not domain(c):
            continue
        ref_c = c
        for th in ref_angles:
            ref_c = ref(ref_c, th)
            d, idx = tree.query(ref_c, distance_upper_bound=rad)
            if not np.isinf(d):
                mirror_source.append(n)
                mirror_target.append(idx)
                mirror_density[idx] = mirror_density[n]

    return mirror_density, (mirror_source, mirror_target)

class Ellipse(fe.UserExpression):
    
    def __init__(self, V, a, b):
        super().__init__()
        self.V = V
        self.a = a
        self.b = b
        
    def eval(self, values, x):
        xc, yc, a, b = 0.5, 0.5, self.a, self.b
        values[0] = 0.5*self.V if ((x[0] - xc)/a)**2 + ((x[1] - yc)/b)**2 < 1 else self.V
        # values[0] = 0.5*self.V if (x[0] - xc)**2 + (x[1] - yc)**2 < r**2 else self.V