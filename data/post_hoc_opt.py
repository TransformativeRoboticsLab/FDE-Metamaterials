import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from incense import ExperimentLoader
from incense.artifact import PickleArtifact
from loguru import logger

from metatop import V_DICT
from metatop.filters import jax_projection, setup_filter
from metatop.mechanics import (calculate_elastic_constants,
                               calculate_elastic_constants_jnp_no_einsum,
                               mandelize)
from metatop.Metamaterial import setup_metamaterial
from metatop.optimization.OptimizationComponents import \
    ScalarOptimizationComponent
from metatop.optimization.OptimizationState import OptimizationState


def filter_and_project(ops, x):
    x = ops.filt_fn(x)
    return jax_projection(x, ops.beta, ops.eta)


def load_exps(exp_ids):
    logger.info("Loading experiments")
    exp_dict = {}

    loader = ExperimentLoader(mongo_uri="mongodb://localhost:27017",
                              db_name="metatop")

    exps = [loader.find_by_id(id) for id in exp_ids]

    for exp in exps:
        config = exp.to_dict()['config']
        for k, v in exp.artifacts.items():
            if 'pkl' in k.lower():
                data = v.as_type(PickleArtifact).render()
                final_x = data['x_history'][-1]

        metamaterial_config = dict(
            E_max=config['E_max'],
            E_min=config['E_min'],
            nu=config['nu'],
            nelx=config['nelx'],
            nely=config['nely']
        )
        metamate = setup_metamaterial(**metamaterial_config)

        filt, filt_fn = setup_filter(metamate, config['norm_filter_radius'])
        final_beta = config['start_beta']*2**(config['n_betas'] - 1)
        ops_config = dict(
            basis_v=V_DICT[config['basis_v']],
            extremal_mode=config['extremal_mode'],
            metamaterial=metamate,
            filt=filt,
            filt_fn=filt_fn,
            beta=final_beta,
            show_plot=False,
            verbose=False,
            silent=True,
        )
        ops = OptimizationState(**ops_config)
        metamate.x.vector()[:] = filter_and_project(ops, final_x)

        sample_dict = dict(
            final_x=final_x,
            exp=exp,
            ops=ops,
            metamaterial=metamate,
            config=config,
        )

        exp_dict |= {exp.id: sample_dict}

    return exp_dict


class PropertyMatchObjective(ScalarOptimizationComponent):

    def __init__(self, exp_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.exp_dict = exp_dict
        self.exp_ids = exp_dict.keys()
        self.N = len(exp_dict)

        self._compare_props = ('E1', 'E2', 'nu12', 'nu21', 'eta122', 'eta121')

        # self.fig, self.ax = plt.subplots()
        self.obj_evals = []

    def __call__(self, x, grad):

        xs = np.split(x, self.N)
        gs = []
        obj_eval = 0.

        # Run the forward simulation for all the metamaterials we are tyring to fit. Return the sum error of all the properties and update the gradient so they will fit better.
        for id, x in zip(self.exp_ids, xs):
            sols, Chom, dChom_dxfem, dxfem_dx_vjp = self.forward(x)
            Chom = jnp.array(Chom)
            def _eval(C): return self.eval(C, id)
            c, dc_dChom = jax.value_and_grad(_eval)(Chom)
            obj_eval += c

            if grad.size > 0:
                gs.append(self.adjoint(dc_dChom, dChom_dxfem, dxfem_dx_vjp))

        if grad.size > 0:
            grad[:] = np.concatenate(gs)
            if self.verbose:
                logger.debug(np.linalg.norm(grad))

        if not self.silent:
            logger.info(
                f"{len(self.obj_evals)+1:03d}:\tf(x) -- {obj_eval:.4e}")

        self.obj_evals.append(obj_eval)

        return float(obj_eval)

    # Objective comparing properties for an individual metamaterial
    def eval(self, C, id):
        M = mandelize(C)
        sim_props = calculate_elastic_constants_jnp_no_einsum(M)
        err = 0.
        weights = {'E1': 1., 'E2': 1., 'nu12': 1.,
                   'nu21': 1., 'eta122': 1e-2, 'eta121': 1e-2}
        for p in self._compare_props:
            num = weights[p]*(sim_props[p] - self.exp_dict[id][p])
            den = self.exp_dict[id][p]
            err += (num/den)**2

        return err


def main():

    ids = [1467, 2246, 1049]
    exp_dict = load_exps(ids)
    measured_props = {1049:
                      {'E1': 1.85, 'E2': 0.99, 'nu21': 0.154,
                       'nu12': 0.275, 'eta121': 1e-3, 'eta122': 1e-3},
                      2246:
                      {'E1': 2.10, 'E2': 1.62, 'nu21': 0.058,
                       'nu12': 0.063, 'eta121': 1e-3, 'eta122': 1e-3},
                      1467:
                      {'E1': 0.86, 'E2': 0.84, 'nu21': 0.378,
                       'nu12': 0.404, 'eta121': 1e-3, 'eta122': 1e-3}
                      }
    E_max = 3.2
    for v in measured_props.values():
        v['E1'] /= E_max
        v['E2'] /= E_max
    print(measured_props)
    for k, v in exp_dict.items():
        if k in measured_props:
            v |= measured_props[k]
        else:
            logger.error(f"{k} id not in measured data")

    n_dims = sum(data['metamaterial'].R.dim() for data in exp_dict.values())
    # print(n_dims)
    opt = nlopt.opt(nlopt.LD_MMA, n_dims)
    opt.set_lower_bounds(0.)
    opt.set_upper_bounds(1.)
    opt.set_maxeval(100)
    opt.set_ftol_rel(1e-8)
    opt.set_xtol_rel(1e-4)

    ops = exp_dict[1467]['ops']
    ops.silent = False
    ops.verbose = False
    f = PropertyMatchObjective(exp_dict, ops=exp_dict[1467]['ops'])
    opt.set_min_objective(f)

    x = np.hstack([data['final_x'] for data in exp_dict.values()])
    orig_x = x.copy()
    x[:] = opt.optimize(x)

    fig, axs = plt.subplots(2, len(ids), figsize=(4*len(ids), 8))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    xs = np.split(x, len(ids))
    orig_xs = np.split(orig_x, len(ids))

    for ax, x, exp in zip(axs[0], orig_xs, exp_dict.values()):
        m = exp['metamaterial']
        m.x.vector()[:] = filter_and_project(exp['ops'], x)
        m.plot_density(ax=ax, block=False)
        plt.tight_layout()

    for ax, x, (id, exp) in zip(axs[1], xs, exp_dict.items()):
        print(f"ID: {id}")
        m = exp['metamaterial']
        m.x.vector()[:] = filter_and_project(exp['ops'], x)
        m.plot_density(ax=ax, block=False)
        Chom = mandelize(m.solve()[1])
        new_sim_props = calculate_elastic_constants(Chom)

        for k, v in new_sim_props.items():
            mp = measured_props[id]
            if k in mp:
                print(10*'=' + k + 10*'=')
                print(f"Meas: {mp[k]:.3f}, Sim: {v:.3f}")
                print(f"Diff: {mp[k] - v}")
                print(f"Err: {abs(mp[k] - v)/abs(mp[k])}")
                print()
        plt.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(f.obj_evals)
    ax.set(yscale='log')

    plt.show(block=True)


if __name__ == "__main__":
    main()
