import unittest
from unittest import TestCase

from escnn_jax.nn import *
from escnn_jax.gspaces import *
from escnn_jax.group import *

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import random


class TestBatchnorms(TestCase):
    
    # def test_o2_iid_bnorm(self):
    #     N = 3
    #     g = flipRot2dOnR2(-1, maximum_frequency=N)
        
    #     reprs = list(g.representations.values())
        
    #     irreps_sum = []
    #     for irr in g.irreps:
    #         irreps_sum += [irr] * int(irr.size // irr.sum_of_squares_constituents)
    #     reprs += [directsum(irreps_sum)]
        
    #     r = FieldType(g, reprs)
        
    #     bn = IIDBatchNorm2d(r, affine=False, momentum=1.)
        
    #     self.check_bn(bn)
    
    # def test_so2_iid_bnorm(self):
    #     N = 3
    #     g = rot2dOnR2(-1, maximum_frequency=N)
        
    #     reprs = list(g.representations.values())
        
    #     irreps_sum = []
    #     for irr in g.irreps:
    #         irreps_sum += [irr] * int(irr.size // irr.sum_of_squares_constituents)
    #     reprs += [directsum(irreps_sum)]
        
    #     r = FieldType(g, reprs)

    #     bn = IIDBatchNorm2d(r, affine=False, momentum=1.)
        
    #     self.check_bn(bn)
    
    # def test_dihedral_general_bnorm(self):
    #     N = 8
    #     g = flipRot2dOnR2(N)
        
    #     r = FieldType(g, list(g.representations.values()))
        
    #     bn = GNormBatchNorm(r, affine=False, momentum=1.)
        
    #     self.check_bn(bn)

    # def test_cyclic_general_bnorm(self):
    #     N = 32
    #     g = rot2dOnR2(N)

    #     r = FieldType(g, list(g.representations.values()))

    #     bn = GNormBatchNorm(r, affine=False, momentum=1.)

    #     self.check_bn(bn)

    # def test_so3_general_bnorm(self):
    #     g = rot3dOnR3()

    #     r = FieldType(g, [g.fibergroup.bl_regular_representation(2)] * 3)

    #     bn = GNormBatchNorm(r, affine=False, momentum=1.)

    #     self.check_bn(bn)

    # def test_so3_norm_bnorm(self):
    #     g = rot3dOnR3()
        
    #     r = FieldType(g, [g.irrep(l) for l in range(1, 4)]*2)
        
    #     bn = NormBatchNorm(r, affine=False, momentum=1.)
        
    #     self.check_bn(bn)

    # def test_cyclic_induced_norm(self):
    #     N = 8
    #     g = rot2dOnR2(N)

    #     sg_id = 4

    #     sg, _, _ = g.fibergroup.subgroup(sg_id)

    #     r = FieldType(g, [g.induced_repr(sg_id, r) for r in sg.irreps() if not r.is_trivial()])

    #     bn = InducedNormBatchNorm(r, affine=False, momentum=1.)

    #     self.check_bn(bn)

    # def test_dihedral_induced_norm(self):
    #     N = 8
    #     g = flipRot2dOnR2(N)
        
    #     sg_id = (None, N)
        
    #     sg, _, _ = g.fibergroup.subgroup(sg_id)
    
    #     r = FieldType(g, [g.induced_repr(sg_id, r) for r in sg.irreps() if not r.is_trivial()])
    
    #     bn = InducedNormBatchNorm(r, affine=False, momentum=1.)
    
    #     self.check_bn(bn)

    def test_dihedral_inner_norm(self):

        N = 8
        g = flipRot2dOnR2(N)
        
        g.fibergroup._build_quotient_representations()
        
        reprs = []
        for r in g.representations.values():
            if 'pointwise' in r.supported_nonlinearities:
                reprs.append(r)
        print("reprs", reprs)
        
        r = FieldType(g, reprs)
        bn = InnerBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)
        self.check_bn(bn.eval())

    def test_ico_inner_norm(self):
        g = icoOnR3()

        for sg in [
            (False, 1),
            (False, 2),
            (False, 3),
            (False, 5),
            (True, 1),
            # (True, 2),
            # (True, 3),
            # (True, 5),
        ]:
            print(sg)
            g.fibergroup.quotient_representation(sg)

        reprs = []
        for r in g.representations.values():
            if 'pointwise' in r.supported_nonlinearities:
                reprs.append(r)

        r = FieldType(g, reprs)

        bn = InnerBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)
        self.check_bn(bn.eval())

    def check_bn(self, bn: EquivariantModule):
        key = jax.random.PRNGKey(0)

        state = eqx.nn.State(bn)

        d = bn.in_type.gspace.dimensionality
        D = 500
        # x = 15*torch.randn(D, bn.in_type.size, *[1]*d)
        key, step_key = jax.random.split(key)
        x = 15*jax.random.normal(step_key, (D, bn.in_type.size, *[1]*d))
        
        p = 0
        trivials = []
        for irr in bn.in_type.irreps:
            irr = bn.in_type.fibergroup.irrep(*irr)
            if irr.is_trivial():
                trivials.append(p)
            p += irr.size
        
        if len(trivials) > 0:
            key, step_key = jax.random.split(key)
            bias = 2*jax.random.normal(step_key, (D, len(trivials), *[1]*d)) + 5.
            Q = jnp.array(bn.in_type.representation.change_of_basis[:, trivials], dtype=float)
            bias = jnp.einsum('ij,bj...->bi...', Q, bias)
            x += bias
        
        G = list(bn.in_type.gspace.testing_elements)
        for i in range(D):
            g = G[np.random.randint(len(G))]
            central_slice = i, slice(None), *[0]*d
            x = x.at[central_slice].set(jnp.array(bn.in_type.representation(g), dtype=float) @ x[central_slice])

        x = GeometricTensor(x, bn.in_type)
        
        # bn.reset_running_stats()
        # bn.train()
        
        # for name, size in bn._sizes:
        #     running_var = getattr(bn, f"{name}_running_var")
        #     running_mean = getattr(bn, f"{name}_running_mean")
        #     if running_mean.numel() > 0:
        #         self.assertTrue(torch.allclose(running_mean, torch.zeros_like(running_mean)))
        #     self.assertTrue(torch.allclose(running_var, torch.ones_like(running_var)))

        # bn(x)
        
        # bn.eval()
        # for name, size in bn._sizes:
        #     running_var = getattr(bn, f"{name}_running_var")
        #     running_mean = getattr(bn, f"{name}_running_mean")
        #     print(name)
        #     if running_mean.numel() > 0:
        #         print(running_mean.abs().max())
        #     print(running_var.abs().max())
        
        ys = []
        xs = []
        for g in bn.in_type.gspace.testing_elements:
            x_g = x.transform_fibers(g)
            y_g, state = bn(x_g, state)
            xs.append(x_g.tensor)
            ys.append(y_g.tensor)
        
        xs = jnp.concatenate(xs, axis=0)
        ys = jnp.concatenate(ys, axis=0)
        
        np.set_printoptions(precision=5, suppress=True, threshold=1000000, linewidth=3000)
        
        print([r.name for r in bn.out_type.representations])
        
        mean = xs.mean(0)
        std = xs.std(0)
        print('Pre-normalization stats')
        print(mean.reshape(-1))
        print(std.reshape(-1))

        mean = ys.mean(0)
        std = ys.std(0)
        print('Post-normalization stats')
        print(mean.reshape(-1))
        print(std.reshape(-1))
        
        self.assertTrue(jnp.allclose(mean, jnp.zeros_like(mean), rtol=4e-2, atol=2e-2))
        self.assertTrue(jnp.allclose(std, jnp.ones_like(std), rtol=4e-2, atol=2e-2), msg=f"""
            Standard deviations after normalization: \n {std.reshape(-1)}
        """)
        

if __name__ == '__main__':
    unittest.main()
