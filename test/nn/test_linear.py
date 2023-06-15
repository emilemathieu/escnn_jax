import unittest
from unittest import TestCase

import equinox as eqx
# import torch
import jax
import jax.numpy as jnp
import numpy as np

import escnn.nn.init as init
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

key = jax.random.PRNGKey(0)

class TestConvolution(TestCase):
    
    # def test_cyclic(self):
    #     N = 8
    #     g = no_base_space(cyclic_group(N))
        
    #     r1 = FieldType(g, list(g.representations.values()))
    #     r2 = FieldType(g, list(g.representations.values()) * 2)
    #     # r1 = FieldType(g, [g.trivial_repr])
    #     # r2 = FieldType(g, [g.regular_repr])
        
    #     cl = Linear(r1, r2, bias=True, key=key)
    #     # cl.bias.data = 20*torch.randn_like(cl.bias.data)
    #     cl.bias = 20 * jax.random.normal(key, cl.bias.shape)

    #     for _ in range(1):
    #         cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
    #         cl.eval()
    #         cl.check_equivariance()
        
    #     cl.train()
    #     for _ in range(1):
    #         cl.check_equivariance()
        
    #     cl.eval()
        
    #     for _ in range(5):
    #         cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
    #         cl.eval()
    #         matrix = cl.matrix.clone()
    #         cl.check_equivariance()
    #         self.assertTrue(jnp.allclose(matrix, cl.matrix))

    def test_so2(self):
        key = jax.random.PRNGKey(0)

        N = 7
        g = no_base_space(so2_group(N))

        # reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = g.type(*reprs)
        r2 = g.type(*reprs)

        from escnn.nn.modules.basismanager import (BasisManager,
                                                   BlocksBasisExpansion)
        basismanager = BlocksBasisExpansion(r1.representations, r2.representations, r1.gspace.build_fiber_intertwiner_basis, np.zeros((1, 1)))
        print(basismanager.dimension())
        print(dict(basismanager.get_basis_info()))
        
        for _ in range(8):
            # cl.basisexpansion._init_weights()
            # cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
            key, w_key, e_key = jax.random.split(key, 3)
            cl = Linear(r1, r2, bias=True, key=w_key)
            print(cl.weights)
            print(init.generalized_he_init(w_key, cl.weights, cl.basisexpansion))
            raise
            # cl.eval()
            cl = eqx.tree_inference(cl, True)
            cl.check_equivariance(e_key)

    # def test_dihedral(self):
    #     N = 8
    #     g = no_base_space(dihedral_group(N))

    #     r1 = FieldType(g, list(g.representations.values()))
    #     r2 = FieldType(g, list(g.representations.values()))
    #     # r1 = FieldType(g, [g.trivial_repr])
    #     # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
    #     # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
    #     # r2 = FieldType(g, [g.regular_repr])
    
    #     cl = Linear(r1, r2, bias=True, key=key)

    #     for _ in range(8):
    #         # cl.basisexpansion._init_weights()
    #         cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
    #         cl.eval()
    #         cl.check_equivariance()

    # def test_o2(self):
    #     N = 7
    #     g = no_base_space(o2_group(N))

    #     reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(4)]
    #     r1 = g.type(*reprs)
    #     r2 = g.type(*reprs)

    #     cl = Linear(r1, r2, bias=True, key=key)

    #     for _ in range(8):
    #         cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
    #         cl.eval()
    #         cl.check_equivariance()

    # def test_so3(self):
    #     g = no_base_space(so3_group(1))

    #     reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] + [g.fibergroup.bl_regular_representation(3)]
    #     r1 = g.type(*reprs)
    #     r2 = g.type(*reprs)

    #     cl = Linear(r1, r2, bias=True, key=key)
        
    #     for _ in range(8):
    #         cl.weights = init.generalized_he_init(key, cl.weights, cl.basisexpansion)
    #         # cl.weights.data.normal_()
    #         cl.eval()
    #         cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
