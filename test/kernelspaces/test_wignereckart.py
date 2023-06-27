import unittest
from unittest import TestCase

import numpy as np

from escnn_jax.group import *
from escnn_jax.kernels import *

import jax
import jax.numpy as jnp
import numpy as np

class TestWEbasis(TestCase):
    
    # def test_spherical_shells(self):
    #     G = o3_group(4)
    #     X = SphericalShellsBasis(
    #         L=4,
    #         radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
    #     )

    #     irreps = [G.irrep(f, l) for f in range(2) for l in range(4)]
    #     for in_rep in irreps:
    #         for out_rep in irreps:
    #             self._check_irreps(X, in_rep, out_rep)
                
    def test_circular_shell(self):
        G = o2_group(8)
        
        axes = [0., np.pi/2, np.pi/3, np.pi/4] + (np.random.rand(5)*np.pi).tolist()
        for axis in axes:
            X = CircularShellsBasis(
                L=4,
                radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
                axis=axis
            )

            irreps = [G.irrep(0, 0)] + [G.irrep(1, l) for l in range(4)]
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep)

    def test_point(self):
        for G in [
            cyclic_group(1),
            cyclic_group(2),
            cyclic_group(3),
            cyclic_group(7),
            dihedral_group(1),
            dihedral_group(2),
            dihedral_group(3),
            dihedral_group(7),
            o2_group(8),
            so2_group(8),
            o3_group(6),
            so3_group(6),
            # ico_group(),
        ]:
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
            irreps = irreps[:min(4, len(irreps))]
            for in_rep in irreps:
                for out_rep in irreps:
                    X = PointBasis(G)
                    self._check_irreps(X, in_rep, out_rep)

    def _check_irreps(self, X: SteerableFiltersBasis, in_rep: IrreducibleRepresentation, out_rep: IrreducibleRepresentation):
        # print(X, in_rep, out_rep)
        key = jax.random.PRNGKey(0)
        
        G = X.group
        
        try:
            basis = WignerEckartBasis(X, in_rep, out_rep)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return
        
        P = 10
        # points = torch.randn(P, X.dimensionality)
        key, step_key = jax.random.split(key)
        points = jax.random.normal(step_key, (P, X.dimensionality))

        assert points.shape == (P, X.dimensionality)
        
        B = 5
        
        # features = torch.randn(P, B, in_rep.size)
        key, step_key = jax.random.split(key)
        features = jax.random.normal(step_key, (P, B, in_rep.size))
        
        # filters = jnp.zeros((P, basis.dim, out_rep.size, in_rep.size), dtype=float)
        # basis_sample = jax.jit(basis.sample)
        basis_sample = basis.sample
        filters = basis_sample(points)
        # filters = basis_sample(points)
        
        self.assertFalse(jnp.isnan(filters).any())
        self.assertFalse(jnp.allclose(filters, jnp.zeros_like(filters)))
        
        a = basis_sample(points)
        b = basis_sample(points)
        assert jnp.allclose(a, b)
        del a, b

        output = jnp.einsum("pfoi,pbi->fbo", filters, features)
        
        # for g in G.testing_elements():
        for _ in range(50):
            g = G.sample()

            output1 = jnp.einsum("oi,fbi->fbo",
                                   jnp.array(out_rep(g), dtype=output.dtype),
                                   output)
            a = X.action(g)
            transformed_points = points @ jnp.array(a, dtype=points.dtype).T

            transformed_filters = basis_sample(transformed_points)
            
            transformed_features = jnp.einsum("oi,pbi->pbo",
                                                jnp.array(in_rep(g), dtype=features.dtype),
                                                features)
            output2 = jnp.einsum("pfoi,pbi->fbo", transformed_filters, transformed_features)

            if not jnp.allclose(output1, output2, atol=1e-5, rtol=1e-4):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)
                
                aerr = jnp.abs(output1 - output2)
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(jnp.allclose(output1, output2, atol=1e-5, rtol=1e-4),
                            f"Group {G.name}, {in_rep.name} - {out_rep.name},\n"
                            f"element {g},\n"
                            f"action:\n"
                            f"{a}")
                            # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
