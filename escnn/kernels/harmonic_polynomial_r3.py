
from typing import Dict, List, Tuple, Union

# import torch
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from escnn.group import *

__all__ = ["HarmonicPolynomialR3Generator"]


# class HarmonicPolynomialR3Generator(torch.nn.Module):
class HarmonicPolynomialR3Generator:
    cob: Dict[int, Array]

    def __init__(self, L: int):
        r"""
            Module which computes the *harmonic polynomials* in :math:`\R^3` up to order `L`.

            This equivariant module takes a set of 3-dimensional points transforming according to the
            :meth:`~escnn.group.O3.standard_representation` of :math:`O(3)` and outputs :math:`(L+1)^2` dimensional
            feature vectors transforming like spherical harmonics according to
            :meth:`~escnn.group.O3.bl_sphere_representation` with `L=L`.

            Harmonic polynomial are related to the spherical harmonics.
            In particular, when harmonic polynomials are evaluated on the unit sphere, they match the spherical
            harmonics.
            Check the
            `Wikipedia page <https://en.wikipedia.org/wiki/Spherical_harmonics#Harmonic_polynomial_representation>`_
            about them.

        """

        super(HarmonicPolynomialR3Generator, self).__init__()

        self.G: O3 = o3_group(L)

        self.L = L
        self.rho = self.G.bl_sphere_representation(L)
        self.d = self.rho.size

        if self.L > 0:
            # self.register_buffer(f'cob_1', torch.tensor(self.G.standard_representation().change_of_basis_inv, dtype=float))
            self.cob[1] = jnp.array(self.G.standard_representation().change_of_basis_inv, dtype=float)

        for l in range(2, self.L+1):
            rho_l = self.G.irrep((l-1)%2, l-1).tensor(self.G.irrep(1, 1))
            d = 2*l+1
            cob = rho_l.change_of_basis_inv[-d:, :]

            p = cob @ cob.T
            assert np.allclose(p, np.eye(2*l+1), atol=1e-5, rtol=1e-5), l

            # to ensure normalization

            # this guarantees the column is normalized, i.e. corresponds to the central column of the Wigner D matrix
            cob *= np.sqrt((2*l-1)/l)

            # self.register_buffer(f'cob_{l}', jnp.array(cob, dtype=float))
            self.cob[l] = jnp.array(cob, dtype=float)

    def forward(self, points: Array):
        assert points.shape[-1] == 3, points.shape
        shape = points.shape[:-1]

        features_0 = jnp.ones((*shape, 1), dtype=points.dtype) #device=points.device

        if self.L == 0:
            return features_0

        feature_1 = jnp.einsum('...i,ji->...j', points, getattr(self, 'cob_1'))

        features = [
            features_0,
            feature_1,
        ]

        for l in range(2, self.L+1):
            # cob = getattr(self, f'cob_{l}')
            cob = self.cob[l]
            f = jnp.einsum('...i,...j->...ij', features[-1], feature_1).reshape(*shape, -1)
            assert f.shape[-1] == 3*(2*l-1), (f.shape, l)
            f = jnp.einsum('...i,ji->...j', f, cob)
            assert f.shape[-1] == 2*l+1, (f.shape, l)

            features.append(f)

        features = jnp.concatenate(features, dim=-1)

        return features

    def check_equivariance(self, key: PRNGKeyArray, atol: float = 1e-5, rtol: float = 1e-3):

        # device = self.cob_1.device

        N = 40
        points = jax.random.normal(key, (N, 3))
        # points = torch.randn(N, 3, device=device)
        # radii = torch.norm(points, dim=-1).view(-1, 1)
        # points = points / radii
        # points[radii.view(-1) < 1e-3, :] = 0.

        sh = self(points)
        for _ in range(10):
            g = self.G.sample()

            sh_rot = self(points @ jnp.array(self.G.standard_representation()(g).T, dtype=float))
            rot_sh = sh @ jnp.array(self.rho(g).T, dtype=float)
            assert jnp.allclose(rot_sh, sh_rot, atol=atol, rtol=rtol), (rot_sh - sh_rot).abs().max().item()

