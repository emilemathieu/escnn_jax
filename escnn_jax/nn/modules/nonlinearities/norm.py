
from collections import defaultdict

# from torch.nn import Parameter

from escnn_jax.gspaces import *
from escnn_jax.nn import FieldType
from escnn_jax.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
# import torch
from jaxtyping import Array, PRNGKeyArray

from typing import List, Tuple, Any, Mapping, Optional, Dict

import numpy as np

__all__ = ["NormNonLinearity"]


class NormNonLinearity(EquivariantModule):
    _nfields: Optional[Dict[int, int]] = eqx.field(static=True)
    log_bias: Optional[Array]
    _function: Mapping[Array, Array] = eqx.field(static=True)
    _order: List = eqx.field(static=True)
    eps: Array = eqx.field(static=True)
    indices: Dict[int, Array] = eqx.field(static=True)
    _contiguous: Dict[int, bool] = eqx.field(static=True)

    def __init__(self, in_type: FieldType, function: str = 'n_relu', use_bias: bool = True):
        r"""
        
        Norm non-linearities.
        This module applies a bias and an activation function over the norm of each field.
        
        The input representation of the fields is preserved by this operation.
        
        .. note ::
            If 'squash' non-linearity (`function`) is chosen, no bias is allowed
        
        Args:
            in_type (FieldType): the input field type
            function (str, optional): the identifier of the non-linearity. It is used to specify which function to
                                      apply. By default (``'n_relu'``), ReLU is used.
            bias (bool, optional): add bias to norm of fields before computing the non-linearity. Default: ``True``

        """

        assert isinstance(in_type.gspace, GSpace)
        
        super(NormNonLinearity, self).__init__()

        for r in in_type.representations:
            assert 'norm' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "norm" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        
        self._nfields = None
        self.log_bias = None

        if function == 'n_relu':
            self._function = jax.nn.relu
        elif function == 'n_sigmoid':
            self._function = jax.nn.sigmoid
        elif function == 'n_softplus':
            self._function = jax.nn.softplus
        elif function == "squash":
            self._function = lambda t: t / (1.0 + t)
            assert use_bias is False, 'Error! When using squash non-linearity, norm bias is not allowed'
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
        # group fields by their size and
        #   - check if fields of the same size are contiguous
        #   - retrieve the indices of the fields

        # number of fields of each size
        self._nfields = defaultdict(int)
        
        # indices of the channales corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        
        # whether each group of fields is contiguous or not
        self._contiguous = {}
        
        position = 0
        last_size = None
        for i, r in enumerate(self.in_type.representations):
            
            if r.size != last_size:
                if not r.size in self._contiguous:
                    self._contiguous[r.size] = True
                else:
                    self._contiguous[r.size] = False
            last_size = r.size
            
            _indices[r.size] += list(range(position, position + r.size))
            self._nfields[r.size] += 1
            position += r.size
        
        for s, contiguous in self._contiguous.items():
            _indices[s] = jnp.array(_indices[s], dtype=int)
            # if contiguous:
            #     # for contiguous fields, only the first and last indices are kept
            #     _indices[s] = torch.LongTensor([min(_indices[s]), max(_indices[s])+1])
            # else:
            #     # otherwise, transform the list of indices into a tensor
            #     _indices[s] = torch.LongTensor(_indices[s])
                
            # # register the indices tensors as parameters of this module
            # self.register_buffer('indices_{}'.format(s), _indices[s])
        self.indices = _indices

        
        if use_bias:
            # build a bias for each field
            # self.log_bias = Parameter(torch.zeros(1, len(self.in_type), 1, 1, dtype=torch.float), requires_grad=True)
            self.log_bias = jnp.zeros((1, len(self.in_type), 1, 1), dtype=float)
        else:
            self.log_bias = None
    
        # build a sorted list of the fields groups, such that every time they are iterated through in the same order
        self._order = sorted(self._contiguous.keys())
        
        # self.eps = Parameter(torch.tensor(1e-10), requires_grad=False)
        self.eps = jnp.array(1e-10)
    
    def __call__(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        Apply norm non-linearities to the input feature map
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        
        # scalar multipliers needed to turn the old norms into the newly computed ones
        multipliers = jnp.empty_like(input)

        b, c = input.shape[:2]
        spatial_dims = input.shape[2:]

        next_bias = 0
        
        if self.log_bias is not None:
            # build the bias
            # biases = torch.nn.functional.elu(self.log_bias)
            biases = jnp.exp(self.log_bias)
            # biases = torch.nn.functional.elu(self.log_bias) + 1
        else:
            biases = None
        
        # iterate through all field sizes
        for s in self._order:
            
            # retrieve the corresponding fiber indices
            # indices = getattr(self, f"indices_{s}")
            indices = self.indices[s]
            
            # if self._contiguous[s]:
            #     # if the fields were contiguous, we can use slicing
            #     # retrieve the fields
            #     fm = input[:, indices[0]:indices[1], ...]
            # else:
            #     # otherwise we have to use indexing
            #     # retrieve the fields
            #     fm = input[:, indices, ...]
            fm = input[:, indices, ...]

            # compute the norm of each field
            # norms = fm.view(b, -1, s, *spatial_dims).norm(dim=2, keepdim=True)
            norms = jnp.linalg.norm(fm.reshape(b, -1, s, *spatial_dims), axis=2, keepdims=True)
            
            # compute the new norms
            if biases is not None:
                # retrieve the bias elements corresponding to the current fields
                bias = biases[:, next_bias:next_bias + self._nfields[s], ...].reshape(1, -1, 1, *[1]*len(spatial_dims))
                new_norms = self._function(norms - bias)
            else:
                new_norms = self._function(norms)

            # compute the scalar multipliers needed to turn the old norms into the newly computed ones
            # m = torch.zeros_like(new_norms)
            # in order to avoid division by 0
            # mask = norms > 0.
            # m[mask] = new_norms[mask] / norms[mask]
            # m = new_norms / jnp.max(norms, self.eps)
            m = new_norms / norms
            # m[norms <= self.eps] = 0.
            m = jnp.where(norms <= self.eps, 0., m)

            # if self._contiguous[s]:
            #     # expand the multipliers tensor to all channels for each field
            #     multipliers[:, indices[0]:indices[1], ...] = m.expand(b, -1, s, *spatial_dims).reshape(b, -1, *spatial_dims)
            
            # else:
            #     # expand the multipliers tensor to all channels for each field
            #     multipliers[:, indices, ...] = m.expand(b, -1, s, *spatial_dims).reshape(b, -1, *spatial_dims)
            multipliers = multipliers.at[:, indices, ...].set(jnp.broadcast_to(m, (b, m.shape[1], s, *spatial_dims)).reshape(b, -1, *spatial_dims))

            # shift the position on the bias tensor
            next_bias += self._nfields[s]
        
        # multiply the input by the multipliers computed and wrap the result in a GeometricTensor
        return GeometricTensor(input * multipliers, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, key: PRNGKeyArray, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
    
        x = jax.random.normal(key, (3, c, 10, 10))
    
        x = GeometricTensor(x, self.in_type)
    
        errors = []
    
        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))
        
            errs = np.array(out1.tensor - out2.tensor)
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert jnp.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors
