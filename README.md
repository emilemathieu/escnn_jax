
E(n)-equivariant Steerable CNNs (*escnn*)
--------------------------------------------------------------------------------
**[Documentation](https://quva-lab.github.io/escnn/)** | **[escnn](<https://github.com/QUVA-Lab/escnn/>) library** |

*escnn_jax* is a [Jax](http://jax.readthedocs.io/) port of the [PyTorch](https://pytorch.org/) [*escnn* library](https://github.com/QUVA-Lab/escnn/) for equivariant deep learning.
*escnn_jax* supports steerable CNNs equivariant to both 2D and 3D isometries, as well as equivariant MLPs.

--------------------------------------------------------------------------------

The library is structured into four subpackages with different high-level features:

| Component | Dependency | Description  |
|-----------------------------------------------------------------------------|------------------------------------------------------------------| ------------------------------------------------------------------|
| [**escnn.group**](https://github.com/QUVA-Lab/escnn/blob/master/group/) | Pure `Python`  | implements basic concepts of *group* and *representation* theory | test |
| [**escnn.gspaces**](https://github.com/QUVA-Lab/escnn/blob/master/gspaces/) | Pure `Python` | defines the Euclidean spaces and their symmetries                |
| [**escnn.kernels**](https://github.com/QUVA-Lab/escnn/blob/master/kernels/) | `Jax` | solves for spaces of equivariant convolution kernels             |
| [**escnn.nn**](https://github.com/QUVA-Lab/escnn/blob/master/nn/)   | `Equinox`       | contains equivariant modules to build deep neural networks       |
--------------------------------------------------------------------------------------------------------------------------------------------------

## TODOs

### Priority
- reproduce examples and baselines
    - [ ] `mlp.ipynb`
    - [ ] `introduction.ipynb`
    - [ ] `model.ipynb`
    - [ ] `octahedral_cnn.ipynb`
- [ ] mimic `requires_grad=false` for 'buffer' variables to avoid including them in `opt_state` and `grads`
- [ ] enhance `model.eval()` behaviour; make `EquivariantModule.eval` recursively call submodules?
- add common missing `escnn.nn.modules`
- [ ] speed up module's `__init__` e.g. `nn.Linear`
    - [ ] `InnerBatchNorm`
    - [x] `QuotientFourierELU`
    - [ ] `IIDBatchNorm1d`
    - [ ] `R3Conv`
    - [ ] `IIDBatchNorm3d`
    - [ ] `PointwiseAvgPoolAntialiased3D`
    - [x] `NormNonLinearity`
    - [x] `TensorProductModule`

### Nice to have
- [ ] add support for [`haiku`](https://dm-haiku.readthedocs.io/en/latest/) / [`flax`](https://flax.readthedocs.io/en/latest/) under `escnn.nn.haiku` / `escnn.nn.flax`
- [ ] `jaxlinop` for `Representation` class akin to [`emlp`](https://emlp.readthedocs.io), and more generally rewrite `escnn_jax.group` in `jax`?
- [ ] init function `deltaorthonormal_init`
- add missing `escnn.nn.modules`
    - [ ] `BranchingModule`
    - [ ] `MergeModule`
    - [ ] `MultipleModule`
    - [ ] `R3Conv`
    - [ ] `R2ConvTransposed`
    - [ ] `R3ConvTransposed`
    - [ ] `R3IcoConv`
    - [ ] `R3IcoConvTransposed`
    - [ ] `R2PointConv`, `R3PointConv`
    - [ ] `R2Upsampling`
    - [ ] `R3Upsampling`
    - [ ] `GatedNonLinearity1`
    - [ ] `GatedNonLinearity2`
    - [ ] `GatedNonLinearityUniform`
    - [ ] `InducedGatedNonLinearity1`
    - [ ] `NormNonLinearity`
    - [ ] `InducedNormNonLinearity`
    - [ ] `PointwiseNonLinearity`
    - [ ] `ConcatenatedNonLinearity`
    - [ ] `VectorFieldNonLinearity`
    - [ ] `QuotientFourierPointwise`
    - [ ] `QuotientFourierELU`
    - [ ] `TensorProductModule`
    - [ ] `ReshuffleModule`
    - [ ] `NormMaxPool`
    - [ ] `PointwiseMaxPool2D` `PointwiseMaxPool`
    - [ ] `PointwiseMaxPool3D`
    - [ ] `PointwiseMaxPoolAntialiased2D` `PointwiseMaxPoolAntialiased`
    - [ ] `PointwiseMaxPoolAntialiased3D`
    - [ ] `PointwiseAvgPool3D`
    - [ ] `PointwiseAvgPoolAntialiased3D`
    - [ ] `PointwiseAdaptiveAvgPool2D` `PointwiseAdaptiveAvgPool`
    - [ ] `PointwiseAdaptiveAvgPool3D`
    - [ ] `PointwiseAdaptiveMaxPool2D` `PointwiseAdaptiveMaxPool`
    - [ ] `PointwiseAdaptiveMaxPool3D`
    - [ ] `MaxPoolChannels`
    - [ ] `NormPool`
    - [ ] `InducedNormPool`
    - [ ] `InnerBatchNorm`
    - [ ] `NormBatchNorm`
    - [ ] `InducedNormBatchNorm`
    - [ ] `GNormBatchNorm`
    - [ ] `IIDBatchNorm1d`
    - [ ] `IIDBatchNorm2d`
    - [ ] `IIDBatchNorm3d`
    - [ ] `RestrictionModule`
    - [ ] `DisentangleModule`
    - [ ] `FieldDropout`
    - [ ] `PointwiseDropout`
    - [ ] `HarmonicPolynomialR3`

## Getting Started

*escnn_jax* is easy to use since it provides a high level user interface which abstracts most intricacies of group and representation theory away.
The following code snippet shows how to perform an equivariant convolution from an RGB-image to 10 *regular* feature fields (corresponding to a
[group convolution](https://arxiv.org/abs/1602.07576)).

```python3
from escnn_jax import gspaces                                          #  1
from escnn_jax import nn                                               #  2
import jax                                                             #  3
key = jax.random.PRNGKey(0)                                            #  4
key1, key2 = jax.random.split(key, 2)                                  #  5
                                                                       #  6
r2_act = gspaces.rot2dOnR2(N=8)                                        #  7
feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])         #  8
feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])         #  9
                                                                       # 10
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5, key=key1) # 11
relu = nn.ReLU(feat_type_out)                                          # 12
                                                                       # 13
x = jax.random.normal(key2, (16, 3, 32, 32))                           # 14
x = feat_type_in(x)                                                    # 15
                                                                       # 16
y = relu(conv(x))                                                      # 17
```

## Dependencies

The library is based on Python3.7

```
jax
equinox
jaxtyping
numpy
scipy
lie_learn
joblibx
py3nj
```
Optional:
```
pymanopt>=1.0.0
optax
distrax
chex
```

> **WARNING**: `py3nj` enables a fast computation of Clebsh Gordan coefficients.
If this package is not installed, our library relies on a numerical method to estimate them.
This numerical method is not guaranteed to return the same coefficients computed by `py3nj` (they can differ by a sign).
For this reason, models built with and without `py3nj` might not be compatible.

> To successfully install `py3nj` you may need a Fortran compiler installed in you environment.

## Installation

You can install the latest [release](https://github.com/QUVA-Lab/escnn/releases) as

```
pip install escnn_jax
```

or you can clone this repository and manually install it with
```
pip install git+https://github.com/QUVA-Lab/escnn_jax
```


## Contributing

Would you like to contribute to **escnn_jax**? That's great!

Then, check the instructions in [CONTRIBUTING.md](https://github.com/QUVA-Lab/escnn/blob/master/CONTRIBUTING.md) and help us to
improve the library!


## Cite

The development of this library was part of the work done for our papers
[A Program to Build E(N)-Equivariant Steerable CNNs](https://openreview.net/forum?id=WE4qe9xlnQw)
and [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).
Please cite these works if you use our code:

```

   @inproceedings{cesa2022a,
        title={A Program to Build {E(N)}-Equivariant Steerable {CNN}s },
        author={Gabriele Cesa and Leon Lang and Maurice Weiler},
        booktitle={International Conference on Learning Representations},
        year={2022},
        url={https://openreview.net/forum?id=WE4qe9xlnQw}
    }
    
   @inproceedings{e2cnn,
       title={{General E(2)-Equivariant Steerable CNNs}},
       author={Weiler, Maurice and Cesa, Gabriele},
       booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
       year={2019},
   }
```

Feel free to [contact us](mailto:cesa.gabriele@gmail.com).

## License

*escnn_jax* is distributed under BSD Clear license. See LICENSE file.
