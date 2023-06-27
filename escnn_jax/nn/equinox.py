import equinox as eqx
from jaxtyping import Array


class ParameterArray(eqx.Module):
    array: Array

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype