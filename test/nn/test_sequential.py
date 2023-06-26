# import torch
import random
import unittest
from unittest import TestCase

from escnn_jax.gspaces import *
from escnn_jax.nn import *

import jax

class TestSequential(TestCase):

    def test_iter(self):
        N = 8
        g = flipRot2dOnR2(N)

        r = g.type(*[g.regular_repr] * 3)

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        modules = [
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[0]),
            ReLU(r),
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[1]),
            ReLU(r),
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[2]),
        ]
        module = SequentialModule(*modules)

        self.assertEquals(len(modules), len(module))

        for i, module in enumerate(module):
            self.assertEquals(module, modules[i])

    def test_get_item(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        r = g.type(*[g.regular_repr]*3)

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        modules = [
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[0]),
            ReLU(r),
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[1]),
            ReLU(r),
            R2Conv(r, r, 3, use_bias=False, initialize=False, key=keys[2]),
        ]
        module = SequentialModule(*modules)

        self.assertEquals(len(modules), len(module))

        for i in range(len(module)):
            self.assertEquals(module[i], modules[i])

if __name__ == '__main__':
    unittest.main()
