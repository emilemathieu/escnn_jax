import inspect
import time
# import sys
# sys.path.append('../')
# import torch
from escnn_jax import gspaces
from escnn_jax import group
from escnn_jax import nn

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import numpy as np
import torch
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray  # https://github.com/google/jaxtyping
from typing import List, Dict

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode

import numpy as np

from PIL import Image


# download the dataset
# !wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
# # uncompress the zip file
# !unzip -n mnist_rotation_new.zip -d mnist_rotation_new

class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode='F')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)


class C8SteerableCNN(nn.EquivariantModule):
    layers: list[eqx.Module]
    fully_net: list
    input_type: nn.FieldType

    def __init__(self, key, n_classes=10):
        keys = jax.random.split(key, 8)
        layers = []

        super(C8SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(r2_act, [r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(r2_act, 24*[r2_act.regular_repr])
        layers.extend([
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, use_bias=False, key=keys[0]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 1")
        
        # convolution 2
        # the old output type is the input type to the next layer
        # in_type = self.block1.out_type
        in_type = out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(r2_act, 48*[r2_act.regular_repr])
        layers.extend([
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, use_bias=False, key=keys[1]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 2")
        layers.append(nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
        print("PointwiseAvgPoolAntialiased")
        
        # convolution 3
        # the old output type is the input type to the next layer
        # in_type = self.block2.out_type
        in_type = out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(r2_act, 48*[r2_act.regular_repr])
        layers.extend([
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, use_bias=False, key=keys[2]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 3")
        
        # convolution 4
        # the old output type is the input type to the next layer
        # in_type = self.block3.out_type
        in_type = out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(r2_act, 96*[r2_act.regular_repr])
        layers.extend([
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, use_bias=False, key=keys[3]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 4")
        layers.append(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        print("PointwiseAvgPoolAntialiased")
        
        # convolution 5
        # the old output type is the input type to the next layer
        # in_type = self.block4.out_type
        in_type = out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(r2_act, 96*[r2_act.regular_repr])
        layers.extend([
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, use_bias=False, key=keys[4]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 5")
        
        # convolution 6
        # the old output type is the input type to the next layer
        # in_type = self.block5.out_type
        in_type = out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(r2_act, 64*[r2_act.regular_repr])
        layers.extend([
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, use_bias=False, key=keys[5]),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type)
        ])
        print("conv 5")
        layers.append(nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0))
        print("PointwiseAvgPoolAntialiased")
        
        layers.append(nn.GroupPooling(out_type))
        print("GroupPooling")

        self.layers = layers
        
        # number of output channels
        # c = self.gpool.out_type.size
        c = layers[-1].out_type.size
        
        # Fully Connected
        self.fully_net = [
            jnp.ravel,
            eqx.nn.Linear(c, 64, key=keys[6]),
            # eqx.nn.BatchNorm(64, axis_name="batch"),
            jax.nn.elu,
            eqx.nn.Linear(64, n_classes, key=keys[7]),
            # jax.nn.log_softmax
        ]
        print("Sequential")
    
    def __call__(self, input: Array, state: eqx.nn.State):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        for layer in self.layers:
            # print(layer, x.shape)
            if "state" in inspect.signature(layer).parameters:
                x, state = layer(x, state)
            else:
                x = layer(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        # x = x.reshape(x.shape[0], -1)
        
        # classify with the final fully connected layers)
        for layer in self.fully_net:
            # print("x", x.shape)
            if "state" in inspect.signature(layer).parameters:
                x, state = jax.vmap(layer, (0, None), (0, None), axis_name="batch")(x, state)
            else:
                x = jax.vmap(layer, axis_name="batch")(x)

        return x, state
    
    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) ==2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
    

@eqx.filter_jit
def loss(
    # model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    model, state, x: Float[Array, "batch 1 29 29"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    # pred_y = jax.vmap(model)(x)
    pred_y, state = model(x, state)
    loss = cross_entropy(y, pred_y)
    return loss, state


@eqx.filter_jit
def loss2(params, static, *args):
    is_param = lambda x: isinstance(x, nn.ParameterArray)
    model = eqx.combine(params, static, is_leaf=is_param)
    return loss(model, *args)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    # pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    # return -jnp.mean(pred_y)
    return jnp.mean(optax.softmax_cross_entropy(logits=pred_y, labels=jax.nn.one_hot(y, 10)))


@eqx.filter_jit
def compute_accuracy(
    model: eqx.Module, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    # pred_y = jax.vmap(model)(x)
    pred_y, _ = model(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model: eqx.Module, state: eqx.nn.State, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    inference_model = model.eval()
    # inference_model = eqx.tree_inference(model, value=True)
    inference_model = eqx.Partial(inference_model, state=state)

    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = jnp.array(x.numpy())
        y = jnp.array(y.numpy())
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        # avg_loss += loss(lambda x, _: inference_model(x), state, x, y)[0]
        avg_acc += compute_accuracy(inference_model, x, y)
    # return avg_loss / len(testloader), avg_acc / len(testloader)
    return avg_acc / len(testloader)


def train(
    model: eqx.Module,
    state: eqx.nn.State,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    # steps: int,
    epochs: int,
    print_every: int,
) -> eqx.Module:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    # opt_state = optim.init(eqx.filter(model, eqx.is_array))
    is_param = lambda x: isinstance(x, nn.ParameterArray)
    opt_state = optim.init(eqx.filter(model, is_param, is_leaf=is_param))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: eqx.Module,
        state: eqx.nn.State, 
        opt_state: PyTree,
        # x: Float[Array, "batch 1 28 28"],
        x: Float[Array, "batch 1 29 29"],
        y: Int[Array, " batch"],
    ):
        params, static = eqx.partition(model, is_param, is_leaf=is_param)
        # (loss_value, state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, state, x, y)
        (loss_value, state), grads = jax.value_and_grad(loss2, has_aux=True)(params, static, state, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader
 
    elapsed_time = 0.
    for epoch in range(31):
        start = time.time()

        # for step, (x, y) in zip(range(steps), infinite_trainloader()):
        for step, (x, y) in enumerate(trainloader):
            # PyTorch dataloaders give PyTorch tensors by default,
            # so convert them to NumPy arrays.
            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            model, state, opt_state, train_loss = make_step(model, state, opt_state, x, y)
            # if (step % print_every) == 0 or (step == steps - 1):
        end = time.time()
        elapsed_time += end - start
        if (epoch % print_every) == 0 or (epoch == epochs - 1):
            print(step)
            test_accuracy = evaluate(model, state, testloader)
            print(
                f"{epoch=}",
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_accuracy={test_accuracy.item()}, "
                f"time={elapsed_time}, "
            )
            elapsed_time = 0.
    return model


if __name__ == '__main__':
    # images are padded to have shape 29x29.
    # this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
    pad = Pad((0, 0, 1, 1), fill=0)

    # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
    # we upsample an image by a factor of 3, rotate it and finally downsample it again
    resize1 = Resize(87)
    resize2 = Resize(29)

    totensor = ToTensor()


    train_transform = Compose([
        pad,
        resize1,
        RandomRotation(180., interpolation=InterpolationMode.BILINEAR, expand=False),
        resize2,
        totensor,
    ])

    mnist_train = MnistRotDataset(mode='train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, drop_last=True)

    test_transform = Compose([
        pad,
        totensor,
    ])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, drop_last=True)

    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key, 2)

    elapsed_time = time.time()
    model = C8SteerableCNN(subkey)
    elapsed_time = time.time() - elapsed_time
    print(f"{elapsed_time=}")

    state = eqx.nn.State(model)
    optim = optax.adamw(learning_rate=5e-5, weight_decay=1e-5)
    # optim = optax.adamw(learning_rate=1e-3, weight_decay=1e-5)

    # steps = 30 * len(train_loader)
    # print(len(train_loader))
    # steps = 100 * len(train_loader)
    # print_every = 5 * len(train_loader)
    steps = 31
    print_every = 5
    model = train(model, state, train_loader, test_loader, optim, steps, print_every)
