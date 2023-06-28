import time

from escnn import gspaces
from escnn import group
from escnn import nn

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode

import numpy as np

from PIL import Image


# # download the dataset
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


class C8SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(C8SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("conv 1")
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("block 2")
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        print("pool 1")
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("block 3")
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("block 4")
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        print("pool 2")
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("block 5")
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        print("block 6")
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        print("pool 3")
        
        self.gpool = nn.GroupPooling(out_type)
        print("gool")
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
        print("Sequential")
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x


def compute_accuracy(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = model(x)
    _, pred_y = torch.max(pred_y.data, 1)
    return torch.mean(y == pred_y, dtype=torch.float)


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    model.eval()
    with torch.no_grad():
        avg_acc = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            avg_acc += compute_accuracy(model, x, y)
    return avg_acc / len(test_loader)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    epochs: int,
    print_every: int,
    device: str = "cpu"
):

    loss_function = torch.nn.CrossEntropyLoss()
   
    elapsed_time = 0.
    for epoch in range(epochs):
        start = time.time()
        model.train()

        for step, (x, y) in enumerate(train_loader):
            
            optim.zero_grad()

            x = x.to(device)
            y = y.to(device)

            pred_y = model(x)

            train_loss = loss_function(pred_y, y)

            train_loss.backward()

            optim.step()
        
        end = time.time()
        elapsed_time += end - start

        if (epoch % print_every) == 0 or (epoch == epochs - 1):
            test_accuracy = evaluate(model, test_loader)
            print(
                f"{epoch=}",
                f"{step=}, train_loss={train_loss.item():.2f}, "
                f"test_accuracy={100*test_accuracy.item():.1f}%, "
                f"time={elapsed_time:.1f}s"
            )
            elapsed_time = 0.
                
            # print(f"epoch {epoch} | test accuracy: {correct/total*100.}")


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    elapsed_time = time.time()
    model = C8SteerableCNN().to(device)
    elapsed_time = time.time() - elapsed_time
    print(f"Time for initialising model: {elapsed_time:.1f}s")

    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    steps = 31
    print_every = 5
    elapsed_time = time.time()
    model = train(model, train_loader, test_loader, optim, steps, print_every, device=device)
    elapsed_time = time.time() - elapsed_time
    print(f"Total time for training and evaluating model: {elapsed_time:.1f}s")

