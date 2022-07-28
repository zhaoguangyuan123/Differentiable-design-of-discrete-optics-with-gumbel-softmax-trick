# %%
import torch
import torch.nn.functional as F
from config import *
from utils.general_utils import circular_pad
from torchvision import datasets, transforms
from sgd_holography import SGDHolo


def load_mnist_target():
    transform = transforms.Compose(
        [transforms.ToTensor(), ])

    trainset = datasets.MNIST('data/mnist_train', train=True,
                              download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=False)

    inputs, labels = next(iter(trainloader))
    inputs = circular_pad(inputs, 32/28)
    inputs = F.interpolate(inputs, scale_factor=2)
    target = inputs[:1, :, :, :].to(device)
    return target


# cond_mkdir('data/')
target = load_mnist_target()
print('shape of target is: ', target.shape)
# %%
img_size = 64
partition = 32
lr = 1e1
Holo_model = SGDHolo(img_size, partition, lr, target)
Holo_model.train(itrs=200)
