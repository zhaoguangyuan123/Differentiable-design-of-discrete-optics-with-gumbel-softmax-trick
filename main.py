# %%
from config import *
from utils.general_utils import load_mnist_target
from sgd_holography import SGDHolo

target = load_mnist_target()
print('shape of target is: ', target.shape)
# %%
img_size = 64
partition = 32
lr = 1e1
Holo_model = SGDHolo(img_size, partition, lr, target)
Holo_model.train(itrs=200)
