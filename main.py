# %%
from config import *
from utils.general_utils import load_mnist_target
from trainer_sgd_holography import SGDHoloTrainer

target = load_mnist_target()
print('shape of target is: ', target.shape)
# %%
img_size = 64
partition = 32
lr = 5e1
itrs = 200

Holo_model = SGDHoloTrainer(img_size, partition, lr, target)
Holo_model.train(itrs)
