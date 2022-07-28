

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.device_count() > 1:
    device_num = 1
else:
    device_num = 0

device = torch.device("cuda:" + str(device_num)
                      if torch.cuda.is_available() else "cpu")
