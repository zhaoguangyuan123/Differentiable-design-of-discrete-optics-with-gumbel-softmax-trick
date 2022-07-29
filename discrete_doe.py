
from config import *
import math
from utils.visualize_utils import show


class DOE(nn.Module):
    """DOE Module"""
    def __init__(self, doe_size, doe_level) -> None:
        super(DOE, self).__init__()
        self.logits = nn.parameter.Parameter(
            torch.rand(doe_size, doe_size, doe_level))
        self.doe_level = doe_level
        self.level_logits = torch.arange(0, self.doe_level).to(device)

    def logits_to_doe_profile(self):
        _, doe_res = self.logits.max(dim=-1)
        print('doe_res', doe_res.shape)
        return doe_res

    def doe_levels_to_phase(self, doe_instance):
        phase_step = 2*math.pi/self.doe_level
        doe_phase = doe_instance*phase_step
        return doe_phase

    def get_doe_sample(self):
        # Sample soft categorical using reparametrization trick:
        sample_one_hot = F.gumbel_softmax(self.logits, tau=1, hard=False)
        doe_sample = (sample_one_hot *
                      self.level_logits[None, None, :]).sum(dim=-1)
        doe_sample = doe_sample[None, None, :, :]
        return doe_sample
    
    def forward(self):
        phase = self.doe_levels_to_phase(self.get_doe_sample())
        return phase


def __main__():
    doe = DOE(doe_size=20, doe_level=16)
    # doe.get_doe_sample()
    doe_res = doe.logits_to_doe_profile()
    print('doe res',doe_res.shape)
    # show()
