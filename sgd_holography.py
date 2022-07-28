

from config import *
from propagator import FresnelProp
from utils.visualize_utils import plot_loss, show
from utils.general_utils import normalize
from doe import DOE


class TestOptics(nn.Module):
    def __init__(self, input_dx, input_field_shape, output_dx, output_field_shape, wave_lengths, z, response_type, pad_scale, slm_size, num_partition, doe_level):
        super().__init__()
        self.doe = DOE(doe_size=num_partition, doe_level=doe_level)
        self.propagate_fresnel = FresnelProp(
            input_dx, input_field_shape, output_dx, output_field_shape, wave_lengths, z, response_type, pad_scale)
        self.m = nn.Upsample(scale_factor=slm_size /
                             num_partition, mode='nearest')

    def forward(self):
        phase = self.doe.doe_levels_to_phase(self.doe.get_doe_sample())
        phase = self.m(phase)
        x = torch.exp(1j*phase)
        x = self.propagate_fresnel(x)
        return x


class SGDHolo(object):
    def __init__(self, in_size, partition, lr, target) -> None:
        super().__init__()

        self.testoptics = TestOptics(input_dx=4.8, input_field_shape=[in_size, in_size], output_dx=4.8, output_field_shape=[
                                     in_size, in_size], wave_lengths=0.633, z=1e4, response_type=None, pad_scale=2, slm_size=in_size, num_partition=partition, doe_level=4).to(device)

        self.phase_optimizer = torch.optim.SGD(
            self.testoptics.parameters(), lr=lr)
        self.photometric_loss_fn = nn.MSELoss()
        self.target = target

    def train(self, itrs):
        losses = []
        itr_list = []
        for itr in range(itrs):
            out_field = self.testoptics()
            out_amp = normalize(torch.abs(out_field)**2)
            loss = torch.log(self.photometric_loss_fn(out_amp, self.target))
            self.phase_optimizer.zero_grad()
            loss.backward()
            self.phase_optimizer.step()

            losses.append(loss.item())
            itr_list.append(itr)
            if itr % 20 == 0 or itr == (itrs-1):
                plot_loss(itr_list, losses)
                show(out_amp[0, 0].detach().cpu().numpy(),
                     'img at itr {}'.format(itr))
                show(self.testoptics.doe.logits_to_doe_profile().detach(
                ).cpu().numpy(), 'doe at itr {}'.format(itr))
