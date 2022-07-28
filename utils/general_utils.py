# import math
from config import *


def normalize(x):
    """normalize to range [0-1]"""
    batch_size, num_obj, height, width = x.shape

    x = x.view(batch_size, -1)
    # x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(batch_size, num_obj, height, width)
    return x


def central_crop(variable, tw=None, th=None, dim=2):
    if dim == 2:
        w = variable.shape[-2]
        h = variable.shape[-1]
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., x1: x1 + tw, y1: y1 + th]
    elif dim == 1:
        h = variable.shape[-1]
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., y1: y1 + th]
    else:
        raise NotImplementedError
    return cropped

def circular_pad(u, pad_scale):
    """circular padding last two dimension of a tensor"""
    w, h = u.shape[-2], u.shape[-1]
    w_padded, h_padded = w*pad_scale, h*pad_scale
    ww = int(round((w_padded - w) / 2.0))
    hh = int(round((h_padded - h) / 2.0))
    p2d = (hh, hh, ww, ww)
    u_padded = F.pad(u, p2d, mode="constant", value=0)
    return u_padded


class InterpolateComplex2d(nn.Module):
    def __init__(self, input_dx, input_field_shape, output_dx, output_field_shape, mode='bicubic') -> None:
        super().__init__()
        self.mode = mode
        if output_dx == input_dx:
            print(
                'pitch size of input plane matches the output plane; no interpolation will occur')

        if input_dx * input_field_shape[-2] <= output_dx * output_field_shape[-2]:
            input_pad_scale_x = (
                output_dx * output_field_shape[-2]) / (input_dx * input_field_shape[-2])
        else:
            input_pad_scale_x = 1

        if input_dx * input_field_shape[-1] <= output_dx * output_field_shape[-1]:
            input_pad_scale_y = (
                output_dx * output_field_shape[-1]) / (input_dx * input_field_shape[-1])
        else:
            input_pad_scale_y = 1

        self.input_pad_scale = max(input_pad_scale_y, input_pad_scale_x)

        self.interpolated_input_field_shape = [
            int(input_dx*side_length*self.input_pad_scale/output_dx) for side_length in input_field_shape[-2:]]

        self.output_field_shape = output_field_shape

    def interp_complex(self, x):
        x_in_real_imag = torch.view_as_real(x)  # shape [..., w, h, 2]
        x_real_interpolated = F.interpolate(x_in_real_imag[..., 0], (
            self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode)
        x_imag_interpolated = F.interpolate(x_in_real_imag[..., 1], (
            self.interpolated_input_field_shape[-2], self.interpolated_input_field_shape[-1]), mode=self.mode)
        x_interpolated = torch.stack(
            [x_real_interpolated, x_imag_interpolated], dim=-1)
        x_interpolated = torch.view_as_complex(x_interpolated)
        return x_interpolated

    def match_energy(self, x, x_interpolated):
        energy_before_interpolation = torch.sum(
            torch.abs(x)**2, dim=[-1, -2], keepdim=True)
        energy_after_interpolation = torch.sum(
            torch.abs(x_interpolated)**2, dim=[-1, -2], keepdim=True)
        energy_change_ratio = energy_before_interpolation / energy_after_interpolation
        # print('energy_change_ratio', energy_change_ratio)
        # torch.sqrt convert energy ratio to intensity ratio
        x_interpolate_energy_conserved = x_interpolated * \
            torch.sqrt(energy_change_ratio)
        return x_interpolate_energy_conserved

    def forward(self, x):
        x = circular_pad(x, self.input_pad_scale)
        x_interpolated = self.interp_complex(x)

        variable_interpolate_energy_conserved = self.match_energy(
            x, x_interpolated)

        # central crop to get the desired ouput shape
        output = central_crop(variable_interpolate_energy_conserved,
                              tw=self.output_field_shape[-2], th=self.output_field_shape[-1])

        return output
