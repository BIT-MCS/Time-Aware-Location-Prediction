import torch

from weeplaces.step1model.utils import *


def get_controller_init_hidden(batch_size, hidden_channels, hidden_size, device):
    init_hidden_states = []
    for hidden_channel in hidden_channels:
        init_hidden_states.append((torch.zeros(batch_size, hidden_channel, hidden_size[0], hidden_size[1]).to(device),
                                   torch.zeros(batch_size, hidden_channel, hidden_size[0], hidden_size[1]).to(device)))
    return init_hidden_states


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel, hidden_size, kernel_size, bias=True, is_bn=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_channel: int
            Number of channels of input tensor.
        hidden_channel: int
            Number of channels of hidden state.
        hidden_size: (int, int)
            Height and width of input tensor as (height, width).
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = hidden_size
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               # nn.init.xavier_uniform_,

                               nn.init.calculate_gain('conv2d'))
        # if is_bn:
        #     self.conv = nn.Sequential(
        #         init_(nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
        #                         out_channels=4 * self.hidden_channel,
        #                         kernel_size=self.kernel_size,
        #                         padding=self.padding,
        #                         bias=self.bias)),
        #         nn.BatchNorm2d(4 * self.hidden_channel)
        #     )
        # else:
        self.conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
                            out_channels=4 * self.hidden_channel,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)),
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        combined_conv = torch.layer_norm(combined_conv, combined_conv.shape[1:])

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMLayers(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    """
    Initialize ConvLSTM cell.

    Parameters
    ----------
    input_channel: int
        Number of channels of input tensor.
    hidden_channel: int
        Number of channels of hidden state.
    hidden_size: (int, int)
        Height and width of input tensor as (height, width).
    kernel_size: int
        Size of the convolutional kernel.

    """

    def __init__(self, input_channel, hidden_channels, hidden_size, kernel_size):
        super(ConvLSTMLayers, self).__init__()
        print('ConvLSTM3')
        self.input_channels = [input_channel] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self._all_layers = nn.ModuleList()
        self.hidden_size = hidden_size
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            # if i == self.num_layers - 1:
            #     cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
            #                         (self.kernel_size, self.kernel_size), is_bn=False)
            # else:
            #     cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
            #                         (self.kernel_size, self.kernel_size), is_bn=False)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
                                (self.kernel_size, self.kernel_size))
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, layers_hidden_states):
        x = input
        layers_output = []
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            # do forward
            (h, c) = layers_hidden_states[i]
            x, new_c = getattr(self, name)(x, (h, c))
            layers_hidden_states[i] = (x, new_c)
            layers_output.append(x)
        return layers_output, layers_hidden_states
