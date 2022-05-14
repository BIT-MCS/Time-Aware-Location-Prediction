from weeplaces.step2model.utils import *
from weeplaces.step2model.utils import *

def get_init_hidden(batch_size, cell_size, device):
    h = torch.zeros(batch_size, cell_size).to(device)
    c = torch.zeros(batch_size, cell_size).to(device)
    return h, c


class TimeLSTMCell(nn.Module):
    def __init__(self, input_size, cell_size):
        super(TimeLSTMCell, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('linear'))
        self.input_size = input_size
        self.cell_size = cell_size

        self.Wxi = nn.Sequential(
            init_(nn.Linear(
                in_features=self.input_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )
        self.Whi = nn.Sequential(
            init_(nn.Linear(
                in_features=self.cell_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )

        self.Wxf = nn.Sequential(
            init_(nn.Linear(
                in_features=self.input_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )
        self.Whf = nn.Sequential(
            init_(nn.Linear(
                in_features=self.cell_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )

        self.Wxg = nn.Sequential(
            init_(nn.Linear(
                in_features=self.input_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )
        self.Whg = nn.Sequential(
            init_(nn.Linear(
                in_features=self.cell_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )

        self.Wxo = nn.Sequential(
            init_(nn.Linear(
                in_features=self.input_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )
        self.Who = nn.Sequential(
            init_(nn.Linear(
                in_features=self.cell_size,
                out_features=self.cell_size,
            )),
            # nn.LayerNorm([self.cell_size]),
        )

    def forward(self, inputs, hx):
        if hx is None:
            h = torch.zeros([inputs.shape[0], self.cell_size], device=inputs.device, dtype=torch.float32)
            c = torch.zeros([inputs.shape[0], self.cell_size], device=inputs.device, dtype=torch.float32)
            hx = (h, c)

        h, c = hx
        i = torch.sigmoid(self.Wxi(inputs) + self.Whi(h))
        f = torch.sigmoid(self.Wxf(inputs) + self.Whf(h))
        g = torch.sigmoid(self.Wxg(inputs) + self.Whg(h))
        o = torch.sigmoid(self.Wxo(inputs) + self.Who(h))

        new_c = c * f + i * g
        new_h = torch.tanh(new_c) * o  # double sigmoid for output gate
        return new_h, new_c
