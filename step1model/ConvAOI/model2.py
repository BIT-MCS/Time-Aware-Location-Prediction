from weeplaces.step1model.att_channel_convlstm_64_v2_c6.conv_lstm3 import *


class PredictiveModel(nn.Module):
    def __init__(self, input_channel, hidden_channels, frame_size, cnn_kernel_size, rnn_kernel_size, device, input_len,
                 seq_len,
                 ):
        super(PredictiveModel, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.frame_size = frame_size
        self.cnn_kernel_size = cnn_kernel_size
        self.rnn_kernel_size = rnn_kernel_size
        self.device = device
        self.input_len = input_len
        self.seq_len = seq_len

        self.cnn_padding = int((self.cnn_kernel_size - 1) / 2)
        self.rnn_padding = int((self.rnn_kernel_size - 1) / 2)

        # def init_(m):
        #     init(m, nn.init.orthogonal_, nn.init.calculate_gain('conv2d'))
        init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('conv2d'))
        # init_relu = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('relu'))
        # init_tanh = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('tanh'))

        self.cnn_layer = nn.Sequential(
            init_(nn.Conv2d(
                in_channels=6,
                out_channels=32,
                kernel_size=self.cnn_kernel_size,
                stride=2,
            )),

            nn.LayerNorm([32, 31, 31]),
            nn.ReLU(),
            # nn.Dropout(0.5),

            init_(nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=1,
            )),
            nn.LayerNorm([64, 29, 29]),
            nn.ReLU(),
            # nn.Dropout(0.5),

            init_(nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=2,
            )),

            nn.LayerNorm([64, 14, 14]),
            nn.ReLU(),
            # nn.Dropout(0.5),

            init_(nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
            )),
            nn.LayerNorm([64, 14, 14]),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        self.hidden_size = (14, 14)
        self.rnn_layers = ConvLSTMLayers(input_channel=64, hidden_channels=self.hidden_channels,
                                         hidden_size=self.hidden_size,
                                         kernel_size=self.rnn_kernel_size, )

        self.reverse_layer = nn.Sequential(
            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
            )),

            nn.LayerNorm([64, 14, 14]),
            nn.ReLU(),
            # nn.Dropout(0.5),

            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=2,
            )),
            nn.LayerNorm([64, 29, 29]),
            nn.ReLU(),
            # nn.Dropout(0.5),
            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=1,
            )),
            nn.LayerNorm([64, 31, 31]),
            nn.ReLU(),
            # nn.Dropout(0.5),
            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=6,
                kernel_size=4,
                stride=2,
            )),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, input_seq):
        """
        :param input_seq: [Batch,Time,Channel,Size_m,Size_n]->[B,T,C,M,N]
        :param is_valid: bool
        :return: [B,T,C,M,N]
        """
        hidden = get_controller_init_hidden(batch_size=input_seq.shape[0], hidden_size=self.hidden_size,
                                            hidden_channels=self.hidden_channels, device=self.device)
        # zero_tensor = torch.zeros_like(input_seq[:, 0, ...], device=self.device).unsqueeze(dim=1)
        # input_seq = torch.cat([input_seq, zero_tensor], dim=1)

        output_batch_list = []
        cnn_out_list = []
        rnn_out_list = []
        for t in range(input_seq.shape[1]):
            # for t in range(5):
            cnn_out = self.cnn_layer(input_seq[:, t, ...])
            cnn_out_list.append(cnn_out.unsqueeze(dim=1))
            _, hidden = self.rnn_layers(cnn_out, hidden)
            rnn_out_list.append(hidden[-1][0].unsqueeze(dim=1))
            output_batch_list.append(self.reverse_layer(hidden[-1][0]).unsqueeze(dim=1))
        output_batch = torch.cat(output_batch_list, dim=1)
        cnn_out = torch.cat(cnn_out_list, dim=1)
        rnn_out = torch.cat(rnn_out_list, dim=1)
        return output_batch, cnn_out, rnn_out
