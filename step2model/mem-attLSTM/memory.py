from weeplaces.step2model.utils import *
from weeplaces.step2model.lma_l_50.lstm_cell import *
from weeplaces.step2model.utils import *

def get_init_hidden(batch_size, frame_size, cell_size, device):
    m = torch.zeros(batch_size, frame_size * frame_size, cell_size).to(device)
    return m


class MemoryCell(nn.Module):
    def __init__(self, input_size, frame_size, cell_size, seq_len):
        super(MemoryCell, self).__init__()
        # init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('linear'))
        self.input_size = input_size
        self.mem_len = frame_size * frame_size
        self.cell_size = cell_size
        self.seq_len = seq_len
        # init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('linear'))
        init_ = lambda m: init(m, )
        self.w_i_r = nn.Sequential(
            init_(
                nn.Linear(input_size, cell_size),
            ),
            nn.LayerNorm(cell_size),
        )
        self.w_h_r = nn.Sequential(
            init_(
                nn.Linear(cell_size, cell_size)
            ),
            nn.LayerNorm(cell_size),
        )
        self.w_i_z = nn.Sequential(
            init_(
                nn.Linear(input_size, cell_size),
            ),
            nn.LayerNorm(cell_size),

        )
        self.w_h_z = nn.Sequential(
            init_(
                nn.Linear(cell_size, cell_size)
            ),
            nn.LayerNorm(cell_size),
        )

        self.w_i_n = nn.Sequential(
            init_(
                nn.Linear(input_size, cell_size),
            ),
            nn.LayerNorm(cell_size),
        )
        self.w_h_n = nn.Sequential(
            init_(
                nn.Linear(cell_size, cell_size)
            ),
            nn.LayerNorm(cell_size),
        )

        self.x2r_k_layer = nn.Sequential(
            init_(
                nn.Linear(input_size * 2, cell_size, bias=True)
            ),
            # nn.LayerNorm(cell_size),
        )
        self.x_emb_layer = nn.Sequential(
            init_(
                nn.Linear(input_size, cell_size)
            ),
            # nn.LayerNorm(cell_size),
            nn.Tanh(),
        )
        self.out_layer = nn.Sequential(
            init_(
                nn.Linear(cell_size * 3, cell_size)
            ),
            # nn.LayerNorm(cell_size),
            nn.Tanh(),
        )
        self.x2w_k_layer = nn.Sequential(
            init_(
                nn.Linear(input_size * 2, cell_size, bias=True)
            ),
            # nn.LayerNorm(cell_size),
        )
        self.global_rnn = TimeLSTMCell(input_size, cell_size)

    def get_m_batch_from_m_dict_list(self, m_dict_list, device):

        m_batch_np = np.zeros([len(m_dict_list), self.seq_len, self.cell_size])
        mask_np = np.zeros([len(m_dict_list), self.seq_len], dtype=np.long)
        for b, m_dict in enumerate(m_dict_list):
            for t, m_k in enumerate(m_dict):
                m_batch_np[b, t] = m_dict[m_k].detach().cpu().numpy()
                mask_np[b, t] = 1
        m_batch_tensor = torch.tensor(m_batch_np, device=device, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, device=device, dtype=torch.long)

        return m_batch_tensor, mask_tensor

    def read(self, r_k, m_dict_list):
        """
        :param r_k:[b,cell_size]
        :param m: [b,m_len,cell_size]
        :return:
        """
        m, mask = self.get_m_batch_from_m_dict_list(m_dict_list, r_k.device)

        r_k = r_k.unsqueeze(dim=2)
        # [b,cell_size,1]

        score = torch.einsum('bij,bjk->bik', m, torch.tanh(r_k)).squeeze(dim=2)
        # [b,m_len]

        score = score.masked_fill(mask == 0, -np.inf)
        # [b,m_len]

        score_norm = torch.softmax(score, dim=1).unsqueeze(dim=2)
        # [b,m_len,1]

        r = torch.einsum('bij,bjk->bik', m.permute([0, 2, 1]), score_norm).squeeze(dim=2)
        # [b,cell_size]

        # for debug
        m_len = score_norm.shape[1]
        score_np = np.zeros([score_norm.shape[0], 1, self.seq_len])
        score_np[:, 0, :m_len] = score_norm.squeeze(dim=2).detach().cpu().numpy()
        # for debug

        return r, score_np

    def write(self, x, loc, m):
        """
        :param x:[b,x_size]
        :param loc:[b]
        :param m: [b,m_len,cell_size]]
        :return:
        """
        m_batch = []
        for b in range(x.shape[0]):
            if loc[b] not in m[b]:
                m[b][loc[b]] = torch.zeros(self.cell_size).to(x.device)
            m_batch.append(m[b][loc[b]].unsqueeze(dim=0))
        m_batch = torch.cat(m_batch, dim=0)

        gru_r = torch.sigmoid(self.w_i_r(x) + self.w_h_r(m_batch))
        gru_z = torch.sigmoid(self.w_i_z(x) + self.w_h_z(m_batch))
        gru_n = torch.tanh(self.w_i_n(x) + gru_r * self.w_h_n(m_batch))
        m_batch_new = (1 - gru_z) * gru_n + gru_z * m_batch

        # m = m.cpu().detach().numpy()
        for b in range(x.shape[0]):
            # m[b][loc[b]] = torch.tensor(m_batch_new[b].detach().cpu().numpy(), device=x.device)
            m[b][loc[b]] = m_batch_new[b]
        return m

    def forward(self, x, loc, m, h):
        if m is None:
            m = [{} for i in range(x.shape[0])]
            # h = torch.zeros(x.shape[0], self.cell_size).to(x.device)
        h = self.global_rnn(x, h)

        w_k = self.x2w_k_layer(torch.cat([x, h[0]], dim=1))
        m = self.write(w_k, loc, m)

        r_k = self.x2r_k_layer(torch.cat([x, h[0]], dim=1))
        r, score = self.read(r_k, m)

        # out = self.out_layer(torch.cat([r_k, r], dim=1))
        # out = self.out_layer(torch.cat([x, r], dim=1))

        out = self.out_layer(torch.cat([self.x_emb_layer(x), r, h[0]], dim=1))

        return out, m, score, h
