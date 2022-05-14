from weeplaces.step2model.lma_l_50.memory import *
from weeplaces.step2model.lma_l_50.time_att import *
from weeplaces.step2model.utils import *


class SequentialModel(nn.Module):
    def __init__(self, poi_num, frame_size, seq_len, input_size=128, cell_size=128, device='cuda:0'):
        super(SequentialModel, self).__init__()
        self.input_size = input_size
        self.cell_size = cell_size
        self.frame_size = frame_size
        self.device = device
        self.rnn = MemoryCell(
            input_size=self.input_size,
            frame_size=self.frame_size,
            cell_size=self.cell_size,
            seq_len=seq_len,  # for debug
        )
        self.att = TimeAwareAtt(
            dimensions=48,
            attention_type='general'
        )
        self.ouput_trans = nn.Sequential(
            # nn.LayerNorm([self.cell_size]),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(
                in_features=self.cell_size * 2,
                out_features=int(poi_num / 2),
            ),
            # nn.LayerNorm([int(poi_num/2)]),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(
                in_features=int(poi_num / 2),
                out_features=poi_num,
            ),
        )
        # self.drop_out = nn.Dropout(0.1)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, train_input_batch, train_help_input_batch, train_loc_batch):
        B, T, input_size = train_input_batch.shape
        m = None
        h = None

        h_list = []
        mem_score_list = []
        for t in range(T):
            memory_out, m, mem_score, h = self.rnn(
                x=train_input_batch[:, t],
                loc=train_loc_batch[:, t],
                m=m,
                h=h,
            )
            h_list.append(memory_out.unsqueeze(dim=1))
            mem_score_list.append(mem_score)
        mem_score = np.concatenate(mem_score_list, axis=1)
        seq_h = torch.cat(h_list, dim=1)
        # [batch size, history size, hidden size]
        flat_seq_h = seq_h.view(-1, seq_h.shape[-1])
        mix_h_list = []
        for t in range(T):
            mix_h, att_weight = self.att(
                target_time=train_help_input_batch[:, t + 1],
                history_time=train_help_input_batch[:, :t + 1],
                history_h=seq_h[:, :t + 1]
            )
            mix_h_list.append(mix_h.unsqueeze(dim=1))

        att_weight_final = att_weight

        seq_mix_h = torch.cat(mix_h_list, dim=1)
        flat_seq_mix_h = seq_mix_h.view(-1, seq_mix_h.shape[-1])
        # flat_output = self.ouput_trans(flat_seq_h * 0.8 + 0.2 * flat_seq_mix_h)# 0.1322
        # flat_output = self.ouput_trans(flat_seq_h + flat_seq_mix_h) # 0.1386
        # flat_output = self.ouput_trans(flat_seq_h * 0.5 + 0.5 * flat_seq_mix_h) # 0.1393
        # flat_output = self.ouput_trans(
        #     self.drop_out(flat_seq_h) + self.drop_out(flat_seq_mix_h))
        flat_output = self.ouput_trans(
            torch.cat([self.drop_out(flat_seq_h), self.drop_out(flat_seq_mix_h)], dim=1))

        # [B*T,poi_num]

        output = flat_output.view(B, T, -1)
        # [B,T,poi_num]

        return output, mem_score
