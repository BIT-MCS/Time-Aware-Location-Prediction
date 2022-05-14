from weeplaces.step1model.utils import *
import datetime


class Dataset:
    def __init__(
            self,
            train_data_path,
            test_data_path,
            uid2ckin_path,
            pid2ll_path,
            valid_pid_path,
            train_batch_size,
            test_batch_size,
            device,
            seq_len,
            frame_size,
            times,
            gamma,
            input_channel,
            daytime_inter,

            # new_pid2pid_path=None,
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.uid2ckin_path = uid2ckin_path
        self.pid2ll_path = pid2ll_path
        self.valid_pid_path = valid_pid_path

        print(train_data_path)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.times = times

        self.gamma = gamma

        # self.new_pid2pid_path = new_pid2pid_path
        self.input_channel = input_channel
        self.daytime_inter = daytime_inter
        self.load_data()

        self.test_counter = 0

    def get_seqs(self, data):
        seq_frame_list = []
        seq_ckin_list = []
        seq_uid_list = []
        seq_cell_view_list = []

        for seq in data:
            temp_seq = []
            temp_ckin = []
            temp_uid = []
            for trp in seq[0]:
                temp_seq.append(trp[0])
                temp_ckin.append(trp[1])
                temp_uid.append(trp[2])
            seq_frame_list.append(temp_seq)
            seq_ckin_list.append(temp_ckin)
            seq_uid_list.append(temp_uid)
            seq_cell_view_list.append(seq[1:3])
            # seq_wk_frame_list.append(seq[3])
        return np.array(seq_frame_list), seq_ckin_list, seq_uid_list, seq_cell_view_list

    def load_data(self):

        self.train_data = np.load(self.train_data_path, allow_pickle=True)
        self.test_data = np.load(self.test_data_path, allow_pickle=True)

        self.train_seq_frame, self.train_seq_ckin, _, _ = self.get_seqs(self.train_data)
        self.test_seq_frame, self.test_seq_ckin, self.test_seq_uid, self.test_seq_cell_view= self.get_seqs(
            self.test_data)

        self.uid2ckin = np.load(self.uid2ckin_path, allow_pickle=True)[()]
        self.pid2ll = np.load(self.pid2ll_path, allow_pickle=True)[()]
        # if self.new_pid2pid_path is not None:
        #     self.new_pid2pid = np.load(self.new_pid2pid_path, allow_pickle=True)[()]

        self.valid_pid = np.load(self.valid_pid_path, allow_pickle=True)
        self.train_data_len = len(self.train_seq_frame)
        self.test_data_len = len(self.test_seq_frame)

        print('train data len: %d, test data len %d' % (self.train_data_len, self.test_data_len))

    def get_np_batch_from_dict_batch(self, batch_dict_list):
        batch_size = batch_dict_list.shape[0]
        seq_len = batch_dict_list.shape[1]
        batch_np = np.zeros(shape=[batch_size, seq_len, self.input_channel, self.frame_size, self.frame_size],
                            dtype=np.float32)

        for batch in range(batch_size):
            for seq in range(seq_len):
                frm_dict = batch_dict_list[batch, seq]
                for key in frm_dict:
                    if self.gamma == 0:
                        # batch_np[batch, seq, key[0], key[1], key[2]] = frm_dict[key] * self.times
                        # pass
                        batch_np[batch, seq, key[0], key[1], key[2]] = 1 * self.times
        return batch_np

    def get_train_batch(self):
        indices = np.random.choice(self.train_data_len, self.train_batch_size)

        train_batch_dict_list = self.train_seq_frame[indices]
        train_batch_np = self.get_np_batch_from_dict_batch(train_batch_dict_list)
        train_batch = torch.tensor(train_batch_np, dtype=torch.float32, device=self.device)

        # train_batch_wk_dict_list = self.train_wk_frame[indices]
        # train_batch_wk_np = self.get_np_batch_from_dict_batch(train_batch_wk_dict_list)
        # train_batch_wk = torch.tensor(train_batch_wk_np, dtype=torch.float32, device=self.device)

        return train_batch

    def get_last_frame_label(self, test_seq_batch_dict_list):
        batch = len(test_seq_batch_dict_list)
        last_frame_label = []
        for b in range(batch):
            last_frame_label.append(list(test_seq_batch_dict_list[b][-1].keys()))
        return last_frame_label

    def get_test_batch(self):
        if self.test_counter < self.test_data_len:
            test_seq_batch_dict_list = self.test_seq_frame[self.test_counter:self.test_counter + self.test_batch_size]
            test_seq_batch_np = self.get_np_batch_from_dict_batch(test_seq_batch_dict_list)
            test_seq_batch = torch.tensor(test_seq_batch_np, dtype=torch.float32, device=self.device)
            # test_last_frame_checkin_batch = self.get_last_frame_label(test_seq_batch_dict_list)
            test_ckeckin_batch = self.test_seq_ckin[
                                 self.test_counter:self.test_counter + self.test_batch_size]

            test_uid_batch = self.test_seq_uid[
                             self.test_counter:self.test_counter + self.test_batch_size]

            test_cell_view_batch = self.test_seq_cell_view[
                                   self.test_counter:self.test_counter + self.test_batch_size]

            # test_batch_wk_dict_list = self.test_wk_frame[self.test_counter:self.test_counter + self.test_batch_size]
            # test_batch_wk_np = self.get_np_batch_from_dict_batch(test_batch_wk_dict_list)
            # test_batch_wk = torch.tensor(test_batch_wk_np, dtype=torch.float32, device=self.device)

            self.test_counter += self.test_batch_size
            return test_seq_batch, test_seq_batch_dict_list, test_ckeckin_batch, test_uid_batch, test_cell_view_batch

        else:
            self.test_counter = 0
            return None, None, None, None, None

    def tm2dtm(self, str_tm):
        return datetime.datetime.strptime(str_tm, '%Y-%m-%d %H:%M:%S')

    def get_ckin_list_by_channel(self, ckins, channel):
        hst = self.daytime_inter[channel][0]
        hed = self.daytime_inter[channel][1]
        channel_ckin_list = []
        for ckin in ckins:
            dtm = self.tm2dtm(ckin['local_time'])
            if hst <= dtm.hour < hed:
                channel_ckin_list.append(ckin)
        return channel_ckin_list
