import numpy as np
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import *
# import cv2
import random
import copy
import datetime

np.random.seed(1)
random.seed(1)


class DataLoader:
    def __init__(
            self,
            name,

            lon_min,
            lon_max,

            lat_min,
            lat_max,

            cell_size,

            daytime_inter,

            seq_len,

            root_path,

            train_ratio,

            min_ckin_cell_per_frame,

            version,

            is_delete,
    ):
        self.name = name

        self.lon_min = lon_min
        self.lon_max = lon_max

        self.lat_min = lat_min
        self.lat_max = lat_max

        self.cell_size = cell_size

        self.daytime_inter = daytime_inter

        self.seq_len = seq_len

        self.root_path = os.path.join(root_path, name)

        self.train_ratio = train_ratio

        self.min_ckin_cell_per_frame = min_ckin_cell_per_frame

        self.version = version

        self.is_delete = is_delete
        # ---------------------------------------------

        self.lon_step = (lon_max - lon_min) / self.cell_size
        self.lat_step = (lat_max - lat_min) / self.cell_size

        self.save_path = os.path.join(self.root_path, 'person_' + name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # ---------------------------------------------

        self.load_data()

        self.gen_uid2fid_dic()

        self.min_timestamp = None
        self.max_timestamp = None

    def load_data(self):
        pid2ll_path = os.path.join(self.root_path, 'npySet/partNpySet/p_poi_id2poi_ll.npy')
        self.pid2ll = np.load(pid2ll_path, allow_pickle=True)[()]

        if self.is_delete:
            uid2ckin_path = os.path.join(self.root_path,
                                         'npySet/partNpySet/first_delete_p_user_id2user_checkins_dict.npy')
        else:
            uid2ckin_path = os.path.join(self.root_path, 'npySet/partNpySet/p_user_id2user_checkins.npy')

        self.uid2ckin = np.load(uid2ckin_path, allow_pickle=True)[()]

        uid2unm_path = os.path.join(self.root_path, 'npySet/partNpySet/p_user_id2user_name.npy')
        self.uid2unm = np.load(uid2unm_path, allow_pickle=True)[()]

        unm2uid_path = os.path.join(self.root_path, 'npySet/partNpySet/p_user_name2user_id.npy')
        self.unm2uid = np.load(unm2uid_path, allow_pickle=True)[()]

        unm2fnm_path = os.path.join(self.root_path, 'npySet/user_name2friend_name.npy')
        self.unm2fnm = np.load(unm2fnm_path, allow_pickle=True)[()]

    def gen_uid2fid_dic(self):
        self.uid2fid = {}
        for uid in self.uid2unm.keys():
            fid_list = []
            unm = self.uid2unm[uid]
            fnm_list = self.unm2fnm[unm]
            for fnm in fnm_list:
                if fnm in self.unm2uid.keys():
                    fid_list.append(self.unm2uid[fnm])
            self.uid2fid[uid] = fid_list
        return

    def tm2dtm(self, str_tm):
        return datetime.datetime.strptime(str_tm, '%Y-%m-%d %H:%M:%S')

    def set_max_min_timestamp(self, timestamp):
        if self.max_timestamp is None:
            self.max_timestamp = timestamp
            self.min_timestamp = timestamp
        self.max_timestamp = max(self.max_timestamp, timestamp)
        self.min_timestamp = min(self.min_timestamp, timestamp)

    def pid2xy(self, pid):
        idx_x_float = (float(self.pid2ll[pid][0]) - self.lon_min) / self.lon_step
        idx_y_float = (float(self.pid2ll[pid][1]) - self.lat_min) / self.lat_step
        x = min(int(idx_x_float), self.cell_size - 1)
        y = min(int(idx_y_float), self.cell_size - 1)
        return x, y

    def get_channel_by_hour(self, hour):
        for idx, h_r in enumerate(self.daytime_inter):
            if h_r[0] <= hour < h_r[1]:
                return idx

    def is_the_same_day(self, cur_tm, pre_tm):
        return cur_tm.year == pre_tm.year and \
               cur_tm.month == pre_tm.month and \
               cur_tm.day == pre_tm.day

    def is_adjacent_day(self, cur_tm, pre_tm):
        pre_tm = pre_tm + datetime.timedelta(days=1)
        return self.is_the_same_day(cur_tm, pre_tm)

    def gen_a_seq_by_usr_id(self, usr_id):

        """
        :param usr_id: int
        :return: seq->
        seq = []  # [(frm_dict,frm_ckin_list),...]
            frm_dict = {}  # {(c,x,y):value,...}
            frm_ckin_list = []  # [ckin,....]
        1. delete the frame which has no enough checkin

        """

        # declare a list for usr seq
        frm_dict = {}  # {(c,x,y):value,...}
        frm_ckin_list = []  # [ckin,....]
        seq = []  # [({(c,x,y):value,...},frm_ckin_list),...]

        # --------------------------

        # declare a list for recording checkins for each frame
        # ---------------------------

        usr_ckin_list = self.uid2ckin[usr_id]

        pre_tm = self.tm2dtm(usr_ckin_list[0]['local_time'])
        for idx, ckin in enumerate(usr_ckin_list):

            self.set_max_min_timestamp(ckin['timestamp'])

            cur_tm = self.tm2dtm(ckin['local_time'])

            # if cur_tm changes, add frame to seq and clear frame
            if not self.is_the_same_day(cur_tm, pre_tm):
                # add frame to seq
                if len(frm_ckin_list) >= self.min_ckin_cell_per_frame:
                    seq.append((frm_dict, frm_ckin_list))
                # ------------------

                # clear frame
                frm_dict = {}
                frm_ckin_list = []
                # -----------------------

            # update frame dict
            pid = ckin['pid']
            x, y = self.pid2xy(pid)
            c = self.get_channel_by_hour(cur_tm.hour)
            if not (c, x, y) in frm_dict:
                frm_dict[(c, x, y)] = 0.
            frm_dict[(c, x, y)] += 1.
            # ----------------------

            # update frame ckin list
            frm_ckin_list.append(ckin)
            # -------------------------

            # process the last frame
            if idx == len(usr_ckin_list) - 1 and len(frm_ckin_list) >= self.min_ckin_cell_per_frame:
                seq.append((frm_dict, frm_ckin_list))
            # ----------------------------------------

            pre_tm = cur_tm

        if len(seq) == 0:
            return None
        return seq

    def gen_all_people_seq(self):
        # valid seq counter
        val_frm_counter = 0
        # -------------------------

        self.left_usr_id = []

        self.usr_seq = {}
        for usr_id in self.uid2ckin.keys():
            seq = self.gen_a_seq_by_usr_id(usr_id)
            if seq is not None:
                self.usr_seq[usr_id] = seq
                self.left_usr_id.append(usr_id)
                val_frm_counter += len(seq)
        # ----------------------------------------------
        print('valid frame num:', val_frm_counter, 'total user num:', len(self.uid2ckin.keys()))

    def get_new_cell(self):
        return [set() for i in range(len(self.daytime_inter))]

    def update_cell_view_from_frm_dict(self, cell_view, frm_dict):
        for f_k in frm_dict.keys():
            cell_view[f_k[0]].add((f_k[1], f_k[2]))
        return cell_view

    def update_cell_view_from_seq(self, cell_view, seq):
        for (frm_dict, _, _) in seq:
            cell_view = self.update_cell_view_from_frm_dict(cell_view, frm_dict)
        return cell_view

    def get_cell_view_res(self, cell_view):
        res = []
        for channel_cell in cell_view:
            res.append(len(channel_cell))
        return res

    def gen_contin_train_and_test(self):

        bound_timestamp = int(self.min_timestamp + (self.max_timestamp - self.min_timestamp) * self.train_ratio)

        print('min timestamp:', self.min_timestamp, 'bound timestamp:', bound_timestamp, 'max timestamp:',
              self.max_timestamp)

        self.contin_train_seq = []
        self.contin_test_seq = []
        self.valid_uid_set = set()
        for uid in self.usr_seq:
            pre_tm = None
            pre_frm_dict = None

            seq = []
            # print('#############################')

            cell_of_long_view = self.get_new_cell()

            for idx, (frm_dict, ckin_list) in enumerate(self.usr_seq[uid]):
                if pre_tm is None:
                    pre_tm = self.tm2dtm(ckin_list[0]['local_time'])
                    pre_frm_dict = frm_dict

                cur_tm = self.tm2dtm(ckin_list[0]['local_time'])

                cur_tmstamp = ckin_list[-1]['timestamp']

                if len(seq) == self.seq_len:
                    if len(seq) == self.seq_len:

                        self.valid_uid_set.add(uid)

                        long_res = self.get_cell_view_res(cell_of_long_view)

                        cell_of_short_view = self.update_cell_view_from_seq(self.get_new_cell(), seq[:-1])
                        short_res = self.get_cell_view_res(cell_of_short_view)

                        if cur_tmstamp <= bound_timestamp:
                            self.contin_train_seq.append((seq, long_res, short_res))
                        else:
                            self.contin_test_seq.append((seq, long_res, short_res))
                        seq = seq[1:]

                    else:
                        seq = []

                seq.append((frm_dict, ckin_list, uid))
                cell_of_long_view = self.update_cell_view_from_frm_dict(cell_of_long_view, pre_frm_dict)

                if idx == len(self.usr_seq[uid]) - 1 and \
                        len(seq) == self.seq_len:

                    self.valid_uid_set.add(uid)

                    long_res = self.get_cell_view_res(cell_of_long_view)

                    cell_of_short_view = self.update_cell_view_from_seq(self.get_new_cell(), seq[:-1])
                    short_res = self.get_cell_view_res(cell_of_short_view)

                    if cur_tmstamp <= bound_timestamp:
                        self.contin_train_seq.append((seq, long_res, short_res))
                    else:
                        self.contin_test_seq.append((seq, long_res, short_res))
                    seq = []

                pre_tm = cur_tm
                pre_frm_dict = frm_dict

        print('train seq number.:', len(self.contin_train_seq), 'test seq number.:', len(self.contin_test_seq))
        print('valid uid number:', len(self.valid_uid_set))

    def save_train_and_test_seq(self):
        # save two dataset
        train_seq_save_path = os.path.join(self.save_path,
                                           'train_split_by_time_' + self.name + str(
                                               self.cell_size) + self.version + '.npy')
        test_seq_save_path = os.path.join(self.save_path,
                                          'test_split_by_time_' + self.name + str(
                                              self.cell_size) + self.version + '.npy')
        np.save(train_seq_save_path, self.contin_train_seq)
        np.save(test_seq_save_path, self.contin_test_seq)
        print(train_seq_save_path, test_seq_save_path)
        print(len(self.contin_train_seq), len(self.contin_test_seq))
