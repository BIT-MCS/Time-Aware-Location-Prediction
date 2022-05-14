from weeplaces.step2model.utils import *
from weeplaces.step2model.conf import *
from weeplaces.step2model.lma_l_50.model_param import *
import os.path

dataset_version = '4'
embedding_size = CONF['embedding_size']
uid2ckin = np.load(CONF['uid2ckin_path'], allow_pickle=True)[()]
train_uid2seq_list = \
    np.load('../train_uid2seq_list_' + dataset_version + '_extend_' + str(total_seq_len) + '.npy', allow_pickle=True)[
        ()]
test_uid2seq_list = \
np.load('../test_uid2seq_list_' + dataset_version + '_extend_' + str(total_seq_len) + '.npy', allow_pickle=True)[()]

train_uid_set = list(train_uid2seq_list.keys())
test_uid_set = list(test_uid2seq_list.keys())

print('train_user_num:', len(train_uid_set), 'test_user_num:', len(test_uid_set))
seq_len = len(train_uid2seq_list[train_uid_set[0]][0])
train_seq_counter = 0
test_seq_counter = 0

for uid in train_uid_set:
    train_seq_counter += len(train_uid2seq_list[uid])

for uid in test_uid_set:
    test_seq_counter += len(test_uid2seq_list[uid])

print('train_seq_num:', train_seq_counter, 'test_seq_num:', test_seq_counter)

frame_size = CONF['frame_size']
lon_step = (lon_max - lon_min) / frame_size
lat_step = (lat_max - lat_min) / frame_size
pid2ll = np.load(CONF['pid2ll_path'], allow_pickle=True)[()]


def pid2xy(pid):
    idx_x_float = (float(pid2ll[pid][0]) - lon_min) / lon_step
    idx_y_float = (float(pid2ll[pid][1]) - lat_min) / lat_step
    x = min(int(idx_x_float), frame_size - 1)
    y = min(int(idx_y_float), frame_size - 1)
    return x, y


def xy2idx(xy):
    idx = xy[0] * frame_size + xy[1]
    return idx



poi_id_set = np.load('../small_poi_id_set.npy', allow_pickle=True)
poi_id_set.sort()
print(poi_id_set[101:111])
poi_num = len(poi_id_set)
print(poi_num)


uid_ckin_id2res_item_id = {}
result_npy = np.load(CONF['result_npy_path'], allow_pickle=True)
for res_item_id, items in enumerate(result_npy):
    for ckin_id in items['ckin_id_list']:
        uid_ckin_id2res_item_id[(items['uid'], ckin_id)] = res_item_id

test_uid2valid_seq_list = {}
test_seq_num = 0
valid_test_seq_num = 0
for uid in test_uid2seq_list:
    valid_seq_list = []
    for seq in test_uid2seq_list[uid]:
        if (uid, seq[-1]) in uid_ckin_id2res_item_id:
            valid_seq_list.append(seq)
    if len(valid_seq_list) > 0:
        test_uid2valid_seq_list[uid] = valid_seq_list
    test_seq_num += len(test_uid2seq_list[uid])
    valid_test_seq_num += len(valid_seq_list)
print('test_seq_num:', test_seq_num, 'valid_test_seq_num', valid_test_seq_num)


test_uid_ckin_id2history_pid_set = {}
for uid in test_uid2valid_seq_list:
    pre_ckin_id = -1
    history_pid_set = set()
    for idx, seq in enumerate(test_uid2valid_seq_list[uid]):
        ckin_id = seq[-1]
        history_seq = []
        if idx == 0 and uid in train_uid2seq_list:
            for train_seq in train_uid2seq_list[uid]:
                for train_ckin_id in train_seq:
                    train_pid = uid2ckin[uid][train_ckin_id][-1]
                    history_pid_set.add(train_pid)
        for test_seq in test_uid2seq_list[uid]:
            if test_seq[-1] == ckin_id:
                pre_ckin_id = ckin_id
                break
            if pre_ckin_id <= test_seq[-1] < ckin_id:
                for test_ckin_id in test_seq:
                    test_pid = uid2ckin[uid][test_ckin_id][-1]
                    history_pid_set.add(test_pid)

        test_uid_ckin_id2history_pid_set[(uid, ckin_id)] = history_pid_set


# test_input = np.zeros([valid_test_seq_num, seq_len, embedding_size])
# test_help_input = np.zeros([valid_test_seq_num, seq_len, 48])
# test_right_answer = np.zeros([valid_test_seq_num, seq_len])
# test_loc = np.zeros([valid_test_seq_num, seq_len], dtype=np.int)

test_pred_pid_list = []
test_label_pid_list = []
test_long_view = np.zeros([test_seq_num, 1], dtype=np.int)
test_short_view = np.zeros([test_seq_num, 1], dtype=np.int)
test_history_pid_list = []

test_seq_idx = 0
valid_seq_idx = 0
for uid in test_uid2seq_list:
    for seq in test_uid2seq_list[uid]:
        if (uid, seq[-1]) in uid_ckin_id2res_item_id:
            res_item_id = uid_ckin_id2res_item_id[(uid, seq[-1])]
            test_pred_pid_list.append(result_npy[res_item_id]['pred_pid_list'])
            test_label_pid_list.append(result_npy[res_item_id]['label_pid_list'])
            test_long_view[test_seq_idx, 0] = result_npy[res_item_id]['long_view']
            test_short_view[test_seq_idx, 0] = result_npy[res_item_id]['short_view']
            test_history_pid_list.append(test_uid_ckin_id2history_pid_set[(uid, seq[-1])])
            valid_seq_idx += 1
        else:
            test_pred_pid_list.append(None)
            test_label_pid_list.append(None)
            test_long_view[test_seq_idx, 0] = 0
            test_short_view[test_seq_idx, 0] = 0
            test_history_pid_list.append(None)
        test_seq_idx += 1

print('test_seq_idx:', test_seq_idx, 'valid_seq_idx:', valid_seq_idx)

#
# np.save('train_input_' + dataset_version + '.npy', train_input)
# np.save('train_help_input_' + dataset_version + '.npy', train_help_input)
# np.save('train_right_answer_' + dataset_version + '.npy', train_right_answer)
# np.save('train_loc_' + dataset_version + '_' + str(frame_size) + '.npy', train_loc)
#
# np.save('test_input_' + dataset_version + '.npy', test_input)
# np.save('test_help_input_' + dataset_version + '.npy', test_help_input)
# np.save('test_right_answer_' + dataset_version + '.npy', test_right_answer)
# np.save('test_loc_' + dataset_version + '_' + str(frame_size) + '.npy', test_loc)

np.save('test_pred_pid_list' + dataset_version + '.npy', test_pred_pid_list)
np.save('test_label_pid_list' + dataset_version + '.npy', test_label_pid_list)
np.save('test_long_view' + dataset_version + '.npy', test_long_view)
np.save('test_short_view' + dataset_version + '.npy', test_short_view)
np.save('test_history_pid_list' + dataset_version + '.npy', test_history_pid_list)
