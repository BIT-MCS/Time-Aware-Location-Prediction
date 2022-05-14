from weeplaces.step2model.utils import *
from weeplaces.step2model.conf import *
from weeplaces.step2model.lma_l_50.model_param import *
import os.path

dataset_version = ''
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


train_input = np.zeros([train_seq_counter, seq_len, embedding_size])
train_help_input = np.zeros([train_seq_counter, seq_len, 48])
train_right_answer = np.zeros([train_seq_counter, seq_len])
train_loc = np.zeros([train_seq_counter, seq_len], dtype=np.int)

test_input = np.zeros([test_seq_counter, seq_len, embedding_size])
test_help_input = np.zeros([test_seq_counter, seq_len, 48])
test_right_answer = np.zeros([test_seq_counter, seq_len])
test_loc = np.zeros([test_seq_counter, seq_len], dtype=np.int)
user_appear = np.zeros([test_seq_counter], dtype=np.int)

poi_id_set = np.load('../small_poi_id_set.npy', allow_pickle=True)
poi_id_set.sort()
print(poi_id_set[101:111])
poi_num = len(poi_id_set)
print(poi_num)

poi_idx2ebd128 = nn.Embedding(poi_num, 128)
poi_id2ebd128 = {}

for i, poi_id in enumerate(poi_id_set):
    poi_id2ebd128[poi_id] = poi_idx2ebd128(torch.tensor(i)).detach().numpy()

np.save('../poi_id2ebd128.npy', poi_id2ebd128)

hour_idx2ebd48 = nn.Embedding(total_hour_id, 48)
hour2ebd48 = {}

for i in range(total_hour_id):
    hour2ebd48[i] = hour_idx2ebd48(torch.tensor(i)).detach().numpy()

np.save('../hour2ebd48.npy', hour2ebd48)


def findHour_idx(datetime_str):
    temp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    hour = temp.hour
    minute = temp.minute
    second = temp.second
    hour_float = hour + minute / 60. + second / 3600.
    step = 24 / total_hour_id
    hour_idx = hour_float // step
    return hour_idx


train_seq_idx = 0
for uid in train_uid2seq_list:
    for seq in train_uid2seq_list[uid]:
        for i, ckin_id in enumerate(seq):
            pid = uid2ckin[uid][ckin_id][-1]
            poi_id_ebd = poi_id2ebd128[pid]
            hour_ebd = hour2ebd48[findHour_idx(uid2ckin[uid][ckin_id][1])]
            train_input[train_seq_idx, i] = poi_id_ebd
            train_help_input[train_seq_idx, i] = hour_ebd
            train_right_answer[train_seq_idx, i] = pid
            train_loc[train_seq_idx, i] = xy2idx(pid2xy(pid))
        train_seq_idx += 1

train_input = train_input[:, :-1]
train_right_answer = train_right_answer[:, 1:]
train_loc = train_loc[:, :-1]

print('train_seq_idx:', train_seq_idx)

test_seq_idx = 0
for uid in test_uid2seq_list:
    for seq in test_uid2seq_list[uid]:
        for i, ckin_id in enumerate(seq):
            pid = uid2ckin[uid][ckin_id][-1]
            poi_id_ebd = poi_id2ebd128[pid]
            hour_ebd = hour2ebd48[findHour_idx(uid2ckin[uid][ckin_id][1])]
            test_input[test_seq_idx, i] = poi_id_ebd
            test_help_input[test_seq_idx, i] = hour_ebd
            test_right_answer[test_seq_idx, i] = pid
            test_loc[test_seq_idx, i] = xy2idx(pid2xy(pid))
            if uid in train_uid2seq_list.keys():
                user_appear[test_seq_idx] = 1
        test_seq_idx += 1

test_input = test_input[:, :-1]
test_right_answer = test_right_answer[:, 1:]
test_loc = test_loc[:, :-1]

print('test_seq_idx:', test_seq_idx)

np.save('train_input_' + dataset_version + '.npy', train_input)
np.save('train_help_input_' + dataset_version + '.npy', train_help_input)
np.save('train_right_answer_' + dataset_version + '.npy', train_right_answer)
np.save('train_loc_' + dataset_version + '_' + str(frame_size) + '.npy', train_loc)

np.save('test_input_' + dataset_version + '.npy', test_input)
np.save('test_help_input_' + dataset_version + '.npy', test_help_input)
np.save('test_right_answer_' + dataset_version + '.npy', test_right_answer)
np.save('test_loc_' + dataset_version + '_' + str(frame_size) + '.npy', test_loc)
np.save('user_appear.npy', user_appear)
