from weeplaces.step2model.utils import *
from weeplaces.step2model.conf import *

uid2ckin = np.load(CONF['uid2ckin_path'], allow_pickle=True)[()]
pid2ll = np.load(CONF['pid2ll_path'], allow_pickle=True)[()]
train_seq_len = CONF['train_seq_len']
predict_seq_len = CONF['predict_seq_len']
hour_offset = CONF['hour_offset']
train_uid2label_ckin_id_list = np.load(CONF['train_uid2label_ckin_id_list_path'], allow_pickle=True)[()]
test_uid2label_ckin_id_list = np.load(CONF['test_uid2label_ckin_id_list_path'], allow_pickle=True)[()]

version = '4'

train_uid2seq_list = {}
test_uid2seq_list = {}

for uid in train_uid2label_ckin_id_list:
    ckin_list = uid2ckin[uid]
    for ckin_id in train_uid2label_ckin_id_list[uid]:
        seq_list = []
        counter = 1
        seq_list.append(ckin_id)
        for sub_ckin_id in range(ckin_id - 1, -1, -1):
            seq_list.append(sub_ckin_id)
            counter += 1
            if counter == train_seq_len + predict_seq_len:
                break
        seq_list.reverse()
        if uid not in train_uid2seq_list:
            train_uid2seq_list[uid] = []
        train_uid2seq_list[uid].append(seq_list)

for uid in test_uid2label_ckin_id_list:
    ckin_list = uid2ckin[uid]
    for ckin_id in test_uid2label_ckin_id_list[uid]:
        seq_list = []
        counter = 1
        seq_list.append(ckin_id)
        for sub_ckin_id in range(ckin_id - 1, -1, -1):
            seq_list.append(sub_ckin_id)
            counter += 1
            if counter == train_seq_len + predict_seq_len:
                break
        seq_list.reverse()
        if uid not in test_uid2seq_list:
            test_uid2seq_list[uid] = []
        test_uid2seq_list[uid].append(seq_list)

np.save('train_uid2seq_list_' + version + '.npy', train_uid2seq_list)
np.save('test_uid2seq_list_' + version + '.npy', test_uid2seq_list)
