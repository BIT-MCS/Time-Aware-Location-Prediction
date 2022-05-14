from weeplaces.step2model.utils import *
from weeplaces.step2model.conf import *
from weeplaces.step2model.utils import *

uid2ckin = np.load(CONF['uid2ckin_path'], allow_pickle=True)[()]
train_seq_len = CONF['train_seq_len']
predict_seq_len = CONF['predict_seq_len']
hour_offset = CONF['hour_offset']
split_timestamp = CONF['split_timestamp']

train_uid2label_ckin_id_list = {}
test_uid2label_ckin_id_list = {}

for uid in uid2ckin:
    ckin_list = uid2ckin[uid]
    ckin_num = len(ckin_list)
    for ckin_id in range(ckin_num):
        counter = 1
        for sub_ckin_id in range(ckin_id - 1, -1, -1):
            delta_t = int(ckin_list[ckin_id][0]) - int(ckin_list[sub_ckin_id][0])
            if delta_t < hour_offset * 3600 or delta_t % (7 * 24 * 3600) <= 0.5 * hour_offset * 3600 or delta_t % (
                    7 * 24 * 3600) >= (7 * 24 * 3600) - 0.5 * hour_offset * 3600:
                counter += 1
                if counter == train_seq_len + predict_seq_len:
                    break
        if counter == train_seq_len + predict_seq_len:
            counter = 1
            for sub_ckin_id in range(ckin_id - 1, -1, -1):
                delta_t = int(ckin_list[ckin_id][0]) - int(ckin_list[sub_ckin_id][0])
                if delta_t < hour_offset * 3600 or delta_t % (24 * 3600) <= 0.5 * hour_offset * 3600 or delta_t % (
                        24 * 3600) >= (24 * 3600) - 0.5 * hour_offset * 3600:
                    counter += 1
                    if counter == train_seq_len + predict_seq_len:
                        break
            if counter == train_seq_len + predict_seq_len:
                if int(ckin_list[ckin_id][0]) < split_timestamp:
                    if uid not in train_uid2label_ckin_id_list:
                        train_uid2label_ckin_id_list[uid] = []
                    train_uid2label_ckin_id_list[uid].append(ckin_id)
                else:
                    if uid not in test_uid2label_ckin_id_list:
                        test_uid2label_ckin_id_list[uid] = []
                    test_uid2label_ckin_id_list[uid].append(ckin_id)

train_sample_num = 0
for uid in train_uid2label_ckin_id_list:
    train_sample_num += len(train_uid2label_ckin_id_list[uid])
test_sample_num = 0
for uid in test_uid2label_ckin_id_list:
    test_sample_num += len(test_uid2label_ckin_id_list[uid])

np.save('train_uid2label_ckin_id_list.npy', train_uid2label_ckin_id_list)
np.save('test_uid2label_ckin_id_list.npy', test_uid2label_ckin_id_list)

print(len(list(train_uid2label_ckin_id_list.keys())), len(list(test_uid2label_ckin_id_list.keys())))
print('train_sample_num:', train_sample_num, 'test_sample_num:', test_sample_num)

# 274 383
# train_sample_num: 10704 test_sample_num: 15770
