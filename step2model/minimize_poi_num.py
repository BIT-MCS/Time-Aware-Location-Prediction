from weeplaces.step2model.conf import *
from weeplaces.step2model.utils import *

uid2ckin = np.load(CONF['uid2ckin_path'], allow_pickle=True)[()]
poi_id_dict = {}
uid_dict = {}
for dataset_version in ['4']:
    train_uid2seq_list = np.load('train_uid2seq_list_' + dataset_version + '.npy', allow_pickle=True)[()]
    test_uid2seq_list = np.load('test_uid2seq_list_' + dataset_version + '.npy', allow_pickle=True)[()]
    for uid in train_uid2seq_list:
        if uid not in uid_dict:
            uid_dict[uid] = 1
        for seq in train_uid2seq_list[uid]:
            for ckin_id in seq:
                if uid2ckin[uid][ckin_id][-1] not in poi_id_dict:
                    poi_id_dict[uid2ckin[uid][ckin_id][-1]] = 1
    for uid in test_uid2seq_list:
        if uid not in uid_dict:
            uid_dict[uid] = 1
        for seq in test_uid2seq_list[uid]:
            for ckin_id in seq:
                if uid2ckin[uid][ckin_id][-1] not in poi_id_dict:
                    poi_id_dict[uid2ckin[uid][ckin_id][-1]] = 1

uid_set = list(uid_dict.keys())
poi_id_set = list(poi_id_dict.keys())

user_num = len(uid_set)
poi_num = len(poi_id_set)

print('user_num:', user_num, 'poi_num:', poi_num)

np.save('small_user_id_set.npy', uid_set)
np.save('small_poi_id_set.npy', poi_id_set)

# user_num: 410 poi_num: 11812
