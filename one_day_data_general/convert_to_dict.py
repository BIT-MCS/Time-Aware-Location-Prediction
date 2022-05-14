import numpy as np
from weeplaces.one_day_data_general.conf import *

root_path = ''
uid2ckin_path = root_path + 'first_delete_p_user_id2user_checkins.npy'
uid2ckin_dict_path = root_path + 'first_delete_p_user_id2user_checkins_dict.npy'

uid2ckin = np.load(uid2ckin_path, allow_pickle=True)[()]

uid2ckin_dict = {}
for uid in uid2ckin.keys():
    ckin_list = uid2ckin[uid]
    new_ckin_list = []
    for idx, ckin in enumerate(ckin_list):
        temp_dict = {}
        temp_dict['timestamp'] = ckin[0]
        temp_dict['local_time'] = ckin[1]
        temp_dict['day_of_week'] = ckin[2]
        temp_dict['pid'] = ckin[3]
        temp_dict['ckin_idx'] = idx
        new_ckin_list.append(temp_dict)
    uid2ckin_dict[uid] = new_ckin_list
np.save(uid2ckin_dict_path, uid2ckin_dict)
