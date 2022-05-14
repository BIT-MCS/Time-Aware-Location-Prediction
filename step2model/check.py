import numpy as np
from step2model.conf import *
from step2model.utils import *

train_uid2label_ckin_id_list = np.load(CONF['train_uid2label_ckin_id_list_path'], allow_pickle=True)[()]
test_uid2label_ckin_id_list = np.load(CONF['test_uid2label_ckin_id_list_path'], allow_pickle=True)[()]

print('len uid in train', len(train_uid2label_ckin_id_list.keys()))
print('len uid in test', len(test_uid2label_ckin_id_list.keys()))
train_uid_set = set(train_uid2label_ckin_id_list.keys())
test_uid_set = set(test_uid2label_ckin_id_list.keys())
inter_set = train_uid_set & test_uid_set
print('len uid in both train and test', len(inter_set))
