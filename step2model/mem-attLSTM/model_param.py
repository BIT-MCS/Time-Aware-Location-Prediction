root_path = '/data1/wangyu191122/'
dataset_version = '4'
model_name = 'lma_l_50'
total_seq_len = 51
database_name = 'Weeplaces'
frame_size = 512
total_hour_id = 24
device = 'cuda:7'
test_path = '2020/01-09/12-39-04'

uid2ckin_path = root_path + 'wangyu_data/' + database_name + '/npySet/partNpySet/first_delete_p_user_id2user_checkins.npy'
pid2ll_path = root_path + 'wangyu_data/' + database_name + '/npySet/partNpySet/p_poi_id2poi_ll.npy'

train_input_path = '../lma_l_50/train_input_' + dataset_version + '.npy'
train_help_input_path = '../lma_l_50/train_help_input_' + dataset_version + '.npy'
train_right_answer_path = '../lma_l_50/train_right_answer_' + dataset_version + '.npy'
train_loc_path = '../lma_l_50/train_loc_' + dataset_version + '_' + str(frame_size) + '.npy'

test_input_path = '../lma_l_50/test_input_' + dataset_version + '.npy'
test_help_input_path = '../lma_l_50/test_help_input_' + dataset_version + '.npy'
test_right_answer_path = '../lma_l_50/test_right_answer_' + dataset_version + '.npy'
test_loc_path = '../lma_l_50/test_loc_' + dataset_version + '_' + str(frame_size) + '.npy'
user_appear_path = '../lma_l_50/user_appear.npy'

log_path = '../lma_l_50/log/' + model_name + '/'

is_long_view = False
view_times = 1
delete_poi_never_come = True

input_size = 128
cell_size = 128

train_batch_size = 128
test_batch_size = 1024

# training params
iter_num = int(5e6)
validate_iter = int(10)

learning_rate = 1e-3
# 'decay_rate': 0.999
eps = 1e-5
seed = 1

train_load_model = False
retrain_path = '11-06/22-32-56'
