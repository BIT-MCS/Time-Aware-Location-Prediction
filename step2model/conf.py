database_name = 'Weeplaces'

if database_name == 'Weeplaces':
    lon_min = -74.0494
    lon_max = -73.8794
    lat_min = 40.6596
    lat_max = 40.7886
elif database_name == 'Brightkite':
    lon_min = 139.496
    lon_max = 139.814127
    lat_min = 35.5285
    lat_max = 35.7874
elif database_name == 'Gowalla':
    lon_min = 17.8136
    lon_max = 18.3186
    lat_min = 59.2238
    lat_max = 59.4822

root_path = '/data1/wangyu191122/'

CONF = {
    'uid2ckin_path': root_path + 'wangyu_data/' + database_name + '/npySet/partNpySet/first_delete_p_user_id2user_checkins.npy',
    'pid2ll_path': root_path + 'wangyu_data/' + database_name + '/npySet/partNpySet/p_poi_id2poi_ll.npy',
    'train_seq_len': 20,
    'predict_seq_len': 1,
    'hour_offset': 1,
    'split_timestamp': 1284119719,
    # 'split_timestamp': 1282670642,
    'train_uid2label_ckin_id_list_path': 'train_uid2label_ckin_id_list.npy',
    'test_uid2label_ckin_id_list_path':  'test_uid2label_ckin_id_list.npy',
    'embedding_size': 128,

    # 'result_npy_path': '/home/liuchi/wangyu191122/wangyu_dad/step1model/log/att_channel_convlstm_64_v2/2020/01-05/14-22-37/result.npy',
    'result_npy_path': '/data1/wangyu191122/wangyu_dad/step1model/log/att_channel_convlstm_64_v2_c6/2020/01-12/22-32-22/z_result.npy',
    'frame_size': 512,

}
