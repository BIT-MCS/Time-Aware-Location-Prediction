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

root_path = ''

CONF = {
    'train_batch_size': 32,
    'test_batch_size': 1024,

    'input_channel': 6,
    'daytime_inter': [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)],

    'hidden_channels': [64],
    'cnn_kernel_size': 3,
    'rnn_kernel_size': 3,

    'num_iter': int(1e5),
    'validate_iter': int(1e3),

    'learning_rate': 1e-3,

    'eps': 1e-5,

    'seed': 1,

    'name': 'att_channel_convlstm_64_v2_c6',

    'gamma': int(0),
    'times': 5.,

    'test_path': '2020/01-12/19-33-24',
    'model_path': '10-17/22-59-30',
    'train_load_model': False,

    'seq_len': 8,
    'input_len': 7,
    'frame_size': 64,

    'device': 'cuda:6',

}

DataCONF = {
    'name': database_name,

    'lon_min': lon_min,
    'lon_max': lon_max,

    'lat_min': lat_min,
    'lat_max': lat_max,

}