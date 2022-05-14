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

CONF = {
    'name': database_name,

    'lon_min': lon_min,
    'lon_max': lon_max,

    'lat_min': lat_min,
    'lat_max': lat_max,

    'daytime_inter': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
                      (11, 12),
                      (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
                      (18, 19), (19, 20),
                      (20, 21), (21, 22), (22, 23), (23, 24)],
    'version': 'genseq8cell64channel24',

    'root_path': '',

    'train_ratio': 0.896,

    'seq_len': 8,
    'cell_size': 64,

    'min_ckin_cell_per_frame': 1,

    'is_delete': True,
}
