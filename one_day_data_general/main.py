from weeplaces.one_day_data_general.data_loader import *
from weeplaces.one_day_data_general.conf import *

if __name__ == '__main__':
    data_loader = DataLoader(
        name=CONF['name'],

        lon_min=CONF['lon_min'],
        lon_max=CONF['lon_max'],

        lat_min=CONF['lat_min'],
        lat_max=CONF['lat_max'],

        cell_size=CONF['cell_size'],

        daytime_inter=CONF['daytime_inter'],

        seq_len=CONF['seq_len'],

        root_path=CONF['root_path'],

        train_ratio=CONF['train_ratio'],

        min_ckin_cell_per_frame=CONF['min_ckin_cell_per_frame'],

        version=CONF['version'],

        is_delete=CONF['is_delete'],
    )

    data_loader.gen_all_people_seq()
    data_loader.gen_contin_train_and_test()
    data_loader.save_train_and_test_seq()
