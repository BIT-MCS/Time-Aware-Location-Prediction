from weeplaces.step2model.lma_l_50.model import *
from weeplaces.step2model.lma_l_50_v2.model_param import *
from weeplaces.step2model.conf import *
from weeplaces.step2model.utils import *
import time
import csv
import xlsxwriter

uid2ckin = np.load(uid2ckin_path, allow_pickle=True)
pid2ll = np.load(pid2ll_path, allow_pickle=True)

train_input = np.load(train_input_path, allow_pickle=True)
train_help_input = np.load(train_help_input_path, allow_pickle=True)
train_right_answer = np.load(train_right_answer_path, allow_pickle=True)

test_input = np.load(test_input_path, allow_pickle=True)
test_help_input = np.load(test_help_input_path, allow_pickle=True)
test_right_answer = np.load(test_right_answer_path, allow_pickle=True)
test_loc = np.load(test_loc_path, allow_pickle=True)
user_appear = np.load(user_appear_path, allow_pickle=True)

poi_id_set = np.load('../small_poi_id_set.npy', allow_pickle=True)
poi_id_set.sort()
poi_id2poi_ctn_id = {}
for i, poi_id in enumerate(poi_id_set):
    poi_id2poi_ctn_id[poi_id] = i
poi_num = len(poi_id_set)


test_long_view = np.load('../lma_l_50/test_long_view' + dataset_version + '.npy', allow_pickle=True)
test_short_view = np.load('../lma_l_50/test_short_view' + dataset_version + '.npy', allow_pickle=True)
test_history_pid_list = np.load('../lma_l_50/test_history_pid_list' + dataset_version + '.npy', allow_pickle=True)


def change_pid2ctn_pid(pid_list):
    ctn_pid_list = []
    test_view = test_long_view if is_long_view else test_short_view
    for b in range(pid_list.shape[0]):
        if pid_list[b] is None:
            ctn_pid_list.append(None)
            continue
        view_len = max(int(test_view[b][0] * view_times), 1)
        ctn_pid_temp = []
        for pid_l in pid_list[b][:view_len]:
            for pid in pid_l:
                if pid in poi_id2poi_ctn_id:
                    ctn_pid_temp.append(poi_id2poi_ctn_id[pid])
        if delete_poi_never_come:
            history_ctn_pid = set()
            for pid in test_history_pid_list[b]:
                history_ctn_pid.add(poi_id2poi_ctn_id[pid])
            ctn_pid_temp = set(ctn_pid_temp) & history_ctn_pid
        ctn_pid_list.append(list(ctn_pid_temp))

    return ctn_pid_list


test_pred_pid_list = np.load('../lma_l_50/test_pred_pid_list' + dataset_version + '.npy', allow_pickle=True)
test_pred_ctn_pid_list = change_pid2ctn_pid(test_pred_pid_list)
test_label_pid_list = np.load('../lma_l_50/test_label_pid_list' + dataset_version + '.npy', allow_pickle=True)
test_label_ctn_pid_list = change_pid2ctn_pid(test_label_pid_list)

def get_sorted_idx(pred, test_ctn_pid_list_batch, b, all_sorted_pred):
    pred_idx = np.array(test_ctn_pid_list_batch[b])
    pred_value = pred[b][test_ctn_pid_list_batch[b]]
    sorted_pred_idx = pred_idx[np.argsort(-pred_value)]

    # left_sorted_pred_idx = []
    # for idx in all_sorted_pred:
    #     if idx not in sorted_pred_idx:
    #         left_sorted_pred_idx.append(idx)
    #
    # return list(sorted_pred_idx) + left_sorted_pred_idx

    left_sorted_pred_idx = np.array(list(set(all_sorted_pred).difference(set(sorted_pred_idx))))
    left_pred_value = pred[b][left_sorted_pred_idx]
    left_sorted_pred_idx = left_sorted_pred_idx[np.argsort(-left_pred_value)]
    return list(sorted_pred_idx) + list(left_sorted_pred_idx)

    # return sorted_pred_idx


def update_acc_dict(acc_dict, user_appear_batch, K, b):
    acc_dict[K][0] += 1.
    if user_appear_batch[b] == 1:
        acc_dict[K][1] += 1.
    else:
        acc_dict[K][2] += 1.
    return acc_dict


def update_MAP(MAP, user_appear_batch, b, idx):
    MAP[0] += 1. / (idx + 1)
    if user_appear_batch[b] == 1:
        MAP[1] += 1. / (idx + 1)
    else:
        MAP[2] += 1. / (idx + 1)
    return MAP


def cal_acc_use_step1(acc, acc_label, pred, label, test_pred_ctn_pid_list_batch, test_label_ctn_pid_list_batch,
                      MAP_step1, user_appear_batch):
    B = pred.shape[0]
    sorted_pred = np.argsort(-pred, axis=1)
    for b in range(B):
        if test_pred_ctn_pid_list_batch[b] is None:
            for K in acc.keys():
                if label[b] in sorted_pred[b, :K]:
                    acc = update_acc_dict(acc, user_appear_batch, K, b)
                    acc_label = update_acc_dict(acc_label, user_appear_batch, K, b)

            if label[b] in sorted_pred[b, :]:
                idx = list(sorted_pred[b, :]).index(label[b])
                MAP_step1 = update_MAP(MAP_step1, user_appear_batch, b, idx)
            continue

        sorted_pred_idx = get_sorted_idx(pred, test_pred_ctn_pid_list_batch, b, sorted_pred[b, :])
        sorted_label_idx = get_sorted_idx(pred, test_label_ctn_pid_list_batch, b, sorted_pred[b, :])
        for K in acc.keys():
            if label[b] in sorted_pred_idx[:K]:
                acc = update_acc_dict(acc, user_appear_batch, K, b)
        for K in acc_label.keys():
            if label[b] in sorted_label_idx[:K]:
                acc_label = update_acc_dict(acc_label, user_appear_batch, K, b)
        if label[b] in sorted_pred_idx:
            idx = list(sorted_pred_idx).index(label[b])
            MAP_step1 = update_MAP(MAP_step1, user_appear_batch, b, idx)
    return MAP_step1



def cal_acc(acc, pred, label, MAP, user_appear_batch):
    sorted_pred = np.argsort(-pred, axis=1)

    B = sorted_pred.shape[0]
    for b in range(B):
        for K in acc.keys():
            if label[b] in sorted_pred[b, :K]:
                acc = update_acc_dict(acc, user_appear_batch, K, b)

        idx = list(sorted_pred[b, :]).index(label[b])
        MAP = update_MAP(MAP, user_appear_batch, b, idx)

    return MAP


tst_counter = 0


def get_test_batch():
    global tst_counter
    test_input_batch = torch.tensor(test_input[tst_counter: tst_counter + test_batch_size],
                                    dtype=torch.float32, device=device)
    test_help_input_batch = torch.tensor(test_help_input[tst_counter: tst_counter + test_batch_size],
                                         dtype=torch.float32, device=device)
    test_right_answer_batch = test_right_answer[tst_counter: tst_counter + test_batch_size]
    test_ctn_right_answer_batch = np.zeros([test_right_answer_batch.shape[0], test_right_answer_batch.shape[1]])

    test_pred_ctn_pid_list_batch = test_pred_ctn_pid_list[tst_counter: tst_counter + test_batch_size]
    test_label_ctn_pid_list_batch = test_label_ctn_pid_list[tst_counter: tst_counter + test_batch_size]
    test_loc_batch = test_loc[tst_counter: tst_counter + test_batch_size]
    user_appear_batch = user_appear[tst_counter: tst_counter + test_batch_size]

    for b in range(test_right_answer_batch.shape[0]):
        for t in range(test_right_answer_batch.shape[1]):
            right_answer = test_right_answer_batch[b, t]
            test_ctn_right_answer_batch[b, t] = poi_id2poi_ctn_id[right_answer]

    if tst_counter + test_batch_size < test_input.shape[0]:
        has_left = True
        tst_counter += test_batch_size
    else:
        has_left = False
        tst_counter = 0

    return test_input_batch, test_help_input_batch, \
           test_ctn_right_answer_batch, \
           test_pred_ctn_pid_list_batch, \
           test_label_ctn_pid_list_batch, \
           has_left, \
           test_loc_batch, \
           user_appear_batch


def normal_acc_dict(acc_dict, acc_total, show_user_acc_total, no_show_user_acc_total):
    for k in acc_dict:
        acc_dict[k][0] /= acc_total
        acc_dict[k][1] /= show_user_acc_total
        acc_dict[k][2] /= no_show_user_acc_total
    return acc_dict


def normal_MAP(MAP, acc_total, show_user_acc_total, no_show_user_acc_total):
    MAP[0] /= acc_total
    MAP[1] /= show_user_acc_total
    MAP[2] /= no_show_user_acc_total
    return MAP


class Excel(object):
    # 初始化，设置文件名
    def __init__(self, name):
        self.book = xlsxwriter.Workbook(name)
        self.sheet = self.book.add_worksheet()

    # 写入列名
    def write_colume_name(self, colums_name):
        for i in range(0, len(colums_name)):
            self.sheet.write(0, i, colums_name[i])

    # 写入数据
    def write_content(self, row_num, data):
        for i in range(0, len(data)):
            self.sheet.write(row_num, i, data[i])

    # 关闭文件
    def close(self):
        self.book.close()


def write_to_file(acc, acc_w_step1, MAP, MAP_step1):
    file_name = database_name + '_' + model_name + '_acc_map.xlsx'

    print(file_name)
    book = Excel(file_name)
    book.write_colume_name(['acc_k', 'total', 'show', 'no_show'])
    counter = 0

    for k in acc.keys():
        counter += 1
        book.write_content(counter, [k] + list(acc[k]))

    counter += 1
    book.write_content(counter, ['MAP'] + list(MAP))

    counter += 1
    book.write_content(counter, ['0 ', '0', '0', '0'])

    for k in acc_w_step1:
        counter += 1
        book.write_content(counter, [k] + list(acc_w_step1[k]))

    counter += 1
    book.write_content(counter, ['MAP_w_step1'] + list(MAP_step1))
    book.close()


def test():
    print('Using device ', device)
    print(poi_num)
    preModel = SequentialModel(
        poi_num=poi_num,
        input_size=input_size,
        cell_size=cell_size,
        device=device,
        frame_size=frame_size,
        seq_len=total_seq_len - 1,
    )

    model_path = log_path + test_path + '/' + 'model.pth'
    preModel.load_state_dict(torch.load(model_path, map_location=device))
    print('load model from ', model_path)
    preModel.to(device)
    preModel.eval()

    with torch.no_grad():
        key_list = [1, 5, 10, 15, 20, 30]
        acc_w_step1 = {}
        acc_w_step1_label = {}
        acc = {}
        for k in key_list:
            acc_w_step1[k] = np.zeros([3])
            acc_w_step1_label[k] = np.zeros([3])
            acc[k] = np.zeros([3])
        MAP = np.zeros([3])
        MAP_step1 = np.zeros([3])
        acc_total = 0
        while True:
            st_tm = time.time()

            test_input_batch, test_help_input_batch, test_ctn_right_answer_batch, \
            test_pred_ctn_pid_list_batch, test_label_ctn_pid_list_batch, \
            has_left, test_loc_batch, user_appear_batch = get_test_batch()

            predict_batch, att_weight_final = preModel(test_input_batch, test_help_input_batch, test_loc_batch)

            acc_total += test_ctn_right_answer_batch.shape[0]
            # print(acc_total)
            predict_batch = predict_batch.cpu().numpy()
            MAP = cal_acc(acc, predict_batch[:, -1], test_ctn_right_answer_batch[:, -1], MAP, user_appear_batch)
            MAP_step1 = cal_acc_use_step1(acc_w_step1, acc_w_step1_label, predict_batch[:, -1],
                                          test_ctn_right_answer_batch[:, -1],
                                          test_pred_ctn_pid_list_batch, test_label_ctn_pid_list_batch, MAP_step1,
                                          user_appear_batch)
            ed_tm = time.time()
            print(ed_tm - st_tm)

            if not has_left:
                break
        print(acc_total)
        show_user_acc_total = np.sum(user_appear)
        no_show_user_acc_total = acc_total - show_user_acc_total

        # for k in acc:
        #     if not abs(acc[k][0] - (acc[k][1] + acc[k][2])) < 1e-6:
        #         print('error')
        # if not abs(MAP[0] - (MAP[1] + MAP[2])) < 1e-6:
        #     print('MAP error')

        acc = normal_acc_dict(acc, acc_total, show_user_acc_total, no_show_user_acc_total)
        acc_w_step1 = normal_acc_dict(acc_w_step1, acc_total, show_user_acc_total, no_show_user_acc_total)
        acc_w_step1_label = normal_acc_dict(acc_w_step1_label, acc_total, show_user_acc_total, no_show_user_acc_total)

        MAP = normal_MAP(MAP, acc_total, show_user_acc_total, no_show_user_acc_total)
        MAP_step1 = normal_MAP(MAP_step1, acc_total, show_user_acc_total, no_show_user_acc_total)
        for k in acc_w_step1.keys():
            print(k, acc[k], acc_w_step1[k], acc_w_step1_label[k])
        print('MAP', MAP, 'MAP_step1', MAP_step1)

        write_to_file(acc, acc_w_step1, MAP, MAP_step1)
        print('end')


if __name__ == '__main__':
    st = time.time()
    test()
    ed = time.time()
    print((ed - st) / 60)
