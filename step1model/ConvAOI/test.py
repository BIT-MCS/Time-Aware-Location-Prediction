from weeplaces.step1model.att_channel_convlstm_64_v2_c6.conf import *
from weeplaces.step1model.utils import *

np.random.seed(CONF['seed'])
torch.manual_seed(CONF['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from weeplaces.step1model.att_channel_convlstm_64_v2_c6.model import *
from weeplaces.step1model.att_channel_convlstm_64_v2_c6.dataset import *

# fix random seed

dataset = Dataset(
    train_data_path=CONF['train_data_path'],
    test_data_path=CONF['test_data_path'],
    uid2ckin_path=CONF['uid2ckin_path'],
    pid2ll_path=CONF['pid2ll_path'],
    valid_pid_path=CONF['valid_pid_path'],

    train_batch_size=CONF['train_batch_size'],
    test_batch_size=CONF['test_batch_size'],
    device=CONF['device'],
    seq_len=CONF['seq_len'],
    frame_size=CONF['frame_size'],
    times=CONF['times'],
    gamma=CONF['gamma'],
    # new_pid2pid_path=CONF['new_pid2pid_path'],
    input_channel=CONF['input_channel'],
    daytime_inter=CONF['daytime_inter'],
)


def adjust_learning_rate(optimizer, lr, decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_recom(predict_batch):
    batch = predict_batch.shape[0]
    channel = predict_batch.shape[1]
    p_seq_flatten = predict_batch.reshape(batch, channel, -1)
    p_idx = np.argsort(-p_seq_flatten, axis=2)
    return p_idx


def idx2xy(idx):
    x = idx // CONF['frame_size']
    y = idx % CONF['frame_size']
    return x, y


def ckin_id_uid2xy(ckin_id, uid):
    pid = dataset.uid2ckin[uid][ckin_id][1]
    lon_step = (lon_max - lon_min) / CONF['frame_size']
    lat_step = (lat_max - lat_min) / CONF['frame_size']
    idx_x_float = (float(dataset.pid2ll[pid][0]) - DataCONF['lon_min']) / lon_step
    idx_y_float = (float(dataset.pid2ll[pid][1]) - DataCONF['lat_min']) / lat_step
    x = min(int(idx_x_float), CONF['frame_size'] - 1)
    y = min(int(idx_y_float), CONF['frame_size'] - 1)
    return x, y


def pid2xy(pid):
    lon_step = (DataCONF['lon_max'] - DataCONF['lon_min']) / CONF['frame_size']
    lat_step = (DataCONF['lat_max'] - DataCONF['lat_min']) / CONF['frame_size']
    idx_x_float = (float(dataset.pid2ll[pid][0]) - DataCONF['lon_min']) / lon_step
    idx_y_float = (float(dataset.pid2ll[pid][1]) - DataCONF['lat_min']) / lat_step
    x = min(int(idx_x_float), CONF['frame_size'] - 1)
    y = min(int(idx_y_float), CONF['frame_size'] - 1)
    return x, y


def gen_xy2pid_dict():
    xy2pid_dict = {}
    for pid in dataset.pid2ll:
        xy = pid2xy(pid)
        if xy not in xy2pid_dict:
            xy2pid_dict[xy] = []
        xy2pid_dict[xy].append(pid)
    return xy2pid_dict


xy2pid_dict = gen_xy2pid_dict()


def get_pid_list_from_idx_arr(idx_arr, b, c):
    R = idx_arr.shape[2]
    xy_pid_lst = []
    for r in range(R):
        idx = idx_arr[b, c, r]
        xy = idx2xy(idx)
        if xy in xy2pid_dict:
            xy_pid_lst.append(xy2pid_dict[xy])
    return xy_pid_lst


def get_ckin_id_list_from_ckins(ckins):
    ckin_id_list = []
    for ckin in ckins:
        ckin_id_list.append(ckin['ckin_idx'])
    return ckin_id_list


def recom_idx2pid(pred_idx_arr, label_idx_arr, ckin_batch, uid_batch, cell_view_batch):
    batch_xy_pid_lst = []
    B = pred_idx_arr.shape[0]

    for b in range(B):
        for c in range(CONF['input_channel']):
            uid = uid_batch[b][0]
            ckins = dataset.get_ckin_list_by_channel(ckin_batch[b][-1], c)
            ckin_id_list = get_ckin_id_list_from_ckins(ckins)
            pred_xy_pid_lst = get_pid_list_from_idx_arr(pred_idx_arr, b, c)
            label_xy_pid_lst = get_pid_list_from_idx_arr(label_idx_arr, b, c)
            temp = {
                'uid': uid,
                'ckin_id_list': ckin_id_list,
                'pred_pid_list': pred_xy_pid_lst,
                'label_pid_list': label_xy_pid_lst,
                'long_view': cell_view_batch[b][0][c],
                'short_view': cell_view_batch[b][1][c],

            }
            batch_xy_pid_lst.append(temp)
    return batch_xy_pid_lst


def cal_acc1_check(predict_batch, ckin_label_batch, m=5, log=None, validate_iter=None):
    batch = len(predict_batch)
    acc = 0.
    for b in range(batch):
        for c in range(CONF['input_channel']):
            xylabel = []
            ckins = dataset.get_ckin_list_by_channel(ckin_label_batch[b][-1], c)

            for ckin in ckins:
                xy = pid2xy(ckin['pid'])
                xylabel.append(xy)
            is_right = False
            for i in range(m):
                idx = predict_batch[b, c, i]
                x, y = idx2xy(idx)
                if (x, y) in xylabel and not is_right:
                    acc += 1.
                    is_right = True
    print('m1=', m, 'is', acc / (batch))


def cal_acc2_check(predict_batch, ckin_label_batch, m=5, log=None, validate_iter=None):
    batch = len(predict_batch)
    acc = 0.
    acc_total = 0.
    for b in range(batch):
        for c in range(CONF['input_channel']):

            xylabel = []
            ckins = dataset.get_ckin_list_by_channel(ckin_label_batch[b][-1], c)
            for ckin in ckins:
                xy = pid2xy(ckin['pid'])
                xylabel.append(xy)
                if xy in [(idx2xy(idx)[0], idx2xy(idx)[1]) for idx in predict_batch[b, c, :m]]:
                    acc += 1
            acc_total += len(ckins)

    print('m2=', m, 'is', acc / acc_total)


def new_cal_IoU_check(predict_batch, data_dict_batch, log=None):
    batch = len(predict_batch)
    IoU_list = []
    for b in range(batch):
        frm = data_dict_batch[b][-1]
        xylabel_set_list = []
        xypred_set_list = []
        for channel in range(CONF['input_channel']):
            xylabel_set_list.append(set(k for k in frm.keys() if k[0] == channel))
        for channel in range(CONF['input_channel']):
            xypred_set_list.append(
                set([(channel, idx2xy(idx)[0], idx2xy(idx)[1]) for idx in
                     predict_batch[b, channel, :len(xylabel_set_list[channel])]]))
        set_inter = 0
        set_union = 0
        for channel in range(CONF['input_channel']):
            set_inter += len(xylabel_set_list[channel] & xypred_set_list[channel])
            set_union += len(xylabel_set_list[channel] | xypred_set_list[channel])
        IoU = set_inter / set_union
        IoU_list.append(IoU)

    if log is not None:
        log.record_IoU(np.mean(IoU_list))

    print('IoU is', np.mean(IoU_list))
    return np.mean(IoU_list)


def record_all_seq_ckin(log, pred, label):
    batch = len(pred)

    for b in range(batch):
        # batch loop
        uid = label[b][1]
        seq_xy = []
        for i in range(len(label[b][0])):
            # frame loop

            frame_xy = []
            for j in range(len(label[b][0][i])):
                # ckin loop
                ckin_id = label[b][0][i][j]
                xy = ckin_id_uid2xy(ckin_id, uid)
                frame_xy.append(xy)
            seq_xy.append(frame_xy)

        pre_xy = []
        for j in range(2 * len(label[b][0][-1])):
            # ckin loop
            idx = pred[b, j]
            x, y = idx2xy(idx)
            pre_xy.append((x, y))
        seq_xy.append(pre_xy)

        log.record_seq_ckins(b, seq_xy)


def test():
    device = CONF['device']

    print('Using device ', device)

    preModel = PredictiveModel(
        input_channel=CONF['input_channel'],
        hidden_channels=CONF['hidden_channels'],
        frame_size=CONF['frame_size'],
        cnn_kernel_size=CONF['cnn_kernel_size'],
        rnn_kernel_size=CONF['rnn_kernel_size'],
        device=CONF['device'],
        input_len=CONF['input_len'],
        seq_len=CONF['seq_len'],

    )

    model_path = os.path.join(CONF['log_path'], CONF['name'], CONF['test_path'], 'model.pth')
    preModel.load_state_dict(torch.load(model_path))

    print('load model from ', model_path)
    preModel.to(device)
    preModel.eval()

    log = LogUtil(root_path=CONF['log_path'], method_name=CONF['name'])

    with torch.no_grad():

        predict_recom_seq = []
        label_recom_seq = []
        all_recom_seq = []
        all_pred_label_xy_pid_lst = []
        data_dict = []

        while True:
            data_batch, data_dict_batch, ckin_batch, uid_batch, cell_view_batch = dataset.get_test_batch()
            if data_batch is None:
                break

            predict_batch = preModel(data_batch[:, -1 - CONF['input_len']:-1])

            predict_recom = get_recom(predict_batch[:, -1].cpu().numpy())
            # predict_recom = get_recom(data_batch[:, -1].cpu().numpy())

            label_recom = get_recom(data_batch[:, -1].cpu().numpy())
            pred_label_xy_pid_lst = recom_idx2pid(predict_recom, label_recom, ckin_batch, uid_batch, cell_view_batch)
            all_pred_label_xy_pid_lst += pred_label_xy_pid_lst

            predict_recom_seq.append(predict_recom)
            label_recom_seq += ckin_batch
            data_dict.append(data_dict_batch)

            print(len(label_recom_seq))

        data_dict = np.concatenate(data_dict, axis=0)
        print('start recording z_result')
        log.record_result(all_pred_label_xy_pid_lst)
        print('end recording z_result')

        predict_recom_seq = np.concatenate(predict_recom_seq, axis=0)
        i = 1

        new_cal_IoU_check(predict_recom_seq, data_dict, log=log)
        cal_acc2_check(predict_recom_seq, label_recom_seq, m=1, log=log, validate_iter=i)
        cal_acc2_check(predict_recom_seq, label_recom_seq, m=5, log=log, validate_iter=i)
        cal_acc2_check(predict_recom_seq, label_recom_seq, m=10, log=log, validate_iter=i)
        cal_acc2_check(predict_recom_seq, label_recom_seq, m=100, )
        cal_acc2_check(predict_recom_seq, label_recom_seq, m=200, )


if __name__ == '__main__':
    test()
