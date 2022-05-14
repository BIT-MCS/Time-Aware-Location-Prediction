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


def idx2xy_wrong(idx):
    x = idx // CONF['frame_size']
    y = idx % CONF['frame_size'] - 1
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
    lon_step = (lon_max - lon_min) / CONF['frame_size']
    lat_step = (lat_max - lat_min) / CONF['frame_size']
    idx_x_float = (float(dataset.pid2ll[pid][0]) - DataCONF['lon_min']) / lon_step
    idx_y_float = (float(dataset.pid2ll[pid][1]) - DataCONF['lat_min']) / lat_step
    x = min(int(idx_x_float), CONF['frame_size'] - 1)
    y = min(int(idx_y_float), CONF['frame_size'] - 1)
    return x, y


def cal_acc1_check(predict_batch, ckin_label_batch, m=5, log=None, validate_iter=None):
    batch = len(predict_batch)
    acc = 0.
    for b in range(batch):
        xylabel = []
        for i in range(len(ckin_label_batch[b][0])):
            ckin_id = ckin_label_batch[b][0][i]
            uid = ckin_label_batch[b][1]
            xy = ckin_id_uid2xy(ckin_id, uid)
            xylabel.append(xy)
        pre = []
        wrong_pre = []
        is_right = False
        for i in range(m):
            idx = predict_batch[b, i]
            x, y = idx2xy(idx)
            wrong_pre.append((x, y))
            if (x, y) in xylabel and not is_right:
                acc += 1.
                pre.append((x, y))
                is_right = True

        if log is not None and is_right:
            log.record_acc1(validate_iter, m, b, xylabel, wrong_pre)
        if log is not None and not is_right:
            log.record_facc1(validate_iter, m, b, xylabel, wrong_pre)
    print('m1=', m, 'is', acc / (batch))

    return acc / (batch)


def cal_acc2_check(predict_batch, ckin_label_batch, m=5, log=None, validate_iter=None):
    batch = len(predict_batch)
    acc = 0.
    acc_total = 0.
    for b in range(batch):
        uid = ckin_label_batch[b][1]
        xylabel = []
        pre = []

        for i in range(len(ckin_label_batch[b][0])):
            ckin_id = ckin_label_batch[b][0][i]
            xy = ckin_id_uid2xy(ckin_id, uid)
            xylabel.append(xy)
            if xy in [(idx2xy(idx)[0], idx2xy(idx)[1]) for idx in predict_batch[b, :m]]:
                acc += 1
                pre.append(xy)
        acc_total += len(ckin_label_batch[b][0])
        if log is not None and len(pre) != 0:
            log.record_acc2(validate_iter, m, b, xylabel, pre)
        if log is not None and len(pre) != 0:
            log.record_facc2(validate_iter, m, b, xylabel, predict_batch[b, :m])

    print('m2=', m, 'is', acc / acc_total)
    return acc / acc_total


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


def train():
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
    if CONF['train_load_model']:
        model_path = os.path.join(CONF['log_path'], CONF['name'], '2019', CONF['model_path'], 'model.pth')
        preModel.load_state_dict(torch.load(model_path))

        print('load model from ', model_path)
    preModel.to(device)

    crit = L2Loss
    # crit2 = MAPELoss
    # crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(preModel.parameters(), lr=CONF['learning_rate'],
                           eps=CONF['eps'],
                           weight_decay=1e-6
                           )

    train_loss_list = []

    log = LogUtil(root_path=CONF['log_path'], method_name=CONF['name'])

    start_time = time.time()
    eval_counter = 0
    max_IoU = 0
    for i in range(CONF['num_iter']):

        data_batch = dataset.get_train_batch()

        predict_batch = preModel(data_batch[:, -1 - CONF['input_len']:-1])
        # TODO

        optimizer.zero_grad()
        loss = crit(predict_batch[:, -1], data_batch[:, -1]) * CONF['input_len']
        # loss2 = crit2(predict_batch, data_batch[:, -1])
        # loss = loss1 + loss2
        # predict_batch_flatten = predict_batch.view(predict_batch.shape[0], -1)

        loss.backward()
        optimizer.step()

        print(dataset_version, 'times:', CONF['times'], CONF['device'], CONF['name'],
              'iter %d, loss %.5f' % (i, loss.item()))
        train_loss_list.append(loss.item())

        # lr = adjust_learning_rate(optimizer=optimizer, lr=lr, decay_rate=CONF['decay_rate'])
        # min_mse = 0.39

        if i % CONF['validate_iter'] == 0 and i > 3e4:

            end_time = time.time()
            with torch.no_grad():

                predict_recom_seq = []
                data_dict = []
                mse_list = []
                preModel.eval()
                while True:
                    data_batch, data_dict_batch, ckin_batch, uid_batch, _ = dataset.get_test_batch()
                    if data_batch is None:
                        break

                    predict_batch = preModel(data_batch[:, -1 - CONF['input_len']:-1])

                    label_batch = data_batch[:, -1].cpu().view(-1).numpy()
                    predict_recom = get_recom(predict_batch[:, -1].cpu().numpy())
                    # predict_recom = get_recom(data_batch[:, -1].cpu().numpy())

                    mse_list.append(np.mean(np.square(predict_batch[:, -1].cpu().view(-1).numpy() - label_batch)))

                    predict_recom_seq.append(predict_recom)
                    data_dict.append(data_dict_batch)

                data_dict = np.concatenate(data_dict, axis=0)
                if len(data_dict) != dataset.test_seq_frame.shape[0]:
                    print('error')
                preModel.train()
                # print('mse ', np.mean(mse_list))
                log.record_eval_mse(np.mean(mse_list))

                predict_recom_seq = np.concatenate(predict_recom_seq, axis=0)
                # acc_m1 = cal_acc1_check(predict_recom_seq, label_recom_seq, m=1, log=log, validate_iter=i)
                # cal_acc1_check(predict_recom_seq, label_recom_seq, m=5, log=log, validate_iter=i)
                # cal_acc1_check(predict_recom_seq, label_recom_seq, m=10, log=log, validate_iter=i)
                # cal_acc1_check(predict_recom_seq, label_recom_seq, m=100, log=log, validate_iter=i)
                # cal_acc1_check(predict_recom_seq, label_recom_seq, m=1000, log=log, validate_iter=i)
                IoU = new_cal_IoU_check(predict_recom_seq, data_dict, log=log)
                # cal_acc2_check(predict_recom_seq, label_recom_seq, m=1, log=log, validate_iter=i)
                # cal_acc2_check(predict_recom_seq, label_recom_seq, m=5, log=log, validate_iter=i)
                # cal_acc2_check(predict_recom_seq, label_recom_seq, m=10, log=log, validate_iter=i)
                # cal_acc2_check(predict_recom_seq, label_recom_seq, m=20, log=log, validate_iter=i)
                # acc_m2 = cal_acc2_check(predict_recom_seq, label_recom_seq, m=200, log=log, validate_iter=i)
                # cal_acc2_check(predict_recom_seq, label_recom_seq, m=1000, log=log, validate_iter=i)

                # cal_acc3_check(predict_recom_seq, label_recom_seq)

                log.record_loss(np.mean(train_loss_list))
                # log.record_acc200(acc_m2)
                eval_counter += 1

                if IoU > max_IoU:
                    max_IoU = IoU
                    log.save_model(preModel)

                report_str = '%d, iter: %d, train_time: %.2f, train_loss: %.5f, max_IoU: %.5f' % (
                    eval_counter, i, end_time - start_time, np.mean(train_loss_list), max_IoU)

                print(report_str)

                log.record_report(report_str)
                # if eval_counter == 7:
                #     break
            train_loss_list = []
            start_time = time.time()


if __name__ == '__main__':
    train()
