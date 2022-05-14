from weeplaces.step2model.lma_l_50.model import *
from weeplaces.step2model.lma_l_50_v2.model_param import *
from weeplaces.step2model.conf import *
from weeplaces.step2model.utils import *

uid2ckin = np.load(uid2ckin_path, allow_pickle=True)
pid2ll = np.load(pid2ll_path, allow_pickle=True)

train_input = np.load(train_input_path, allow_pickle=True)
train_help_input = np.load(train_help_input_path, allow_pickle=True)
train_right_answer = np.load(train_right_answer_path, allow_pickle=True)
train_loc = np.load(train_loc_path, allow_pickle=True)

test_input = np.load(test_input_path, allow_pickle=True)
test_help_input = np.load(test_help_input_path, allow_pickle=True)
test_right_answer = np.load(test_right_answer_path, allow_pickle=True)
test_loc = np.load(test_loc_path, allow_pickle=True)

poi_id_set = np.load('../small_poi_id_set.npy', allow_pickle=True)
poi_id_set.sort()
poi_id2poi_ctn_id = {}
for i, poi_id in enumerate(poi_id_set):
    poi_id2poi_ctn_id[poi_id] = i
poi_num = len(poi_id_set)

def cal_acc(acc, pred, label):
    sorted_pred = np.argsort(-pred, axis=1)

    B = sorted_pred.shape[0]
    for b in range(B):
        for K in acc.keys():
            if label[b] in sorted_pred[b, :K]:
                acc[K] += 1.


def get_train_batch():
    indices = np.random.choice(train_input.shape[0], train_batch_size)
    train_input_batch = torch.tensor(train_input[indices], dtype=torch.float32, device=device)
    train_help_input_batch = torch.tensor(train_help_input[indices], dtype=torch.float32, device=device)
    train_right_answer_batch = train_right_answer[indices]
    train_ctn_right_answer_batch = np.zeros([train_right_answer_batch.shape[0], train_right_answer_batch.shape[1]])
    for b in range(train_right_answer_batch.shape[0]):
        for t in range(train_right_answer_batch.shape[1]):
            right_answer = train_right_answer_batch[b, t]
            train_ctn_right_answer_batch[b, t] = poi_id2poi_ctn_id[right_answer]

    train_ctn_right_answer_batch = torch.tensor(train_ctn_right_answer_batch, dtype=torch.long, device=device)

    train_loc_batch = train_loc[indices]

    return train_input_batch, train_help_input_batch, train_ctn_right_answer_batch, train_loc_batch


tst_counter = 0


def get_test_batch():
    global tst_counter
    test_input_batch = torch.tensor(test_input[tst_counter: tst_counter + test_batch_size],
                                    dtype=torch.float32, device=device)
    test_help_input_batch = torch.tensor(test_help_input[tst_counter: tst_counter + test_batch_size],
                                         dtype=torch.float32, device=device)
    test_right_answer_batch = test_right_answer[tst_counter: tst_counter + test_batch_size]
    test_ctn_right_answer_batch = np.zeros([test_right_answer_batch.shape[0], test_right_answer_batch.shape[1]])
    for b in range(test_right_answer_batch.shape[0]):
        for t in range(test_right_answer_batch.shape[1]):
            right_answer = test_right_answer_batch[b, t]
            test_ctn_right_answer_batch[b, t] = poi_id2poi_ctn_id[right_answer]
    test_loc_batch = test_loc[tst_counter: tst_counter + test_batch_size]
    if tst_counter + test_batch_size < test_input.shape[0]:
        has_left = True
        tst_counter += test_batch_size
    else:
        has_left = False
        tst_counter = 0

    return test_input_batch, test_help_input_batch, test_ctn_right_answer_batch, has_left, test_loc_batch


def train():
    print('Using device ', device)

    preModel = SequentialModel(
        poi_num=poi_num,
        input_size=input_size,
        cell_size=cell_size,
        device=device,
        frame_size=frame_size,
        seq_len=total_seq_len - 1,
    )
    if train_load_model:
        model_path = log_path + '2019' + '/' + retrain_path + '/' + 'model.pth'
        preModel.load_state_dict(torch.load(model_path))

        print('load model from ', model_path)
    preModel.to(device)

    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(preModel.parameters(), lr=learning_rate, eps=eps, weight_decay=1e-6)

    train_loss_list = []

    log = LogUtil(log_path=log_path)

    max_matrix = 0

    for i in range(iter_num):

        train_input_batch, train_help_input_batch, train_ctn_right_answer_batch, train_loc_batch = get_train_batch()

        predict_batch, _ = preModel(train_input_batch, train_help_input_batch, train_loc_batch)

        optimizer.zero_grad()
        flatten_predict_batch = predict_batch.view(-1, poi_num)
        flatten_train_ctn_right_answer_batch = train_ctn_right_answer_batch.view(-1)
        loss = crit(flatten_predict_batch, flatten_train_ctn_right_answer_batch)

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

        # lr = adjust_learning_rate(optimizer=optimizer, lr=lr, decay_rate=CONF['decay_rate'])
        print(device, database_name, model_name, 'iter %d, loss %.5f' % (i, loss.item()))
        if i % validate_iter == 0 and i > 0:
            print(device, database_name, model_name, 'iter %d, loss %.5f' % (i, loss.item()))
            with torch.no_grad():
                preModel.eval()
                acc = {1: 0, 5: 0, 10: 0}
                acc_total = 0
                att_weight_final_list = []
                all_mem_score_list = []
                while True:
                    test_input_batch, test_help_input_batch, test_ctn_right_answer_batch, \
                    has_left, test_loc_batch = get_test_batch()

                    predict_batch, mem_score_batch = preModel(test_input_batch, test_help_input_batch,
                                                              test_loc_batch)

                    all_mem_score_list.append(mem_score_batch)

                    acc_total += test_ctn_right_answer_batch.shape[0]
                    # print(acc_total)
                    predict_batch = predict_batch.cpu().numpy()
                    cal_acc(acc, predict_batch[:, -1], test_ctn_right_answer_batch[:, -1])
                    if not has_left:
                        break
                # all_mem_score = np.concatenate(all_mem_score_list, axis=0)
                # indices = np.random.choice(all_mem_score.shape[0], 1)
                # loc = test_loc[indices]
                # log.record_mem_att(all_mem_score[indices], loc, str(i))
                preModel.train()

                # att_weight_final_array = torch.cat(att_weight_final_list, dim=0)
                # indices = np.random.choice(att_weight_final_array.shape[0], 5)
                # print(att_weight_final_array[indices])

                for k in acc.keys():
                    acc[k] /= acc_total
                for k in acc.keys():
                    print(k, acc[k])
                print('acc@1_max:', max_matrix)

                log.record_loss(np.mean(train_loss_list))
                train_loss_list = []
                if max_matrix < acc[1]:
                    log.save_model(preModel)
                    max_matrix = acc[1]
                    print('--------------------------------')
                    print('new_acc@1_max:', max_matrix)
                    print('--------------------------------')


if __name__ == '__main__':
    train()