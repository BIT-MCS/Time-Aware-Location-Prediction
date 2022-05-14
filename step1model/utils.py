import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import *
import numpy as np
import time
import os


# from sklearn.metrics import *


def init(module, weight_init, gain=1):
    weight_init(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module


def calculate_mse(predict, label):
    """
    :param predict: [B,T,C,M,N]
    :param label:   [B,T,C,M,N]
    :return:
    """

    B = label.shape[0]
    T = label.shape[1]
    mse_list = np.zeros(shape=[B, T], dtype=np.float32)
    for b in range(B):
        for t in range(T):
            img = np.transpose(predict[b, t, ...], [1, 2, 0])
            img_const = np.transpose(label[b, t, ...], [1, 2, 0])
            result = np.mean((np.reshape(img_const, [-1]) - np.reshape(img, [-1])) ** 2)
            # z_result = compare_mse(img, img_const)
            # z_result = compare_psnr(img, img_const,data_range=img_const.max() - img_const.min(),)

            mse_list[b, t] = result

    return mse_list, np.mean(np.mean(mse_list, axis=1))


class LogUtil():
    def __init__(self, root_path, method_name, is_train=True):
        self.time = time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime())
        self.is_train = is_train
        if is_train:
            self.full_path = os.path.join(root_path, method_name, self.time)

        else:
            self.full_path = root_path
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        if self.is_train:
            self.loss_path = os.path.join(self.full_path, 'loss.npy')
            self.loss_list = []

            self.IoU_path = os.path.join(self.full_path, 'IoU.npy')
            self.IoU_list = []

            self.eval_mse_path = os.path.join(self.full_path, 'eval_mse.npy')
            self.eval_mse_list = []

            self.eval_acc200_path = os.path.join(self.full_path, 'eval_acc200.npy')
            self.eval_acc200_list = []

            self.report_path = os.path.join(self.full_path, 'train_log.txt')

            self.acc1_path = os.path.join(self.full_path, 'acc1_log.txt')
            self.acc1_wrong_path = os.path.join(self.full_path, 'acc1_wrong_log.txt')
            self.acc2_path = os.path.join(self.full_path, 'acc2_log.txt')
            self.acc2_wrong_path = os.path.join(self.full_path, 'acc2_wrong_log.txt')

            self.facc1_path = os.path.join(self.full_path, 'facc1_log.txt')
            self.facc1_wrong_path = os.path.join(self.full_path, 'facc1_wrong_log.txt')
            self.facc2_path = os.path.join(self.full_path, 'facc2_log.txt')
            self.facc2_wrong_path = os.path.join(self.full_path, 'facc2_wrong_log.txt')

            self.seq_ckin_path = os.path.join(self.full_path, 'seq_ckin_log.txt')

            self.result_path = os.path.join(self.full_path, 'z_result.npy')

            self.model_path = os.path.join(self.full_path, 'model.pth')

        else:
            # self.test_mse_path = self.full_path + '/' + 'test_mse.npy'
            #
            # self.test_ssim_path = self.full_path + '/' + 'test_ssim.npy'
            #
            # self.test_psnr_path = self.full_path + '/' + 'test_psnr.npy'
            #
            # self.test_nrmse_path = self.full_path + '/' + 'test_nrmse.npy'
            #
            # self.test_log_path = self.full_path + '/test_log.txt'
            pass

    def record_loss(self, loss):
        self.loss_list.append(loss)
        np.save(self.loss_path, np.asarray(self.loss_list))

    def record_IoU(self, IoU):
        self.IoU_list.append(IoU)
        np.save(self.IoU_path, np.asarray(self.IoU_list))

    def record_eval_mse(self, eval_mse):
        self.eval_mse_list.append(eval_mse)
        np.save(self.eval_mse_path, np.asarray(self.eval_mse_list))

    def record_acc200(self, acc200):
        self.eval_acc200_list.append(acc200)
        np.save(self.eval_acc200_path, np.asarray(self.eval_acc200_list))

    def record_report(self, report_str):
        f = open(self.report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def record_acc1(self, v, m, b, list1, list2):
        f = open(self.acc1_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_acc2(self, v, m, b, list1, list2):
        f = open(self.acc2_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_acc1_wrong(self, v, m, b, list1, list2):
        f = open(self.acc1_wrong_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_acc2_wrong(self, v, m, b, list1, list2):
        f = open(self.acc2_wrong_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_facc1(self, v, m, b, list1, list2):
        f = open(self.facc1_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_facc2(self, v, m, b, list1, list2):
        f = open(self.facc2_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_facc1_wrong(self, v, m, b, list1, list2):
        f = open(self.facc1_wrong_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_facc2_wrong(self, v, m, b, list1, list2):
        f = open(self.facc2_wrong_path, 'a')
        f.writelines('valid_iter：' + str(v) + ' m：' + str(m) + ' batch：' + str(b) + '\n')
        f.writelines(str(b) + '\n')
        f.writelines(str(list1) + '\n')
        f.writelines(str(list2) + '\n')
        f.writelines('-----------------------------------\n')
        f.close()

    def record_seq_ckins(self, b, seq_ckin_list):
        f = open(self.seq_ckin_path, 'a')
        for i, frm_ckin_list in enumerate(seq_ckin_list):
            f.writelines('batch:' + str(b) + ' frame:' + str(i) + '\n')
            f.writelines(str(frm_ckin_list) + '\n')
            f.writelines('-----------------------------------\n')
        f.close()

    def set_model_path(self, model_name):
        self.model_path = os.path.join(self.full_path, model_name)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print('model has been save to ', self.model_path)

    def record_result(self, result):
        result = np.array(result)
        print(result.shape)
        np.save(self.result_path, result)
        print(self.result_path)


def L2Loss(pred, tar, per_example=False):
    if not per_example:
        loss = 0.5 * torch.sum((pred - tar) ** 2)
    else:
        loss = 0.5 * torch.sum((pred - tar) ** 2) / pred.shape[0]
    return loss


def MAPELoss(pred, tar, min_value=1e-6):
    return torch.mean((torch.abs(pred - tar) + min_value) / (torch.abs(pred) + min_value))


def reshape_patch(img, patch_size):
    img = np.transpose(img, [0, 1, 3, 4, 2])
    assert 5 == img.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img.shape
    a = np.reshape(img, [batch_size, seq_length,
                         int(img_height / patch_size), patch_size,
                         int(img_width / patch_size), patch_size,
                         num_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    patch = np.reshape(b, [batch_size, seq_length,
                           int(img_height / patch_size),
                           int(img_width / patch_size),
                           int(patch_size * patch_size * num_channels)])
    patch = np.transpose(patch, [0, 1, 4, 2, 3])
    return patch


def reshape_patch_back(patch, patch_size):
    patch = np.transpose(patch, [0, 1, 3, 4, 2])
    assert 5 == patch.ndim
    batch_size, seq_length, patch_height, patch_width, channels = patch.shape

    img_channels = int(channels / (patch_size * patch_size))
    a = np.reshape(patch, [batch_size, seq_length,
                           patch_height, patch_width,
                           patch_size, patch_size,
                           img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img = np.reshape(b, [batch_size, seq_length,
                         int(patch_height * patch_size),
                         int(patch_width * patch_size),
                         img_channels])
    img = np.transpose(img, [0, 1, 4, 2, 3])
    return img


if __name__ == '__main__':
    a = np.zeros([128, 10, 1, 25, 25], dtype=np.float32)
    b = np.ones([128, 10, 1, 25, 25], dtype=np.float32) * 255.
    mse = calculate_mse(b, b)
    print(mse)
