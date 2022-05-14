import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import *
import numpy as np
import time
from datetime import datetime
import cv2
import os
# fix random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# from sklearn.metrics import *


def init(module, weight_init=None, gain=1, name=None):
    if weight_init is None:
        return module
    if name is not None and 'lstmcell' in name:
        weight_init(module.weight_hh, gain=gain)
        weight_init(module.weight_ih, gain=gain)
        nn.init.constant_(module.bias_hh, 0)
        nn.init.constant_(module.bias_ih, 0)
        return module
    weight_init(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module


class LogUtil():
    def __init__(self, log_path):
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
        self.full_path = os.path.join(log_path, self.time)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        self.loss_path = os.path.join(self.full_path, 'loss.npy')
        self.loss_list = []

        # self.eval_mse_path = self.full_path + '/' + 'eval_mse.npy'
        # self.eval_mse_list = []

        self.report_path = self.full_path + '/report_log.txt'

        self.model_path = self.full_path + '/model.pth'

        self.mem_att_path = self.full_path + '/mem_att'
        if not os.path.exists(self.mem_att_path):
            os.makedirs(self.mem_att_path)

    def record_loss(self, loss):
        self.loss_list.append(loss)
        np.save(self.loss_path, self.loss_list)

    def record_eval_mse(self, eval_mse):
        self.eval_mse_list.append(eval_mse)
        np.save(self.eval_mse_path, self.eval_mse_list)

    def record_report(self, report_str):
        f = open(self.report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print('model has been save to ', self.model_path)

    def record_mem_att(self, mem_att, loc, name):
        _, T, L = mem_att.shape
        pix_size = 5
        show_img = np.zeros([(T * pix_size) * 3, (L * pix_size)], dtype=np.uint8)
        loc2idx = {}
        counter = 0
        for t in range(T):
            if not loc[0, t] in loc2idx:
                loc2idx[loc[0, t]] = counter
                counter += 1

        for t in range(0, 3 * T, 3):
            idx = loc2idx[loc[0, int(t / 3)]]
            show_img[t * pix_size:(t + 1) * pix_size, idx * pix_size:(idx + 1) * pix_size] = 255
        for t in range(1, 3 * T + 1, 3):
            for l in range(L):
                show_img[t * pix_size:(t + 1) * pix_size, l * pix_size:(l + 1) * pix_size] = mem_att[0, int(
                    (t - 1) / 3), l] * 255
        for t in range(2, 3 * T + 2, 3):
            show_img[t * pix_size:(t + 1) * pix_size, :] = 255
        save_path = self.mem_att_path + '/' + name + '.jpg'
        cv2.imwrite(save_path, show_img)

    # def record_mem_att(self, mem_att, loc, name):
    #     _, T, L = mem_att.shape
    #     pix_size = 5
    #     show_img = np.zeros([(T * pix_size) * 3, (L * pix_size)], dtype=np.uint8)
    #
    #     for t in range(0, 3 * T, 3):
    #         show_img[t * pix_size:(t + 1) * pix_size,
    #         loc[0, int(t / 3)] * pix_size:(loc[0, int(t / 3)] + 1) * pix_size] = 255
    #     for t in range(1, 3 * T + 1, 3):
    #         for l in range(L):
    #             show_img[t * pix_size:(t + 1) * pix_size, l * pix_size:(l + 1) * pix_size] = mem_att[0, int(
    #                 (t - 1) / 3), l] * 255
    #     for t in range(2, 3 * T + 2, 3):
    #         show_img[t * pix_size:(t + 1) * pix_size, :] = 255
    #     save_path = self.mem_att_path + '/' + name + '.jpg'
    #     cv2.imwrite(save_path, show_img)


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


def L2Loss(pred, tar, per_batch=False):
    if not per_batch:
        loss = 0.5 * torch.sum((pred - tar) ** 2)
    else:
        loss = 0.5 * torch.sum((pred - tar) ** 2) / pred.shape[0]
    return loss


def MAPELoss(pred, tar, min_value):
    return torch.mean((torch.abs(pred - tar) + min_value) / (torch.abs(pred) + min_value))
