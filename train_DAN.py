#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : train_DAN.py
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from libs.config.DAN_config import OPTION as opt
from libs.dataset.transform import TestTransform, TrainTransform
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.models.DAN import DAN
from libs.utils.Logger import LogTime, Loss_record, Tee, TimeRecord
from libs.utils.Logger import TreeEvaluation as Evaluation
from libs.utils.loss import cross_entropy_loss, mask_iou_loss
from libs.utils.optimer import DAN_optimizer
from libs.utils.Restore import get_save_dir, restore, save_model


def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')

    # 文件路径设置
    parser.add_argument("--snapshot_dir", type=str, default=opt.SNAPSHOTS_DIR)
    parser.add_argument("--data_path", type=str, default=opt.DATASETS_DIR)
    parser.add_argument("--arch", type=str, default='DAN')
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=4)

    # 载入模型参数
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--save_epoch", type=int, default=5)

    # 输入视频参数
    parser.add_argument("--input_size", type=int, default=opt.TRAIN_SIZE)
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)
    parser.add_argument("--sample_per_class", type=int, default=100)
    parser.add_argument("--vsample_per_class", type=int, default=50)

    # GPU显存相关参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 训练时长相关参数
    parser.add_argument("--step_iter", type=int, default=100)
    parser.add_argument("--max_epoch", type=int, default=101)

    # 验证相关参数
    parser.add_argument("--novalid", action='store_true')

    return parser.parse_args()


def criterion(pred, target, bootstrap=1):
    return [
        cross_entropy_loss(pred, target, bootstrap),
        mask_iou_loss(pred, target)
    ]


def train(args):
    # build models
    print('==> Building Models', args.arch)
    net = eval(args.arch).DAN()
    print('    Total params: %.2fM' %
          (sum(p.numel() for p in net.parameters()) / 1000000.0))

    optimizer = DAN_optimizer(net)
    net = net.cuda()

    if args.restore_epoch > 0:
        restore(args, net)
        print("Resume training...")
        print("Resume_epoch: %d" % (args.restore_epoch))

    print('==> Preparing dataset')
    tsfm_train = TrainTransform(args.input_size)
    tsfm_val = TestTransform(args.input_size)

    # dataloader iteration
    query_frame = args.query_frame
    support_frame = args.support_frame
    traindataset = YTVOSDataset(data_path=args.data_path,
                                query_frame=query_frame,
                                support_frame=support_frame,
                                sample_per_class=args.sample_per_class,
                                transforms=tsfm_train,
                                set_index=args.group)
    validdataset = YTVOSDataset(valid=True,
                                data_path=args.data_path,
                                query_frame=query_frame,
                                support_frame=support_frame,
                                sample_per_class=args.vsample_per_class,
                                transforms=tsfm_val,
                                set_index=args.group)
    train_list = traindataset.get_class_list()
    valid_list = validdataset.get_class_list()

    train_loader = DataLoader(traindataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(validdataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2)

    # set loss
    print('==> Setting Loss')

    losses = Loss_record()
    train_evaluations = Evaluation(class_list=train_list)
    valid_evaluations = Evaluation(class_list=valid_list)

    # set epoch
    start_epoch = args.restore_epoch
    train_iters = len(train_loader)
    val_iters = len(val_loader)
    print('training iters per epoch: ', train_iters)
    print('valid iters per epoch: ', val_iters)
    best_iou = 0
    max_step = int(train_iters / args.step_iter)
    train_time_record = TimeRecord(max_step, args.max_epoch)
    trained_iter = train_iters * start_epoch
    print('Start training')
    for epoch in range(start_epoch, args.max_epoch):
        print('==> Training epoch {:d}'.format(epoch))
        # train
        begin_time = time.time()
        is_best = False
        net.train()

        for iter, data in enumerate(train_loader):
            trained_iter += 1
            query_img, query_mask, support_img, support_mask, idx = data
            # B N C H W
            query_img, query_mask, support_img, support_mask, idx \
                = query_img.cuda(), query_mask.cuda(), support_img.cuda(), support_mask.cuda(), idx.cuda()

            pred_map = net(query_img, support_img, support_mask)
            # ouptut [batch, Frame, 1, 241 425]

            pred_map = pred_map.squeeze(2)
            query_mask = query_mask.squeeze(2)

            few_ce_loss, few_iou_loss = criterion(pred_map, query_mask)
            total_loss = 5 * few_ce_loss + few_iou_loss
            losses.updateloss(total_loss, few_ce_loss, few_iou_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_evaluations.update_evl(idx, query_mask, pred_map)

            if iter % args.step_iter == 0 and iter > 0:
                step_time, remain_time = train_time_record.gettime(
                    epoch, begin_time)
                iou_str = train_evaluations.logiou(epoch, iter)
                loss_str = losses.getloss(epoch, iter)
                print(
                    loss_str, ' | ', iou_str, ' | ',
                    'Step: %.4f s \t Remain: %.4f h' %
                    (step_time, remain_time))
                begin_time = time.time()

        # validation
        if not args.novalid:
            net.eval()
            valid_step = len(val_loader)
            valid_time = LogTime()
            valid_time.t1()
            with torch.no_grad():
                for step, data in enumerate(val_loader):
                    query_img, query_mask, support_img, support_mask, idx = data
                    query_img, query_mask, support_img, support_mask, idx \
                        = query_img.cuda(), query_mask.cuda(), support_img.cuda(), support_mask.cuda(), idx.cuda()
                    pred_map = net(query_img, support_img, support_mask)
                    pred_map = pred_map.squeeze(2)
                    query_mask = query_mask.squeeze(2)
                    valid_evaluations.update_evl(idx, query_mask, pred_map)
            mean_iou = np.mean(valid_evaluations.iou_list)
            valid_time.t2()
            if best_iou < mean_iou:
                is_best = True
                best_iou = mean_iou
            iou_list = ['%.4f' % n for n in valid_evaluations.iou_list]
            strings_iou_list = ' '.join(iou_list)
            print('valid ', valid_evaluations.logiou(epoch, valid_step), ' ',
                  strings_iou_list, ' | ',
                  'valid_time: %.4f s' % valid_time.getalltime(), 'is_best',
                  is_best)

        save_model(args, epoch, net, optimizer, is_best)


if __name__ == '__main__':

    # 读取参数
    args = get_arguments()

    # 创建快照文件夹
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)

    # 创建日志文件
    logger = Tee(os.path.join(args.snapshot_dir, 'train_log.txt'), 'w')

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # 训练模型
    train(args)
