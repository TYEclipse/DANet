#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : train_DAN.py
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.config.DAN_config import OPTION as opt
from libs.dataset.transform import TestTransform, TrainTransform
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.models.DAN.DAN import DAN
from libs.utils.Logger import Logger, Loss_record, TimeRecord
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
    parser.add_argument("--valid_epoch", type=int, default=5)
    parser.add_argument("--valid_interval", type=int, default=5)

    # 验证相关参数
    parser.add_argument("--novalid", action='store_true')

    return parser.parse_args()


def criterion(pred, target, bootstrap=1):
    return [
        cross_entropy_loss(pred, target, bootstrap),
        mask_iou_loss(pred, target)
    ]


# 验证
def valid_epoch(model: nn.Module, data_loader: DataLoader,
                device: torch.device, valid_evaluation: Evaluation):
    model.eval()
    for i, (query_img, query_mask, support_img, support_mask,
            idx) in enumerate(data_loader):
        query_img = query_img.to(device)
        query_mask = query_mask.to(device)
        support_img = support_img.to(device)
        support_mask = support_mask.to(device)
        idx = idx.to(device)
        with torch.no_grad():
            pred_map = model(query_img, support_img, support_mask)
            pred_map = pred_map.squeeze(2)
            query_mask = query_mask.squeeze(2)
            valid_evaluation.update_evl(idx, query_mask, pred_map)


def train(args, logger: Logger):
    # build models
    logger.info('Building model...')
    net = DAN()
    total_params = sum(p.numel() for p in net.parameters())
    logger.info('Total number of parameters: {}M'.format(total_params / 1e6))

    optimizer = DAN_optimizer(net)
    net = net.cuda()

    if args.restore_epoch > 0:
        restore(args, net)
        logger.info('Restore model from epoch {}'.format(args.restore_epoch))

    logger.info('Loading data...')
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
    logger.info('Setting loss...')

    losses = Loss_record()
    train_evaluations = Evaluation(class_list=train_list)
    valid_evaluations = Evaluation(class_list=valid_list)

    # set epoch
    start_epoch = args.restore_epoch
    train_iters = len(train_loader)
    logger.info('Training {} epochs with {} iters per epoch'.format(
        args.max_epoch, train_iters))
    best_iou = 0
    train_time_record = TimeRecord(max_epoch=args.max_epoch,
                                   max_iter=train_iters)
    for epoch in range(start_epoch, args.max_epoch):
        logger.info('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
        # train
        is_best = False
        net.train()

        for iter, data in enumerate(train_loader):
            query_img, query_mask, support_img, support_mask, idx = data
            # B N C H W
            query_img = query_img.cuda()
            query_mask = query_mask.cuda()
            support_img = support_img.cuda()
            support_mask = support_mask.cuda()
            idx = idx.cuda()

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

            if iter % args.step_iter == 1 and iter > 0:
                total_time_str, remain_time_str = train_time_record.get_time(
                    epoch, iter)
                iou_str = train_evaluations.logiou()
                loss_str = losses.getloss()
                logger.info('[Train:{}/{}] Step: {}/{} Time: {}/{} '
                            'IOU: {} Loss: {}'.format(epoch,
                                                      args.max_epoch - 1, iter,
                                                      train_iters,
                                                      total_time_str,
                                                      remain_time_str, iou_str,
                                                      loss_str))

        # validation
        if not args.novalid and epoch % args.valid_interval == 0:
            mean_iou_list = []
            iou_lists = []
            for _ in range(args.valid_epoch):
                valid_epoch(net, val_loader, torch.device('cuda'),
                            valid_evaluations)
                mean_iou_list.append(valid_evaluations.iou_list)
                iou_lists.append(valid_evaluations.iou_list)

            mean_iou = np.mean(mean_iou_list)
            iou_list = np.mean(iou_lists, axis=0)
            iou_str = ' '.join(['%.4f' % n for n in iou_list])
            if best_iou < mean_iou:
                is_best = True
                best_iou = mean_iou
            logger.info('[Valid:{}/{}] Mean IOU: {:.4f} IOU: {}'.format(
                epoch, args.max_epoch - 1, mean_iou, iou_str))

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
    count = 1
    log_file = os.path.join(args.snapshot_dir,
                            'train_log_{}.txt'.format(count))
    while os.path.exists(log_file):
        count += 1
        log_file = os.path.join(args.snapshot_dir,
                                'train_log_{}.txt'.format(count))
    print('log file: {}'.format(log_file))
    logger = Logger(log_file)
    logger.info('Running parameters:')
    logger.info(str(args))

    # 训练模型
    train(args, logger)
