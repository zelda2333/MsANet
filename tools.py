# -*- coding: utf-8 -*-

import argparse
import sys
import time

import pandas as pd

sys.path.append("..")
import numpy as np
from dataset.BraTSDataSet import BraTSValDataSet, BraTSPreDataSet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataset import ConcatDataset
from numba import jit, prange
from numba.typed import List

import archs
import os
from math import ceil
import nibabel as nib
from sklearn.model_selection import KFold
from tqdm import tqdm

arch_names = list(archs.__dict__.keys())


def ET_statistics():
    """Statistics on the amount of pixels for BraTS files"""
    root_path = r'F:\BraTS2020\MICCAI_BraTS2020_TrainingData'
    list_path = r'F:\BraTS2020\MICCAI_BraTS2020_TrainingData\list\train.txt'
    # get training file list
    img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]
    # record ET pixels of each nii
    results = {}
    results['FileName'] = []
    results['Count'] = []
    for item in img_ids:
        # get GroundTruth file path
        GT_file_path = os.path.join(root_path, item, item + '_seg.nii.gz')
        # read .nii.gz file
        labelNII = nib.load(GT_file_path)
        # label shape (240,240,155)
        label = labelNII.get_fdata()
        # sum of ET(Enhancing Tumor) pixels
        ET_mask = (label == 4)
        counts = np.count_nonzero(ET_mask)
        # record statistic
        results['FileName'].append(item)
        results['Count'].append(counts)
    # save to csv
    stats = pd.DataFrame(results, columns=['FileName', 'Count'])

    stats.to_csv('stats.csv')
    print("Done.")


def get_arguments():
    parser = argparse.ArgumentParser(description="ConResNet for 3D medical image segmentation.")
    parser.add_argument("--data-dir", type=str, default='F:/BraTS2020/MICCAI_BraTS2020_TrainingData/',
                        help="Path to the directory containing your dataset.")
    parser.add_argument("--data-list", type=str, default='list/test.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet3d')
    parser.add_argument("--input-size", type=str, default='80,80,80',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument('--deepsupervision', default=True, type=bool)
    parser.add_argument("--center-crop", type=bool, default=False,
                        help="whether to delete background.")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes to predict (ET, WT, TC).")
    parser.add_argument("--restore-from", type=str,
                        default=r'C:\Users\96454\Documents\WeChat Files\wxid_xqtjqrrdkp0022\FileStorage\File\2021-05\Server_Log\80_144_144_noCrop\UNet3d\30750.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--weight-std", type=bool, default=False,
                        help="whether to use weight standarization in CONV layers.")
    return parser.parse_args()


def pad_image(img, target_size):
    """Pad an image up to the target size.将图像填充到目标大小"""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, img_list, tile_size, classes, args):
    image = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_prediction = net(torch.from_numpy(padded_img).cuda())
                if args.deepsupervision:
                    padded_prediction = F.sigmoid(padded_prediction[-1])
                else:
                    padded_prediction = F.sigmoid(padded_prediction)
                padded_prediction = interp(padded_prediction).cpu().data[-1]
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    full_probs = full_probs.numpy().transpose(1, 2, 3, 0)
    return full_probs


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1

    dice = 2 * num / den

    return dice.mean()


def evaluate_ET(predicts, labels, num_folds=5):
    """Evaluates dice scores for ET using KFolds cross validation
    Args:
        predicts: model prediction mask for ET

        labels: ground truth mask for ET

    Returns:
        best_thresholds: Array of thresholds values that had best performing dice scores
        per each fold in cross validation set.
        best_scores: Array of dices scores calculated by a given thresholds
    """
    assert len(predicts) == len(labels), 'The prediction must be the same shape as ground truth!'

    # Calculate dice score metrics
    thresholds = np.arange(0, 1000, 1)
    num_images = len(predicts)
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    best_scores = np.zeros(num_folds)
    best_thresholds = np.zeros(num_folds)

    indices = np.arange(num_images).astype(np.uint8)
    print(f"--------------- {num_folds} folds cross validation ----------------")
    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print("Evaluating for folds ", fold_index+1)
        print("train set:", train_set)
        # Find the best pixel threshold for the k-fold cross validation using the train set
        dice_trainset = np.zeros(num_thresholds)
        # numba parallel loop
        dice_trainset = parallel_loop(num_thresholds, dice_trainset, predicts, labels, train_set)
        # straight running without cpu speeds up
        # for threshold_index, threshold in enumerate(thresholds):
        #     dice_trainset[threshold_index] = calculate_metrics(
        #         threshold=threshold, predict=predicts, gt=labels, index=train_set
        #     )
        best_threshold_index = np.argmax(dice_trainset)

        print("test set:", test_set)
        # Test on test set using the best pixel threshold
        best_scores[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], predict=predicts, gt=labels, index=test_set
        )
        best_thresholds[fold_index] = best_threshold_index

    return best_thresholds, best_scores


@jit(nopython=True)
def calculate_dice_score(seg, label):
    if np.sum(label) == 0 and np.sum(seg) == 0:
        return 1
    elif np.sum(label) == 0 and np.sum(seg) != 0:
        return 0
    else:
        pred = seg.reshape((1, -1))
        target = label.reshape((1, -1))
        num = np.sum(np.multiply(pred, target), axis=1)
        den = np.sum(pred, axis=1) + np.sum(target, axis=1) + 1
        return np.mean(2 * num / den)


@jit(nopython=True, parallel=True, nogil=True)
def parallel_loop(thresholds, dice_trainset, predicts, labels, train_set):
    for threshold in prange(thresholds):
        dice_ET = 0
        for idx in prange(len(train_set)):
            # i = index[idx]
            seg = np.copy(predicts[train_set[idx]])
            if np.count_nonzero(seg) < threshold:
                seg = np.zeros_like(seg)

            dice_ET += calculate_dice_score(seg, label=labels[train_set[idx]])
        dice_trainset[threshold] = dice_ET / len(train_set)
    return dice_trainset


def calculate_ET_score(seg, label):
    if np.sum(label) == 0 and np.sum(seg) == 0:
        return 1
    elif np.sum(label) == 0 and np.sum(seg) != 0:
        return 0
    else:
        pred = seg.reshape((1, -1))
        target = label.reshape((1, -1))
        num = np.sum(np.multiply(pred, target), axis=1)
        den = np.sum(pred, axis=1) + np.sum(target, axis=1) + 1

        return np.mean(2 * num / den)


def calculate_metrics(threshold, predict, gt, index):
    # if pixels less than threshold, then remove the pixels, calculate dice scores
    # filter pixels(performing filter process, to ignore pixel less than thresholds)

    # get dice scores for ET metrics
    dice_ET = 0
    for idx in prange(len(index)):
        # i = index[idx]
        seg = np.copy(predict[index[idx]])
        if np.count_nonzero(seg) < threshold:
            seg = np.zeros_like(seg)

        dice_ET += calculate_ET_score(seg, label=gt[index[idx]])

    return dice_ET / len(index)


def main():
    args = get_arguments()

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    model = archs.__dict__[args.arch](args)

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))

    model.eval()
    model.cuda()

    # use BraTSValDataSet for testing, which contains GT label
    # Concat train dataset and validation dataset
    # dataset = ConcatDataset(
    #     [BraTSPreDataSet(args.data_dir, 'list/train.txt'),
    #      BraTSPreDataSet(args.data_dir, 'list/val.txt')]
    # )
    testloader = data.DataLoader(
        dataset=BraTSPreDataSet(args.data_dir, 'list/val.txt'),
        batch_size=1, shuffle=False, pin_memory=True)
    outputs_dir = 'outputs/' + args.arch + '/'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    predicts_mask = []
    labels_GT = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size, _, _ = batch
            size = size[0].numpy()
            with torch.no_grad():
                output = predict_sliding(model, image.numpy(), input_size, args.num_classes, args)

            seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)
            # get ET prediction mask (binary mask)
            seg_pred_ET = seg_pred_3class[:, :, :, 0]
            # get ET groundTruth mask
            seg_gt = np.asarray(label[0].numpy()[:size[0], :size[1], :size[2]], dtype=np.int)
            seg_gt_ET = seg_gt[0, :, :, :]

            predicts_mask.extend(seg_pred_ET[np.newaxis, :])
            labels_GT.extend(seg_gt_ET[np.newaxis, :])

    predicts_mask = np.stack(predicts_mask)
    labels_GT = np.stack(labels_GT)
    print("Predicts_mask shape:", predicts_mask.shape)
    print("Labels_GT:", labels_GT.shape)
    # evaluate for best thresholds
    print("Starting evaluating...")
    start = time.time()
    best_thresholds, best_ET_scores = evaluate_ET(predicts=predicts_mask, labels=labels_GT)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    print("best_thresholds:", best_thresholds)
    print("best_scores:", best_ET_scores)


if __name__ == '__main__':
    main()
