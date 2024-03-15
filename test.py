import argparse
import sys

sys.path.append("..")
import numpy as np
from dataset.BraTSDataSet import BraTSTestDataSet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import archs
import os
from math import ceil
import nibabel as nib

arch_names = list(archs.__dict__.keys())



def get_arguments():
    parser = argparse.ArgumentParser(description="ConResNet for 3D medical image segmentation.")
    parser.add_argument("--data-dir", type=str, default='F:/BraTS2020/MICCAI_BraTS2020_ValidationData/',
                        help="Path to the directory containing your dataset.")
    parser.add_argument("--data-list", type=str, default='list/test.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UdilateSCA1')
    parser.add_argument("--input-size", type=str, default='80,80,80',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--data_box", type=str, default='240,240,155')#176,192,150
    parser.add_argument("--center_crop", type=str, default='')
    parser.add_argument('--deepsupervision', default=True, type=bool)
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes to predict (ET, WT, TC).")
    parser.add_argument("--restore-from", type=str, default='snapshots/UdilateSCA1/10750.pth',
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
    padded_img = np.pad(img, ((0, 0), (0, 0),(0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, img_list, tile_size, classes,deepsupervision):
    image = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

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
                if deepsupervision:
                    padded_prediction = F.sigmoid(padded_prediction[-1])
                else:
                    padded_prediction = F.sigmoid(padded_prediction)
                padded_prediction = interp(padded_prediction).cpu().data[-1]
                prediction = padded_prediction[0:img.shape[2],0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    full_probs = full_probs.numpy().transpose(1,2,3,0)
    return full_probs


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) +1

    dice = 2*num / den

    return dice.mean()

def main():

    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    h_crop, w_crop, d_crop = map(int, args.data_box.split(','))
    data_box = [h_crop, w_crop, d_crop]

    model = archs.__dict__[args.arch](args)

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))

    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        BraTSTestDataSet(args.data_dir, args.data_list,data_box=data_box),
        batch_size=1, shuffle=False, pin_memory=True)

    outputs_dir = 'outputs_'+args.center_crop+'/' + args.arch + '/'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    dice_ET = 0
    dice_WT = 0
    dice_TC = 0
    for index, batch in enumerate(testloader):
        if data_box!=[240, 240, 155]:
            image, name, affine, index_min, index_max = batch
        else:
            image, name, affine = batch
        affine = affine[0].numpy()
        with torch.no_grad(): # d h w
            output = predict_sliding(model, image.numpy(), input_size, args.num_classes, args.deepsupervision)

        seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)

        seg_pred_ET = seg_pred_3class[:, :, :, 0]
        seg_pred_WT = seg_pred_3class[:, :, :, 1]
        seg_pred_TC = seg_pred_3class[:, :, :, 2]

        seg_pred = np.zeros_like(seg_pred_ET)
        seg_pred = np.where(seg_pred_WT == 1, 2, seg_pred)
        seg_pred = np.where(seg_pred_TC == 1, 1, seg_pred)
        seg_pred = np.where(seg_pred_ET == 1, 4, seg_pred)

        if data_box != [240, 240, 155]:
            pp = np.zeros((155, 240, 240))
            h_min, w_min, d_min = index_min
            pp[d_min:d_min + d_crop, h_min:h_crop + h_min, w_min:w_min + w_crop] = seg_pred
            print('data_box',data_box)

            pp = pp.transpose((1, 2, 0))

            pp = pp.astype(np.int16)

            seg_pred = nib.Nifti1Image(pp, affine=affine)
            seg_save_p = os.path.join(outputs_dir + '%s.nii.gz' % (name[0]))
            nib.save(seg_pred, seg_save_p)
            print('saved', seg_save_p)
        else:
            seg_pred = seg_pred.transpose((1,2,0))
            seg_pred = seg_pred.astype(np.int16)

            seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
            seg_save_p = os.path.join(outputs_dir+'%s.nii.gz' % (name[0]))
            nib.save(seg_pred, seg_save_p)
            print('saved', seg_save_p)


if __name__ == '__main__':
    main()
