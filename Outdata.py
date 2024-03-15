#-*- coding: utf-8 -*-
from tensorboard.backend.event_processing import event_accumulator
from collections import OrderedDict
from glob import glob
import pandas as pd

dir = r'F:\Bra\snapshot'
#dir = r'F:\Git_Workspace\BraTS segment\ConResNet-main\snapshots'
arch = r'\UNetASPP5ASPP1_0_DSP'
path = dir + arch + '\events.out.tfevents.1625836372.node4'
# 加载日志数据
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
print(ea.scalars.Keys())

seg_score = OrderedDict()
seg_score['iter'] = []
# seg_score['Val_ET_Dice'] = []
# seg_score['Val_WT_Dice'] = []
# seg_score['Val_TC_Dice'] = []
seg_score['loss'] = []
steps = []


def ValDice(val_psnr, name):
    # if name == 'val_ET':
    #     for i in val_psnr:
    #         step = i.step
    #         steps.append(step)
    #         seg_score['iter'].append(step)
    #         value = i.value
    #         seg_score['Val_ET_Dice'].append(value)
    # elif name == 'val_WT':
    #     for i in val_psnr:
    #         value = i.value
    #         seg_score['Val_WT_Dice'].append(value)
    # elif name == 'val_TC':
    #     for i in val_psnr:
    #         value = i.value
    #         seg_score['Val_TC_Dice'].append(value)
    # else:
    for i in val_psnr:
        step = i.step
        steps.append(step)
        seg_score['iter'].append(step)
        value = i.value
        seg_score['loss'].append(value)


# val_ET = ea.scalars.Items('Val_ET_Dice')
# val_ET_dice = ValDice(val_ET, 'val_ET')
#
# val_WT = ea.scalars.Items('Val_WT_Dice')
# val_WT_dice = ValDice(val_WT, 'val_WT')
#
# val_TC = ea.scalars.Items('Val_TC_Dice')
# val_TC_dice = ValDice(val_TC, 'val_TC')

val_loss = ea.scalars.Items('loss')
Loss = ValDice(val_loss, 'loss')

seg_data = pd.DataFrame(seg_score, index=steps)
seg_data.to_csv(dir + arch + '/loss.csv', index=False)
print('saved at:', dir + arch + '/loss.csv')
