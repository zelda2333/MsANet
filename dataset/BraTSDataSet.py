import os.path as osp
import numpy as np
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize

'''
    获得 nii w,h,d的边界位置，margin为预留区域参数，如：margin=3，表示有像素点周围预留三个像素点距离。
'''


def get_box(image, margin):
    shape = image.shape
    nonindex = np.nonzero(image)  # 返回的是3个数组，分别对应三个维度的下标。

    margin = [margin] * len(shape)

    index_min = []
    index_max = []

    for i in range(len(shape)):
        index_min.append(nonindex[i].min())
        index_max.append(nonindex[i].max())

    # 扩大margin个区域
    for i in range(len(shape)):
        index_min[i] = max(index_min[i] - margin[i], 0)
        index_max[i] = min(index_max[i] + margin[i], shape[i] - 1)

    return index_min, index_max


'''
    获得 nii 想要裁切的w,h,d边界位置
    data_box为想要裁切的大小
'''


def make_box(image, index_min, index_max, data_box):
    shape = image.shape

    for i in range(len(shape)):

        # print('before index[%s]: '%i, index_min[i], index_max[i])

        # 按照data_box扩大或缩小box。
        mid = (index_min[i] + index_max[i]) / 2
        index_min[i] = mid - data_box[i] / 2
        index_max[i] = mid + data_box[i] / 2

        flag = index_max[i] - shape[i]
        if flag > 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        flag = index_min[i]
        if flag < 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        # print('index[%s]: '%i, index_min[i], index_max[i])

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]

        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

        # print('after index[%s]: '%i, index_min[i], index_max[i])
    return index_min, index_max


# 按照box裁切图像。
def crop_with_box(image, index_min, index_max):
    # return image[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0] + x, image.shape[1], image.shape[2]))
        img[x // 2:image.shape[0] + x // 2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1] + y, img1.shape[2]))
        img[:, y // 2:image.shape[1] + y // 2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2] + z))
        img[:, :, z // 2:image.shape[2] + z // 2] = img2[:, :, :]

    return img[
        np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]


class BraTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(128, 160, 200), data_box=[176, 192, 150], scale=True,
                 mirror=True,
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.data_box = data_box
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])
        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        label = labelNII.get_data()

        if self.data_box != [240, 240, 155]:
            # 按照flair确定裁剪区域
            box_min, box_max = get_box(flair, 0)
            index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

            # center crop
            flair = crop_with_box(flair, index_min, index_max)
            t1 = crop_with_box(t1, index_min, index_max)
            t1ce = crop_with_box(t1ce, index_min, index_max)
            t2 = crop_with_box(t2, index_min, index_max)
            label = crop_with_box(label, index_min, index_max)

        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)
        else:
            scaler = 1
        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        d_off = random.randint(0, img_d - scale_d)
        h_off = random.randint(15, img_h - 15 - scale_h)
        w_off = random.randint(10, img_w - 10 - scale_w)

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W

        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if self.scale:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True,
                           preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()


class BraTSValDataSet(data.Dataset):
    def __init__(self, root, list_path, data_box=[176, 192, 150]):
        self.root = root
        self.list_path = list_path
        self.data_box = data_box
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())

        label = labelNII.get_data()

        if self.data_box !=[240, 240, 155]:
            # 按照flair确定裁剪区域
            box_min, box_max = get_box(flair, 0)
            index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

            # center crop
            flair = crop_with_box(flair, index_min, index_max)
            t1 = crop_with_box(t1, index_min, index_max)
            t1ce = crop_with_box(t1ce, index_min, index_max)
            t2 = crop_with_box(t2, index_min, index_max)
            label = crop_with_box(label, index_min, index_max)

        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150

        name = datafiles["name"]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy()


class BraTSTestDataSet(data.Dataset):
    def __init__(self, root, list_path,data_box=[176, 192, 150]):
        self.root = root
        self.list_path = list_path
        self.data_box = data_box
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())

        if self.data_box !=[240, 240, 155]:
            # 按照flair确定裁剪区域
            box_min, box_max = get_box(flair, 0)
            index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

            # center crop
            flair = crop_with_box(flair, index_min, index_max)
            t1 = crop_with_box(t1, index_min, index_max)
            t1ce = crop_with_box(t1ce, index_min, index_max)
            t2 = crop_with_box(t2, index_min, index_max)

        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        name = datafiles["name"]
        affine = flairNII.affine

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        image = image.astype(np.float32)
        if self.data_box != [240, 240, 155]:
            return image.copy(), name, affine, index_min, index_max
        else:
            return image.copy(), name, affine




class BraTSPreDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0] + '/' + osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        name = datafiles["name"]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        size = image.shape[1:]
        affine = labelNII.affine

        return image.copy(), label.copy(), np.array(size), name, affine
