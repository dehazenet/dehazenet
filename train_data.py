# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'trainlist.txt'
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] for i in haze_names]
            tr_names = [i.split('_')[1] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.tr_names = tr_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index] + '_' + self.tr_names[index]

        haze_img = Image.open(self.train_data_dir + 'hazy/' + haze_name)

        try:
            gt_img = Image.open(self.train_data_dir + 'trans/' + gt_name + '.jpg')
        except:
            gt_img = Image.open(self.train_data_dir + 'trans/' + gt_name + '.png')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        haze_crop_img = np.array(haze_crop_img)
        npad = ((7,8), (7,8), (0,0))
        haze_crop_img = np.pad(haze_crop_img, npad, 'symmetric')
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

