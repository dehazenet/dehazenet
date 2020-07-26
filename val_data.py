# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] for i in haze_names]
            tr_names = [i.split('_')[1] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.tr_names = tr_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index] + '_' + self.tr_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'trans/' + gt_name)

        haze_img = np.array(haze_img)
        npad = ((7,8), (7,8), (0,0))
        haze_img = np.pad(haze_img, npad, 'symmetric')

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
