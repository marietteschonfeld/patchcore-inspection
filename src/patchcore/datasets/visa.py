import os
from enum import Enum

import PIL
import PIL.ImageEnhance
import torch
from torchvision import transforms
from torchvision.transforms.v2.functional import adjust_brightness
import pandas as pd

_CLASSNAMES = ['candle',
               'capsules',
               'cashew',
               'chewinggum',
               'fryum',
               'macaroni1',
               'macaroni2',
               'pcb1',
               'pcb2',
               'pcb3',
               'pcb4',
               'pipe_fryum']


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for VisA.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=256,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize((resize, resize)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ColorJitter(brightness=255),
            transforms.Resize(resize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "Normal"),
            "image_name": "/".join(image_path.split("/")[-5:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            imgpaths_per_class[classname], maskpaths_per_class[classname] = self._find_paths(classname)
            anomaly_types = imgpaths_per_class[classname].keys()

            for anomaly in anomaly_types:
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "Normal":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


    def _find_paths(self, category):
        image_paths, mask_paths = [], []

        split_df = pd.read_csv(os.path.join(self.source, 'split_csv', '1cls.csv'))
        split_df = split_df[split_df["object"] == category]
        split_df = split_df.fillna(value="")
        image_paths, mask_paths = {}, {}
        if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
            split_df = split_df[split_df['split'] == 'train']
            image_paths['Normal'] = list(split_df['image'])
            mask_paths['Normal'] = list(split_df['mask'])
        else:
            split_df = split_df[split_df['split'] == 'test']
            image_paths['Normal'] = list(split_df['image'][split_df['label']=='normal'])
            image_paths['Anomaly'] = list(split_df['image'][split_df['label']=='anomaly'])

            mask_paths['Normal'] = list(split_df['mask'][split_df['label']=='normal'])
            mask_paths['Anomaly'] = list(split_df['mask'][split_df['label']=='anomaly'])

        

        image_paths = {anomaly_type:[os.path.join(self.source, image_path) for image_path in image_paths[anomaly_type]] for anomaly_type in image_paths.keys()}
        mask_paths = {anomaly_type:[os.path.join(self.source, mask_path) if mask_path != "" else None for mask_path in mask_paths[anomaly_type]] for anomaly_type in mask_paths.keys()}

        return image_paths, mask_paths