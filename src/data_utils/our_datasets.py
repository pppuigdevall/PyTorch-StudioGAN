from torch.utils.data import Dataset
from torchvision.transforms import transforms
import warnings
from PIL import Image
import pandas as pd
import os
import torch
import ast
import PIL

PIL.Image.LOAD_TRUNCATED_IMAGES = True # Otherwise we got ValueError: Decompressed data too large
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class FoodiMLDataset(Dataset):
    def __init__(self, dataset_path, train):
        # dataset_path: "/home/ec2-user/SageMaker/dataset/spanish_subset/"
        
        """
        df_name = dataset_path.split("/")[-2]
        root_path = dataset_path.split(df_name)[-2]
        csv_path = os.path.join(root_path, df_name + "_train" + ".csv")
        self.df = pd.read_csv(csv_path)
        """
        
        self.df = pd.read_csv("/home/ec2-user/SageMaker/dataset/spanish_subset_pizza_train_new_data_def.csv")
        
        print("dataset shape ", self.df.shape)
        print(self.df.columns)
        
        if train:
            self.df = self.df[self.df["split"]!="val"]
        else:
            self.df = self.df[self.df["split"]=="val"]
        
        self.labels = self.df["class_label"].values
        print(self.df["s3_path"].iloc[0])
        print(self.df.columns)
        warnings.simplefilter("ignore")
    def __getitem__(self, index):
        with warnings.catch_warnings():
            img = Image.open(self.df["s3_path"].iloc[index]).convert("RGB")
        label = self.df["class_label"].iloc[index]
        return img, label

    def __len__(self):
        return self.df.shape[0]