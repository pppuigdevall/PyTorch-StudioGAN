from torch.utils.data import Dataset
from torchvision.transforms import transforms
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
        df_name =dataset_path.split("/")[-2]
        root_path = dataset_path.split(df_name)[-2]
        csv_path = os.path.join(root_path, df_name)
        csv_path = os.path.join(root_path, df_name + ".csv")
        self.df = pd.read_csv(csv_path)
        self.df['prod_name_embedding'] = self.df['prod_name_embedding'].apply(ast.literal_eval)
        if train:
            self.df = self.df[self.df["split"]=="train"]
        else:
            self.df = self.df[self.df["split"]=="val"]
        print(self.df["s3_path"].iloc[0])
        
    def __getitem__(self, index):
        img = Image.open(self.df["s3_path"].iloc[index]).convert("RGB")
        embedding = self.df["prod_name_embedding"].iloc[index]
        embedding = torch.Tensor(embedding)
        label = self.df["class_label"].iloc[index]
        return img, embedding, label

    def __len__(self):
        return self.df.shape[0]