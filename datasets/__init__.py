import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import OneHotEncoder
from os import listdir, mkdir
from os.path import exists

from utils.loader import loadDataset



font_size=96

threshold=100


def labeledDataset(dataset):
    n = len(dataset)
    X = np.concatenate(dataset, axis=0)
    X = np.moveaxis(X, -1, 1)
    Y = np.concatenate([np.full([len(dataset[i])], i) for i in range(n)], axis=0)
    oneshot_encoder = OneHotEncoder(sparse_output=False)
    Y = oneshot_encoder.fit_transform(np.expand_dims(Y, -1))
    return X, Y

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, device, dtype = torch.float32):
        self._X = torch.tensor(X, dtype=dtype, device=device)
        self._Y = torch.tensor(Y, dtype=dtype, device=device)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._Y[idx]


def extract_font(font, char, img_shape):
    l, u, r, d= font.getbbox(char)
    # print(l, u, r, d)
    image = Image.new('L', img_shape)
    draw = ImageDraw.Draw(image)
    draw.text((0, -u+(font_size-(d-u))/2), char, font=font, fill=255)
    # display(image)
    return np.array(image)

def fontImageGenerator(sentence, sourcePath, targetPath, img_shape):

    if(exists(targetPath)):
        images=loadDataset(targetPath)
        return images

    # fetch font info
    fontNames = listdir(sourcePath)
    fontPaths = [sourcePath+fontName for fontName in fontNames]
    # print(fontPaths)
    fonts = [ImageFont.truetype(font, font_size) for font in fontPaths]

    # transform into image(train)
    images = [[extract_font(font,char,img_shape)>threshold for char in sentence]for font in fonts]

    # save images
    paths = targetPath.split('/')
    path = ""
    for i in range(len(paths)):
        path += paths[i]+'/'
        if(not exists(path)):
            mkdir(path)
        
    
    for i, fontImages in enumerate(images):
        font = fontNames[i][:-4]
        font = targetPath + font
        mkdir(font)
        # print(font)
        for j, charImage in enumerate(fontImages):
            Image.fromarray(charImage).save(f"{font}/{sentence[j]}.jpg")
            print(f"{font}/{sentence[j]}.jpg")
        
    images = np.expand_dims(images, axis=-1)
    return images