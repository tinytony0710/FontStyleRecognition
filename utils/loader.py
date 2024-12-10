from os import listdir
import numpy as np
from PIL import Image
import pandas as pd


threshold=100

def loadImage(path):
    # print(path)
    img = Image.open(path)
    img = img.convert('L')
    img = np.array(img) > threshold
    # print(img.shape)
    img = np.expand_dims(img, -1) # (img_height, img_width, 1)
    return img

def loadImagesInFolder(folderPath):
    print(folderPath)
    imgNames = listdir(folderPath)
    images = []
    for imgName in imgNames:
        # print(imgName)
        imgPath = folderPath + imgName
        # Use you favourite library to load the image
        image = loadImage(imgPath)
        images.append(image)
    
    images = np.array(images)
    return images

def loadDataset(path):
    print(path)
    folderNames = listdir(path)
    images = []
    for label, folderName in enumerate(folderNames):
        folderPath = path + folderName + "/"
        folderImages = loadImagesInFolder(folderPath)
        images.append(folderImages)
    
    images = np.array(images)
    # print(images.shape)
    return images

def loadText(path):
    print(path)
    text = ""
    with open(path,'r') as file:
        text = file.read()
    
    return text

def loadXsl(path):
    print(path)
    df=pd.read_excel(path)
    return df['常用字'].astype(str).sum()
