import os
from PIL import Image
from resizeimage import resizeimage

mainDirectory = os.path.abspath('./BottleNotBottleDataset')

def resize(mainDirectory, folder):
    directory = os.path.abspath(mainDirectory+'/'+folder)
    for folder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory,folder)):
            with open(os.path.join(directory,folder,file), 'rb') as f:
                with Image.open(f) as image:
                    try:
                        cover = resizeimage.resize_cover(image, [28, 28])
                        cover.save(os.path.join(directory,folder,file), image.format)
                    except:
                        print('Error on ' + file)   
                    
for folder in os.listdir(mainDirectory):
    print(mainDirectory,folder)
    resize(mainDirectory,folder)
    
