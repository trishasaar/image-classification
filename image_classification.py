# -*- coding: utf-8 -*-
# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
from pathlib import Path
import os
import operator

direc = r'C:\Users\trish\AnacondaProjects\Image Classification\Photos'

# Defining a function that will do the detections
def detect(frame, net, transform): #inputs: a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    frame_t = transform(frame)[0] # applying the transformation
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # converting frame to a torch tensor
    x = Variable(x.unsqueeze(0)) # adding a fake dimension to the batch 
    y = net(x) # feeding input to the nueral image
    detections = y.data # creating the detections tensor contained in the output y.
    d = {} # list to choose most reoccuring object in image
    for i in range(detections.size(1)): #looping through each object class
        occurences = 0 # counting occurences of each class with the while loop
        while detections[0, i, occurences, 0] >= 0.6: 
            occurences += 1 
        d[labelmap[i-1]] = occurences
    detect.label = str(max(d.items(), key=operator.itemgetter(1))[0]) # categorizes image according to most frequently occuring object in the image
    return frame

# Creating the SSD neural network
net = build_ssd('test') 
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

pathList = Path(direc).glob('**/*.jpg')

for currentPath in pathList: 
    baseName = os.path.basename(str(currentPath))   
    image = imageio.imread(str(currentPath)) # converting currentPath (object) to a string 
    try:
        frame = detect(image, net.eval(), transform) 
        photoFolder = direc + '\\' + detect.label
    except ValueError:    
        photoFolder = direc + '\misc' #for images that cannot be classified to a type

    if not os.path.exists(photoFolder): #creates a folder to categorize the image if one has not been created already
        os.makedirs(photoFolder)
    photoDestination = photoFolder + '\\' + baseName        
    writer = imageio.get_writer(photoDestination)
    writer.append_data(frame)
    writer.close()
    os.remove(str(currentPath))