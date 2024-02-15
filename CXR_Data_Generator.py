import os
import json
import glob
import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

class DataGenerator(Dataset):
	# Initialize the dataset
	def __init__(self, img_dir='', split_file='', transform, ext='.dicom'):

		self.disease_list = []
		self.img_name_list = []
		self.img_label_list = []
		self.transform = transform
	
		# Read the csv file containing image name and corresponding ground truth labels. First row is the header
		with open(args.split_file, 'r') as split_name:
			lines = split_name.readlines()
			disease_list = lines[0].split("\n")[0].split(",")[2:]
			img_and_label_list = lines[1:]
		
		# Seperate the labels and images, second column is text labels which is not required
		for index in img_and_label_list:
		
			# Get chest x-ray name and find full path for the chest x-ray 
			# img_path = glob.glob(os.path.join(args.img_dir,'**/',(index.split(",")[0]+args.img_ext)), recursive=True)[0] # use this if full path is not known
			img_path = os.path.join(args.img_dir,(index.split(",")[0]+args.img_ext))
			
			# Extract ground truth labels
			arr = index.split(",")
			length = len(arr)-2
			img_label = np.zeros(length)
			for i in range(length):
				img_label[i] = int(arr[i+2])
			# Append the current image path and labels to a list	
			self.img_name_list.append(img_path)
			self.img_label_list.append(img_label)
	# Get item from dataset using index number
	def __getitem__(self, index):

		img_path = self.img_name_list[index]
		img_label = self.img_label_list[index]
		# Extract image name and extension from full path
		img_name = os.path.splitext(os.path.basename(img_path))[0]
		extension = os.path.splitext(os.path.basename(img_path))[1]
		
		# Check image extension if dicom or dcm use read x_ray function to get image data else use PIL
		if (args.img_ext == '.dicom') or (args.img_ext == '.dcm'):
			img = read_xray(img_path)
			image_data = Image.fromarray(img).convert('RGB')
		else:
			image_data = Image.open(img_path).convert('RGB')
			
		# Preprocess image data
		image_data = self.transform(image_data)

		# Return image_data, image_name and their corresponding labels in the form of a tuple
		return (image_data, img_label, img_name)
	# Get the dataset length
	def __len__(self):

		return len(self.img_name_list)
		
#-------------------------------------------------------------------------------- 


		
''' ## Input
# 1. Main Directory (absolute path), if not passed as input in the csv file, where dataset is stored.
# 2. CSV file corresponding to train, val or test set. Format of csv fileis as folow: 1st column image name with absolute or relative path. If only image name is provided, Input 1 is necessary and uncomment line 47 and comment line 48. 2nd column can be blank or can contain class names of abnormalities. from 3rd to last column its a binary value 0 or 1, each value indicating presence or absence of abnormality in that image.
# 3. A transform function containing resize, cropping, number of channel, normalization and other necesorry preprocessing.
# 4. Image extension. If extension is provided in image name pass empty string else pass in format '.jpg' or '.dcm'.
'''

''' ## Output
# 1. Image name
# 2. Tensor corresponding to image data
# 3. Numpy array containg ground truth value.
'''

''' ## How to use the module
# Import some lines (Sore this file in same folder as main file)
from CXR_Data_Generator import DataGenerator
from torch.util.data import DataLoader
from torchvision import transforms

# To create Dataset
dataset = DataGenerator(img_dir="/home/data_path", split_file="text.csv", transform=transforms.Compose(), ext=".png")
dataLoader = DataLoader(dataset, batch_size, shuffle= True)
'''
