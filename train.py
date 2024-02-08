# -*- coding: utf-8 -*-

#Imports
import sys
from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np
import random
import shutil
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

#Take in the arguments for this script
#The user picked the project folder, image resize infox and modelType
PATH = sys.argv[1]

newX = int(sys.argv[2])
newY = int(sys.argv[3])

modelType = sys.argv[4]
modelType_uri = ''
if modelType == 'efficientdet-lite0':
  modelType_uri = 'https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1'
if modelType == 'efficientdet-lite1':
  modelType_uri = 'https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1'
if modelType == 'efficientdet-lite2':
  modelType_uri = 'https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1'
if modelType == 'efficientdet-lite3':
  modelType_uri = 'https://tfhub.dev/tensorflow/efficientdet/lite3/feature-vector/1'
if modelType == 'efficientdet-lite4':
  modelType_uri = 'https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/1'

#Delete non-image file
#This file may exist in the exported data from kili-technology
if os.path.isfile(PATH+'/images/remote_assets.csv'):
    os.remove(PATH+'/images/remote_assets.csv')

#resize the images
for filename in os.listdir(PATH+"/images/"):
  if filename.endswith("jpg"):
      img_org = Image.open(PATH+"/images/"+filename)
      newsize = (newX,newY)
      img_org = img_org.resize(newsize)
      img_org.save(PATH+"/training/images/"+filename)

#keep track of all the lables used in the data
labels = {}
labelNumber = 0

#adjust pascal voc files to match the resized images
for filename in os.listdir(PATH+"/labels/"):
  tree = ET.parse(PATH+"/labels/"+filename)
  root = tree.getroot()

  oldX = float(root.find('size').find('width').text)
  oldY = float(root.find('size').find('height').text)

  scaleX = newX/oldX
  scaleY = newY/oldY

  root.find('size').find('width').text = str(newX)
  root.find('size').find('height').text = str(newY)

  root.find('filename').text = root.find('filename').text[:-3]+'jpg'

  for member in root.findall('object'):
    labelName = member.find('name').text
    found = False
    for key in labels:
      value = labels.get(key)
      if(value == labelName):
        found = True
        break
    if(not found):
      labelNumber += 1
      labels.update({labelNumber:labelName})

    bndbox = member.find('bndbox')

    xmin = bndbox.find('xmin')
    ymin = bndbox.find('ymin')
    xmax = bndbox.find('xmax')
    ymax = bndbox.find('ymax')

    xmin.text = str(int(np.round(float(xmin.text) * scaleX)))
    ymin.text = str(int(np.round(float(ymin.text) * scaleY)))
    xmax.text = str(int(np.round(float(xmax.text) * scaleX)))
    ymax.text = str(int(np.round(float(ymax.text) * scaleY)))

  tree.write(PATH+"/training/Annotations/"+filename)

#Split the data randomly into the training data, validation data, and testing data
image_paths = os.listdir(PATH+'/images/')
random.shuffle(image_paths)

for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.2) and i > int(len(image_paths) * 0.1):
    #move to validation
    shutil.move(PATH+'/training/images/'+image_path, PATH+'/validation/images/')
    shutil.move(PATH+'/training/Annotations/'+image_path.replace("jpg", "xml"), PATH+'/validation/Annotations')
  elif i < int(len(image_paths) * 0.2):
    #move to test
    shutil.move(PATH+'/training/images/'+image_path, PATH+'/testing/images/')
    shutil.move(PATH+'/training/Annotations/'+image_path.replace("jpg", "xml"), PATH+'/testing/Annotations')


### Train the object detection model ###

# Load Datasets
train_data = object_detector.DataLoader.from_pascal_voc(images_dir=PATH+'/training/images',annotations_dir=PATH+'/training/Annotations',label_map=labels)
validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=PATH+'/validation/images',annotations_dir=PATH+'/validation/Annotations',label_map=labels)
test_data = object_detector.DataLoader.from_pascal_voc(images_dir=PATH+'/testing/images',annotations_dir=PATH+'/testing/Annotations',label_map=labels)

# Load model spec
spec = object_detector.EfficientDetSpec(
  model_name=modelType,
  uri=modelType_uri,
  hparams={'max_instances_per_image': 25})

# Train the model
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, epochs=2, validation_data=validation_data)

# Evaluate the model
eval_result = model.evaluate(test_data)

# Print COCO metrics
print("COCO metrics before converting to TFLite:")
for label, metric_value in eval_result.items():
    print(f"{label}: {metric_value}")

# Add a line break after all the items have been printed
print()

# Export the model
model.export(export_dir=PATH, export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

# Evaluate the tflite model
tflite_eval_result = model.evaluate_tflite(PATH+'/model.tflite', test_data)

# Print COCO metrics for tflite
print("COCO metrics for TFLite:")
for label, metric_value in tflite_eval_result.items():
    print(f"{label}: {metric_value}")


print('Training and exporting is complete')
print('Your model is in your google drive at '+PATH+'/model.tflite')
