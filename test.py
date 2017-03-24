import requests
import os
import base64
import sys
import urllib2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageDraw

import utility

try:
    import cv
except ImportError:
    print 'Could not import cv, trying opencv'
    import opencv.cv as cv

TKEY= os.environ['TKEY']
PKEY= os.environ['PKEY']

if len(sys.argv) != 3:
	print ('\nUsage: python test.py <directory to test images> <IRIS project name> \n')
	print ('<directory to test images> e.g. /Users/testuser/Documents/testimages/{}/ \n')
	sys.exit()

if TKEY is None or PKEY is None or TKEY == '' or PKEY == '':
	print ('Please set env variables for training and prediction keys. e.g. "export TKEY=xxxxx" and "export PKEY=xxxxxx"')
	sys.exit()

CLASSFILEPATH=sys.argv[1]
print('\nDirectory path to test images: {}\n'. format(CLASSFILEPATH))
PROJECTNAME=sys.argv[2]
print('IRIS project name: {}\n'. format(PROJECTNAME))
var = raw_input("Is this correct? y/n \n")
if var!='y':
	sys.exit()

path = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR_TRAINING = path + '/Training'
ROOT_DIR_PREDICTION = path + '/Prediction'
sys.path.insert(0, ROOT_DIR_TRAINING)
sys.path.insert(0, ROOT_DIR_PREDICTION)
import training as Training
import prediction as Prediction

print('SDK version: {}\n'.format(Training.__version__))
t = Training.training
p = Prediction.prediction

# var = raw_input("Would you like to create a new IRIS project? y/n \n")
# if var=='y':
# 	print('Create a new project: {}\n'.format(PROJECTNAME))
# 	newprojectupdatemodel = Training.ProjectUpdateModel(name=var)
# 	resp = t.create_project(training_key=TKEY, training_key1=TKEY, project_update_model=newprojectupdatemodel)
# 	print('Result for creating new project: {}\n'.format(resp))
y_true = []
y_pred = []

# print('crop images:\n')
# crop_images(CLASSFILEPATH, 2)

print('Get projects:\n')
resp = t.get_projects(training_key=TKEY)
for project in resp:
	projectModel = project
	print('Project: {}\n'.format(projectModel))

# 	# Delete
# 	if ('deleteme' in projectModel.name):
# 		print('Delete project: {}\n', projectModel.name)
# 		resp = t.delete_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
# 		print('Result for deleting project: {}\n'.format(resp))
	
	# Upload images
	#if (PROJECTNAME in projectModel.name):
		#print('Uploading images to project: {}\n'.format(projectModel.name))
		
	# Train
	#if (PROJECTNAME in projectModel.name):
		#print('Training project: {}\n'.format(projectModel.name))

	needRetrain = False
	# Evaluate
	if (PROJECTNAME in projectModel.name):
		print('Evaluate against project: {}\n', projectModel.name)
		# Get last iteration for project
		iterations = t.get_iterations(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
		if (len(iterations) < 2):
			print('This project needs to be trained first. \n')
			print('Training project: {}\n', projectModel.name)
			resp = t.train_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
			print('Result from retrain project: {}\n', resp)

		iterationId = iterations[len(iterations)-3].id
		print('iterationid: {}\n'.format(iterationId))

		# Process one class at a time
		resp = t.get_classes(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
		
		# Get all classes
		classes = {}
		for imageClass in resp:
			classId = imageClass.get_classid()
			className = imageClass.get_classname()
			classes[className] = classId

		print('Result for get_classes: {} \n'.format(classes))

		for className in classes.keys():
			classId = classes[className]
			# For each class, upload files, then evaluate and upload if necessary
			classDirectoryPath = CLASSFILEPATH.format(className)
			for file in os.listdir(classDirectoryPath):
				if file.endswith('.jpg'):
					filePath = os.path.join(classDirectoryPath, file)
					print(filePath)
					data = open(filePath, 'rb').read()
					resp = p.evaluate_image(project_id=projectModel.id, image_data=data, iteration_id=iterationId, prediction_key=PKEY, prediction_key1=PKEY)
					classifications = resp.get_classifications()
					i = 0
					for classification in classifications:
						predictedClass = classification.get_class()
						predictedProb = classification.get_probability()
						#print('Result for evaluate_image: [class: {}, prob: {}] \n'.format(predictedClass, predictedProb))
						if i == 0:
							print('Top result for evaluate_image: [class: {}, prob: {}] \n'.format(predictedClass, predictedProb))
							y_true.append(className)
							y_pred.append(predictedClass)
						i = i+1

		print(metrics.classification_report(y_true, y_pred))
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(y_true, y_pred)
		np.set_printoptions(precision=2)

		# Plot non-normalized confusion matrix
		plt.figure()
		class_names = sorted(set(y_true))
		utility.plot_confusion_matrix(cnf_matrix, classes=class_names,
		                      title='Confusion matrix, without normalization')

		# Plot normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
		                      title='Normalized confusion matrix')

		plt.show()
					# # If predicted class is incorrect, upload to the right class
					# if (predictedClass != className):
					# 	class_id = classes[className]
					# 	# Generate crop images from failed image
					# 	source_image = Image.open(filePath)
					# 	source_width, source_height = source_image.size
					# 	print 'Image was {}x{}'.format(source_width, source_height)

					# 	ratio = round(random.uniform(0.8, 1),2)
					# 	print 'cropping ratio', ratio
					# 	target_width = source_width * ratio
					# 	target_height = source_height * ratio

					# 	target_x1 = (source_width - target_width)/2
					# 	target_y1 = (source_height - target_height)/2

					# 	print 'Image new {}x{}'.format(target_width, target_height)
					# 	coords = (target_x1, target_y1, target_width, target_height)
					# 	print 'Cropping to', coords
					# 	newFilePath = 'output/_new_' + file
					# 	final_image = source_image.crop(coords)
					# 	final_image.save(newFilePath)
					# 	newFileData = open(newFilePath, 'rb').read()

					# 	print ('Uploading image to class: {}] \n'.format(class_id))
					# 	resp = t.post_images_for_class(project_id=projectModel.id, class_id=class_id, image_data=newFileData, training_key=TKEY, training_key1=TKEY)
					# 	print ('Result from uploading image to class: {}] \n'.format(resp))
					# 	if (resp.get_isSuccessful()):
					# 		needRetrain = True
		# if (needRetrain):
		# 	print('Retrain project: {}\n', projectModel.name)
		# 	resp = t.train_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
		# 	print('Result from retrain project: {}\n', resp)
		# 	needRetrain = False



