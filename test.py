import requests
import os
import base64
import sys
import utility


TKEY= os.environ['TKEY']
PKEY= os.environ['PKEY']
CROPFAILED=False

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
if ('/base/' in CLASSFILEPATH):
	CROPFAILED = True

def create_project(projectName):
	print('Create a new project: {}\n'.format(projectName))
	newprojectupdatemodel = Training.ProjectUpdateModel(name=projectName)
	resp = t.create_project(training_key=TKEY, training_key1=TKEY, project_update_model=newprojectupdatemodel)
	print('Result for creating new project: {}\n'.format(resp))

def delete_project(projectName, projectModel):
	if (projectName in projectModel.name):
		print('Delete project: {}\n', projectModel.name)
		resp = t.delete_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
		print('Result for deleting project: {}\n'.format(resp))

def get_last_iteration(projectModel):
	iterations = t.get_iterations(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
	if (len(iterations) < 2):
		print('This project needs to be trained first. \n')
		train_project(projectModel)

	iterationId = iterations[len(iterations)-2].id
	return iterationId

def train_project(projectModel):
	print('Training project: {}\n', projectModel.name)
	resp = t.train_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
	print('Result from training project: {}\n', resp)

def get_classes(projectModel):
	resp = t.get_classes(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
	# Get all classes
	classes = {}
	for imageClass in resp:
		classId = imageClass.get_classid()
		className = imageClass.get_classname()
		classes[className] = classId
	return classes

def crop_upload_failed(filePath, numOfCrops, class_id, projectModel, fileName):
	# Generate crop images from failed image
	newFilePath = utility.crop_image(filePath, numOfCrops, 'output/', fileName)
	newFileData = open(newFilePath, 'rb').read()

	print ('Uploading image to class: {}] \n'.format(class_id))
	resp = t.post_images_for_class(project_id=projectModel.id, class_id=class_id, image_data=newFileData, training_key=TKEY, training_key1=TKEY)
	print ('Result from uploading image to class: {}] \n'.format(resp))
	return resp.get_isSuccessful()

def eval_plot_test(classes, projectModel, iterationId):
	y_true = []
	y_pred = []
	needRetrain = False
	print ('iteration_id: {}\n'.format(iterationId))

	for className in classes.keys():
		classId = classes[className]

		# For each class, evaluate test files
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

						# If predicted class is incorrect, crop and upload to the right class
						if (CROPFAILED and (predictedClass != className)):
							class_id = classes[className]
							ret = crop_upload_failed(filePath, 1, class_id, projectModel, file)
							if (ret == True):
								needRetrain = True
					i = i+1

	if (needRetrain):
		train_project(projectModel)
		needRetrain = False

	utility.plot_project(y_true, y_pred)


# program starts
print('Get projects:\n')
resp = t.get_projects(training_key=TKEY)
for project in resp:
	projectModel = project
	print('Project: {}\n'.format(projectModel))

	needRetrain = False
	# Evaluate
	if (PROJECTNAME in projectModel.name):
		print('Evaluate against project: {}\n', projectModel.name)
		# Get last iteration for project
		iterationId = get_last_iteration(projectModel)
		print('iterationid: {}\n'.format(iterationId))

		# Process one class at a time
		classes = get_classes(projectModel)
		print('Result for get_classes: {} \n'.format(classes))

		# Eval test images and plot
		eval_plot_test(classes, projectModel, iterationId)



