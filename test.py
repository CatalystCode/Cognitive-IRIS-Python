import requests
import os
import base64
import sys

TKEY= os.environ['TKEY']
PKEY= os.environ['PKEY']

if len(sys.argv) == 1:
	print ('Missing argument for directory path with pattern to test images. e.g. /Users/testuser/Documents/testimages/{}/')
	sys.exit()

if TKEY is None or PKEY is None or TKEY == '' or PKEY == '':
	print ('Please set env variables for training and prediction keys. e.g. "export TKEY=xxxxx" and "export PKEY=xxxxxx"')
	sys.exit()

CLASSFILEPATH=sys.argv[1]
print('Directory path to test images: {}'. format(CLASSFILEPATH))
print('Press enter to continue...')
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

print('version: {}\n'.format(Training.__version__))
t = Training.training
p = Prediction.prediction

print('Get projects:\n')
resp = t.get_projects(training_key=TKEY)
for project in resp:
	projectModel = project
	print('{}\n'.format(projectModel))

	# Delete
	if ('deleteme' in projectModel.name):
		print('Delete project: {}\n', projectModel.name)
		resp = t.delete_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
		print('Result for deleting project: {}\n'.format(resp))
	
	# Upload images
	if ('upload' in projectModel.name):
		print('Uploading images to project: {}\n', projectModel.name)
		
	# Train
	if ('upload' in projectModel.name):
		print('Training project: {}\n', projectModel.name)

	needRetrain = False
	# Evaluate
	if ('ritatest' in projectModel.name):
		print('Evaluate against project: {}\n', projectModel.name)
		# TODO: Get current iteration from project
		iterationId = '7d95c074-224b-442c-809b-de11c8930ff3'

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
					predictedClass = classifications[0].get_class()
					predictedProb = classifications[0].get_probability()
					print('Result for evaluate_image: [class: {}, prob: {}] \n'.format(predictedClass, predictedProb))
					# If predicted class is incorrect, upload to the right class
					if (predictedClass != className):
						class_id = classes[predictedClass]
						print ('Uploading image to class: {}] \n'.format(class_id))
						resp = t.post_images_for_class(project_id=projectModel.id, class_id=class_id, image_data=data, training_key=TKEY, training_key1=TKEY)
						print ('Result from uploading image to class: {}] \n'.format(resp))
						if (resp.get_isSuccessful()):
							needRetrain = True
		if (needRetrain):
			print('Retrain project: {}\n', projectModel.name)
			resp = t.train_project(project_id=projectModel.id, training_key=TKEY, training_key1=TKEY)
			print('Result from retrain project: {}\n', resp)

var = raw_input("Would you like to create a new IRIS project? y/n \n")
if var!='y':
	sys.exit()

var = raw_input("Please provide a name for your new project: \n")
if var=='':
	sys.exit()

print('Create a new project: {}\n'.format(var))
newprojectupdatemodel = Training.ProjectUpdateModel(name=var)
resp = t.create_project(training_key=TKEY, training_key1=TKEY, project_update_model=newprojectupdatemodel)
print('Result for creating new project: {}\n'.format(resp))


