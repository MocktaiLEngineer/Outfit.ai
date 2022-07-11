import os
import pathlib
import shutil
from outfitai import app, db, celery
from celery.result import AsyncResult
from flask import Flask, render_template, request, url_for, redirect, session, flash, jsonify
from flask_login import login_user,login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from outfitai.models import *
from PIL import Image
import numpy as np
import pickle
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from collections import OrderedDict
import collections
import pandas as pd
from annoy import AnnoyIndex
from time import sleep


# Point to the Fashion Classifier Model
clothingClassifierCheckpoint = 'fashion-classifier.pth'
fashion_classifier_model = None
rec_model = None

# Point to the Recommender Model
recommenderCheckpoint = 'recommender-model.pth'
recommender_model = None


@app.route("/")
def index():
	return render_template('index.html')

@app.route("/login")
def login():
	return render_template('login.html')

@login_manager.unauthorized_handler
def unauthorized_callback():
	return redirect('/login?next=' + request.path)

@app.route("/logout")
@login_required
def logout():
	logout_user()
	return redirect(url_for('login'))

@app.route("/signup", methods=['POST'])
def signup():
	if request.method == 'POST':
		Username = request.form.get('name')
		Email = request.form.get('email')
		Password = request.form.get('password')

		usernameExists = Users.query.filter_by(Username=Username).first()

		if usernameExists:
			flash("This username is already taken! Please choose a different username.","error")
			return redirect(url_for('login'))	

		userExists = Users.query.filter_by(Email=Email).first()

		if userExists:
			flash("You are already registered, please login instead.", "error")
			return redirect(url_for('login'))

		new_user = Users(Username,Email,Password)
		db.session.add(new_user)
		db.session.commit()	
		flash("Registered successfully! Please login now.","success")
		return render_template('login.html')
		
	return render_template('login.html')

@app.route("/signin" , methods=['POST'])
def signin():
	if request.method == 'POST':
		Email = request.form.get('email')
		Password = request.form.get('password')
		user = Users.query.filter_by(Email=Email).first()

		if user is not None and not user.check_password(Password):
			flash("Invalid username/password","error")
			return redirect(url_for('login'))
		
		login_user(user)	
		return redirect(url_for('home'))
	
	return render_template('login.html')

@app.route("/home")
@login_required
def home():

	print("82", current_user.UserId)
	clothes = UsersCloset.query.filter(UsersCloset.UserId == current_user.UserId).with_entities(UsersCloset.PredictedCategory, UsersCloset.ImageFile).all()
	
	print("86",clothes)

	global garments_dict
	garments_dict = {}

	global UserPreferencesExists
	UserPreferencesExists = UsersPreferences.query.filter(UsersPreferences.UserId == current_user.UserId).first() # TODO - Get the count of UserPreferences record and match it with the number of questions 

	global dic
	dic = {}
	for i, j in clothes:
	    dic.setdefault(i,[]).append(j)
	print("87",dic)

	allEvents = db.session.query(Events.event).all()
	print("115", allEvents)

	EventsList = []
	for event in allEvents:
		EventsList.append(event[0])

	print("121", EventsList)

	LengthOfEvents = len(EventsList)

	print("123",LengthOfEvents)

	return render_template('home.html', clothes_data = dic, user = str(current_user.UserId), garments = garments_dict, UserPreferencesExists = UserPreferencesExists, Events = EventsList, LengthOfEvents = LengthOfEvents)


@app.route("/updateProfile",methods = ['POST'])
@login_required
def updateProfile():
	if request.method == 'POST':
		Username = request.form.get('username')
		FirstName = request.form.get('first_name')
		LastName = request.form.get('last_name')
		Email = request.form.get('email')
		ProfileImage = request.files['profile_photo']
		Gender = request.form['gender']

		id = current_user.UserId

		if ProfileImage:
			profileImagePath = os.path.join(app.root_path, 'static', 'upload',str(id),'profile_pic')
			pathlib.Path(profileImagePath).mkdir(parents=True, exist_ok=True)
			ProfileImage.save(os.path.join(profileImagePath,secure_filename(ProfileImage.filename)))
			
		user = Users.query.get(id)
		print("85", user)
		user.Username = Username
		user.FirstName = FirstName
		user.LastName = LastName
		user.Email = Email
		user.ProfilePhoto = ProfileImage.filename
		user.Gender = Gender

		db.session.commit()	
		return redirect(url_for('home'))
		
	return redirect(url_for('home'))

@app.route("/submitUserPreferences",methods = ['POST'])
@login_required
def submitUserPreferences():
	if request.method == 'POST':
		party_choices = request.form.getlist('party-choices')
		beach_choices = request.form.getlist('beach-choices')
		wedding_choices = request.form.getlist('wedding-choices')
		dinner_choices = request.form.getlist('dinner-choices')

		print("174",party_choices)
		print("175",beach_choices)
		print("176",wedding_choices)
		print("177",dinner_choices)

		preferencesPath = os.path.join(app.root_path, 'static', 'upload',str(current_user.UserId),'preferences')
		pathlib.Path(preferencesPath).mkdir(parents=True, exist_ok=True)

		choices = []

		choices.append(party_choices)
		choices.append(beach_choices)
		choices.append(wedding_choices)
		choices.append(dinner_choices)

		for event in choices:
			for file in event:
				file_path = os.path.join(app.root_path, 'static','img',file)
				shutil.copy(file_path, preferencesPath)

		UserId = current_user.UserId
		usersPreferences = UsersPreferences(choice = choices,UserId = UserId)

		db.session.add(usersPreferences)
		db.session.commit()

		return redirect(url_for('home'))
		
	return redirect(url_for('home'))

def getEventFromUserText(text):
	text = text.lower()
	allEvents = db.session.query(Events.event).all()
	print("115", allEvents)

	RecognizedEvents = []
	for event in allEvents:
		RecognizedEvents.append(event[0].lower())

	if any(word in text for word in RecognizedEvents):
		words = [word for word in RecognizedEvents if word in text]
		event = words[0]
		print("199", event)
		return event
	else:
		return "I couldn't understand you, please retry!"

@app.route("/search",methods = ['POST'])
@login_required
def search():

	UserPreferencesExists = UsersPreferences.query.filter(UsersPreferences.UserId == current_user.UserId).first()
	UsersClosetExists = UsersCloset.query.filter(UsersCloset.UserId == current_user.UserId).first()

	print("228", UserPreferencesExists)
	print("229", UsersClosetExists)

	if UsersClosetExists and UserPreferencesExists:
		# Train a custom NER to recognize where the user is planning to go
		userText = request.form.get('search-box')
		print("194", userText)

		event = getEventFromUserText(userText)

		if event == "I couldn't understand you, please retry!":
			flash(event,"error")
			return redirect(url_for('home'))

		userid = current_user.UserId
		recommender_task = triggerRecommenderModel.delay(userid,event)
		uuid = recommender_task.task_id
		print("171",type(uuid))
		result = celery.AsyncResult(uuid)
		print("172:", result.ready())
		if result.ready():
			print("174", "Recommendations ready")
			return redirect(url_for('showRecommendations'))
		else:
			print("174", "Loading recommendations")
			return redirect(url_for('loadingRecommendations', task_id = uuid))
	else:
		flash("You either do not have anything in your closet or have not given your preferences","error")
		return redirect(url_for('home'))

@app.route("/loadingRecommendations/<string:task_id>",methods = ['GET','POST'])
@login_required
def loadingRecommendations(task_id):
	result = celery.AsyncResult(task_id)
	print("184:", result.ready())
	while not result.ready():
		print("186", "Before sleep")
		sleep(2)
		print("188", "After sleep")
	print("189:", result.ready())
	return redirect(url_for('showRecommendations'))

@app.route("/showRecommendations")
@login_required
def showRecommendations():
	recommendations = UsersRecommendations.query.filter(UsersRecommendations.UserId == current_user.UserId).order_by(UsersRecommendations.id.desc()).with_entities(UsersRecommendations.img_index).first()
	print("197", recommendations)

	recs = recommendations[0]

	if len(recs) == 1:
		# Full body garment
		pass
	else:
		# 1 top garment, 1 lower garment
		top_garment_items = recs[0]
		lower_garment_items = recs[1]

		print("206",top_garment_items)
		print("207", lower_garment_items)

		top_garments = UsersCloset.query.filter(UsersCloset.UserId == current_user.UserId).with_entities(UsersCloset.ImageFile).all()
		lower_garments = UsersCloset.query.filter(UsersCloset.UserId == current_user.UserId).with_entities(UsersCloset.ImageFile).all()

		top_garments_list = []
		lower_garments_list = []

		for item in top_garment_items:
			top_garments_list.append(top_garments[item])

		print("219",top_garments_list)

		topGarments = [i[0] for i in top_garments_list]

		print("225",topGarments)

		for item in lower_garment_items:
			lower_garments_list.append(lower_garments[item])

		print("228",lower_garments_list)

		lowerGarments = [i[0] for i in lower_garments_list]

		print("234", lowerGarments)

		garments_dict['lowerGarments'] = lowerGarments
		garments_dict['topGarments'] = topGarments

		print("244",garments_dict)

		UserPreferencesExists = UsersPreferences.query.filter(UsersPreferences.UserId == current_user.UserId).first() # TODO - Get the count of UserPreferences record and match it with the number of questions 

	return render_template("home.html", clothes_data = dic, user = str(current_user.UserId), garments = garments_dict, UserPreferencesExists = UserPreferencesExists)

@app.route("/uploadClosetImages", methods = ['POST'])
@login_required
def uploadClosetImages():
	if request.method == 'POST':
		closetPath = os.path.join(app.root_path, 'static', 'upload',str(current_user.UserId),'closet')
		pathlib.Path(closetPath).mkdir(parents=True, exist_ok=True) 
		files = request.files.getlist("file")
		for file in files:
			file.save(os.path.join(closetPath,secure_filename(file.filename)))
		return redirect(url_for('classifyClosetImages'))

@app.route("/classifyClosetImages")
@login_required
def classifyClosetImages():
	userid = current_user.UserId
	classifyClosetImagesAsync.delay(userid)
	return redirect(url_for('home'))

# Functions for Fashion Classifier Model (Code Taken from the internet)

def process_image(image_path):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	
	# Process a PIL image for use in a PyTorch model
	
	pil_image = Image.open(image_path)
	
	# Resize
	if pil_image.size[0] > pil_image.size[1]:
		pil_image.thumbnail((5000, 256))
	else:
		pil_image.thumbnail((256, 5000))
		
	# Crop 
	left_margin = (pil_image.width-224)/2
	bottom_margin = (pil_image.height-224)/2
	right_margin = left_margin + 224
	top_margin = bottom_margin + 224
	
	pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
	
	# Normalize
	np_image = np.array(pil_image)/255
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	np_image = (np_image - mean) / std
	
	# PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
	# Color channel needs to be first; retain the order of the other two dimensions.
	np_image = np_image.transpose((2, 0, 1))
	
	return np_image

def predict(image_path, model, topk=5):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''
	image = process_image(image_path)
	
	# Convert image to PyTorch tensor first
	image = torch.from_numpy(image).type(torch.FloatTensor)

	# Returns a new tensor with a dimension of size one inserted at the specified position.
	image = image.unsqueeze(0)
	
	output = model.forward(image)
	
	probabilities = torch.exp(output)
	
	# Probabilities and the indices of those probabilities corresponding to the classes
	top_probabilities, top_indices = probabilities.topk(topk)
	
	# Convert to lists
	top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
	top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
	
	idx_to_class = {value: key for key, value in model.class_to_idx.items()}
	top_classes = [idx_to_class[index] for index in top_indices]
	return top_probabilities, top_classes

def load_checkpoint(filepath):
	checkpoint = torch.load(filepath,map_location=torch.device('cpu'))
	if checkpoint['arch'] == 'vgg16':
		model = models.vgg16(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
	else:
		print("Architecture not recognized.")
	model.class_to_idx = checkpoint['class_to_idx']
	classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
											('relu', nn.ReLU()),
											('drop', nn.Dropout(p=0.5)),
											('fc2', nn.Linear(5000, 15)),
											('output', nn.LogSoftmax(dim=1))]))
	model.classifier = classifier
	model.load_state_dict(checkpoint['model_state_dict'])
	return model

# End of Fashion Classifier functions

# Recommender Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiOutputModel(nn.Module):
    def __init__(self, n_product_group_classes, n_graphic_classes, n_product_type_classes):
        # super().__init__()
        # self.base_model = torchvision.models.mobilenet_v2().features  # take the model without classifier
        # last_channel = torchvision.models.mobilenet_v2().last_channel  # size of the layer before classifier

        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.productGroup = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=n_product_group_classes)
        )
        self.graphic = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=n_graphic_classes)
        )
        self.productType = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=n_product_type_classes)
        )

    def forward(self, x):
        # x = self.base_model(x)
        x = self.model_wo_fc(x)
        # x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'productGroup': self.productGroup(x),
            'graphic': self.graphic(x),
            'productType': self.productType(x)
        }

# End of Recommender Model

# celery tasks
@celery.task(bind=True)
def classifyClosetImagesAsync(self,userid):
	print("Classifying Closet Images with Celery - Might take some time to complete.")
	print(os.path.join(app.root_path))

	global fashion_classifier_model
	print("229:model",fashion_classifier_model)
	if fashion_classifier_model is None:
		print("Loading fashion classifier model for the first time")
		fashion_classifier_model = load_checkpoint(os.path.join(app.root_path, 'ml', 'models',clothingClassifierCheckpoint))

	global rec_model
	if rec_model is None:
		# Recommender model
		rec_model = MultiOutputModel(n_product_group_classes=4,
                             n_graphic_classes=30,
                             n_product_type_classes=15).to(device)

		rec_model.load_state_dict(torch.load(os.path.join(app.root_path, 'ml', 'models',recommenderCheckpoint)))
		rec_model.cpu()
		rec_model.eval()

	# Use the model object to select the desired layer
	layer = rec_model._modules['model_wo_fc']._modules['8']

	transform = transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])

	def get_vector(image):
	    # Create a PyTorch tensor with the transformed image
	    t_img = transform(image)
	    # Create a vector of zeros that will hold our feature vector
	    # The 'avgpool' layer has an output size of 512
	    my_embedding = torch.zeros(512)

	    # Define a function that will copy the output of a layer
	    def copy_data(m, i, o):
	        my_embedding.copy_(o.flatten())                 # <-- flatten

	    # Attach that function to our selected layer
	    h = layer.register_forward_hook(copy_data)
	    # Run the model on our transformed image
	    with torch.no_grad():                               # <-- no_grad context
	        rec_model(t_img.unsqueeze(0))                       # <-- unsqueeze
	    # Detach our copy function from the layer
	    h.remove()
	    # Return the feature vector
	    return my_embedding

	closetImagesFiles = os.listdir(os.path.join(app.root_path, 'static', 'upload',str(userid),'closet'))

	processedFiles = UsersCloset.query.filter(UsersCloset.UserId == userid).with_entities(UsersCloset.ImageFile).all()
	processedFiles = [imageFile for imageFile, in processedFiles]

	unprocessedFiles = [item for item in closetImagesFiles if item not in processedFiles]

	for file in unprocessedFiles:
		img_path = os.path.join(app.root_path, 'static', 'upload',str(userid),'closet',file)
		probs, classes = predict(img_path, fashion_classifier_model, 1)

		predictedCategory = classes[0]
		print("predictedCategory for",file,"is",predictedCategory)

		img = Image.open(img_path)
		embeddings = get_vector(img)

		pickle_string = pickle.dumps(embeddings)

		# pickle.loads(decoded_result)

		closet_object = UsersCloset(ImageFile = file,PredictedCategory = predictedCategory, Embeddings = embeddings, UserId = userid)
		db.session.add(closet_object)
	
	db.session.commit()

@celery.task(bind=True)
def triggerRecommenderModel(self,userid, event):
	userEvent = event
	print("Doing the magic - might take some time to compute recommendations for you.")

	# Get the UsersCloset database, turn it into a dataframe and index it using Spotify Annoy
	closet = UsersCloset.query.filter(UsersCloset.UserId == userid).with_entities(UsersCloset.ImageFile,UsersCloset.Embeddings).all()

	df = pd.DataFrame(closet, columns =['ImageFile', 'Embeddings'])

	f = len(df['Embeddings'][0])
	t = AnnoyIndex(f, metric='euclidean')

	ntree = 1000 # hyper-parameter, the more the number of trees better the prediction
	for i, vector in enumerate(df['Embeddings']):
	    t.add_item(i, vector)
	_  = t.build(ntree)


	# Initialize the recommender model and get the embeddings of users preferences
	global recommender_model
	if recommender_model is None:
		print("Loading for the first time")

		recommender_model = MultiOutputModel(n_product_group_classes=4,
                             n_graphic_classes=30,
                             n_product_type_classes=15).to(device)

		recommender_model.load_state_dict(torch.load(os.path.join(app.root_path, 'ml', 'models',recommenderCheckpoint)))
		recommender_model.cpu()
		recommender_model.eval()

	# Use the model object to select the desired layer
	layer = recommender_model._modules['model_wo_fc']._modules['8']

	transform = transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])

	def get_vector(image):
	    # Create a PyTorch tensor with the transformed image
	    t_img = transform(image)
	    # Create a vector of zeros that will hold our feature vector
	    # The 'avgpool' layer has an output size of 512
	    my_embedding = torch.zeros(512)

	    # Define a function that will copy the output of a layer
	    def copy_data(m, i, o):
	        my_embedding.copy_(o.flatten())                 # <-- flatten

	    # Attach that function to our selected layer
	    h = layer.register_forward_hook(copy_data)
	    # Run the model on our transformed image
	    with torch.no_grad():                               # <-- no_grad context
	        recommender_model(t_img.unsqueeze(0))                       # <-- unsqueeze
	    # Detach our copy function from the layer
	    h.remove()
	    # Return the feature vector
	    return my_embedding

	closetImagesFiles = os.listdir(os.path.join(app.root_path, 'static', 'upload',str(userid),'closet'))

	usersPref = UsersPreferences.query.filter(UsersPreferences.UserId == userid).with_entities(UsersPreferences.choice).first()

	userPreferences = list(usersPref)
	print("616", userEvent)
	print("617", userPreferences)

	userPreferences = list(np.concatenate(userPreferences).flat)

	print("621", userPreferences)	
	
	print("389",userPreferences)

	recommended_items = []
	
	'''
	From user preferences, choose the array consisting of the event and predit what type of clothing each item is
	and then recommend clothes based on the predicted type
	'''
	
	allEvents = db.session.query(Events.event).all()
	print("115", allEvents)

	EventsList = []
	for event in allEvents:
		EventsList.append(event[0].lower())

	print("642", EventsList)
	print("643", userEvent)

	userPreferencesIndex = EventsList.index(userEvent)

	print("647", userPreferencesIndex)

	print("650", userPreferences)

	# From user preferences, choose 1 of upper body garment and choose 1 of lower body garment or choose 1 of full body garment

	for file in userPreferences[userPreferencesIndex]:
		img_path = os.path.join(app.root_path, 'static', 'upload',str(userid),'preferences',file)
		img = Image.open(img_path)

		embeddings = get_vector(img)

		similar_images = t.get_nns_by_vector(embeddings, n=5, include_distances=True)

		print("508",similar_images)

		recommended_items.append(similar_images[0])

		print("512", recommended_items)

	recommendations = UsersRecommendations(img_index = recommended_items, UserId = userid)
	db.session.add(recommendations)
	
	db.session.commit()




