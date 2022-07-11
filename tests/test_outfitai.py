import pytest
import json
from flask import session
from outfitai import app
from outfitai.models import *
import os

@pytest.fixture
def client():
    return app.test_client()

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Let me in!' in response.data

def test_login(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Sign in' in response.data

def test_new_user(client):
	"""
	GIVEN a user model
	WHEN a new user is created 
	THEN check the email and password hash are defined correctly
	"""

	new_user = Users('kavin','kavinarasu22@gmail.com','1234')
	assert new_user.Username == 'kavin'
	assert new_user.Email == 'kavinarasu22@gmail.com'
	assert new_user.Password != '1234'

def test_home_page_not_logged_in(client):
	"""
	GIVEN a user who isn't logged in
	WHEN the user tries to access a page that they do not have access to
	THEN check if the user is being redirected to the sign in page instead
	"""
	response = client.get('/home', follow_redirects = True)
	assert response.status_code == 200
	assert b'Sign in' in response.data


def test_home_page_logged_in(client):
	"""
	GIVEN a user who isn logged in
	WHEN the user tries to access a page that they have access to
	THEN check if the user is being allowed to access that page
	"""
	with client:
		client.post("/signin",data=dict(email="kavinarasu22@gmail.com",password="1234"))
		response = client.get('/home', follow_redirects = True)
		assert response.status_code == 200
		assert b'My Closet' in response.data

def test_model_checkpoints_exists():
	clothingClassifierCheckpoint = 'fashion-classifier.pth'
	assert os.path.exists(os.path.join(app.root_path, 'ml', 'models',clothingClassifierCheckpoint))
	
	recommenderCheckpoint = 'recommender-model.pth'
	assert os.path.exists(os.path.join(app.root_path, 'ml', 'models',recommenderCheckpoint))

def test_with_no_closet_items(client):

	with client:
		client.post("/signin",data=dict(email="kavinarasu22@gmail.com",password="1234"))
		client.post("/search", data = dict(searchbox="What can I wear to the dinner tomorrow?"))
		response = client.get('/home')
		assert b'You either do not have anything in your closet or have not given your preferences' in response.data

def test_with_no_preferences(client):

	with client:
		client.post("/signin",data=dict(email="kavinarasu22@gmail.com",password="1234"))
		client.post("/search", data = dict(searchbox="What can I wear to the dinner tomorrow?"))
		response = client.get('/home')
		assert b'You either do not have anything in your closet or have not given your preferences' in response.data

