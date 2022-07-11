from outfitai import db, login_manager
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash

class Users(db.Model, UserMixin):
	__tablename__ = 'Users'
	UserId = db.Column(db.Integer,primary_key = True)
	Username = db.Column(db.String(100),nullable=False)
	Email = db.Column(db.String(100),nullable=False)
	Password = db.Column(db.String(128),nullable=False)
	ProfilePhoto = db.Column(db.String(100))
	FirstName = db.Column(db.String(100))
	LastName = db.Column(db.String(100))
	Gender = db.Column(db.String(100))

	#Establishing relationship with UsersCloset table because a User can have many clothing pieces
	UsersCloset = db.relationship('UsersCloset', backref = 'user')

	def __init__(self,username: str, email:str, password_plaintext: str):
		self.Username = username
		print("22",username)
		self.Email = email
		print("24",email)
		self.Password = self._generate_password_hash(password_plaintext)
		print("26",password_plaintext)

	@staticmethod
	def _generate_password_hash(password_plaintext: str):
		return generate_password_hash(password_plaintext)

	def check_password(self,password_plaintext):
		return check_password_hash(self.Password,password_plaintext)

	def get_id(self):
		return (self.UserId)

class UsersCloset(db.Model):
	__tablename__ = 'UsersCloset'
	id = db.Column(db.Integer, primary_key = True)
	ImageFile = db.Column(db.String(100), nullable = False)
	PredictedCategory = db.Column(db.String(100), nullable = False, default = 'Other')
	Embeddings = db.Column(db.PickleType, nullable = False)
	# Foreign key of 'UserId' referencing the 'UserId' of Users table
	UserId = db.Column(db.Integer, db.ForeignKey('Users.UserId'))

class Events(db.Model):
	__tablename__ = 'Events'
	id = db.Column(db.Integer, primary_key = True)
	event = db.Column(db.String(100),nullable=False)

class Choices(db.Model):
	__tablename__ = 'Choices'
	id = db.Column(db.Integer, primary_key = True)
	choice = db.Column(db.String(100),nullable = False)

	# Foreign key of 'id' referencing the 'id' of Events table
	eventId = db.Column(db.Integer, db.ForeignKey('Events.id'))


class UsersPreferences(db.Model):
	__tablename__ = 'UsersPreferences'
	id = db.Column(db.Integer, primary_key = True)
	choice = db.Column(JSON)
	# Foreign key of 'UserId' referencing the 'UserId' of Users table
	UserId = db.Column(db.Integer, db.ForeignKey('Users.UserId'))


class UsersRecommendations(db.Model):
	__tablename__ = 'UsersRecommendations'
	id = db.Column(db.Integer, primary_key = True)
	img_index = db.Column(JSON)
	# Foreign key of 'UserId' referencing the 'UserId' of Users table
	UserId = db.Column(db.Integer, db.ForeignKey('Users.UserId'))

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))