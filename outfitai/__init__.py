import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from celery_utils import get_celery_app_instance

app = Flask(__name__, template_folder="templates")
celery = get_celery_app_instance(app)

app.config.from_mapping(
	SECRET_KEY=os.urandom(42),
)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:root@localhost/outfitaidb"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)


from outfitai import views	
