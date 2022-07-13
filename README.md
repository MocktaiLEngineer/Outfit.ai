# Outfit.ai
I'd suggest you read the Outfit.ai PDF file for every single detail about this project.

This section assumes that the reader has downloaded the code/cloned this repository. This application requires Redis to serve as a message broker, and this demands the need for a Subsystem for Linux.

For windows users, WSL (Windows Subsystem for Linux) must be enabled and installed.
For macOS users, Lima (Linux-on-mac) must be enabled and installed.

Due to file size limitations by Github - I have hosted the 2 ML models on Google Drive (https://drive.google.com/drive/folders/1LziscsJhf4k8WyVr89xSbSCmNjAEuKnu?usp=sharing), you should download both of the models (fashion-classifier.pth and recommender-model.pth) and place it under outfitai/ml/models/ . 
  
1.	Open a terminal and navigate to the root folder of the application.
2.  Create a virtual envrionment and run 'pip install -r requirements.txt' 
3.	Change your current directory to /env/Scripts and run activate
4.	Change your current directory to the root folder of the application and run flask run
5.	Open Ubuntu and run sudo apt install redis-server
6.	In the same Ubuntu terminal, run sudo service redis-server start
7.	Open a second terminal and navigate to the root folder of the application.
8.	Change your current directory to /env/Scripts and run activate
9.	Change your current directory to the root folder of the application and run celery -A outfitai.celery worker --loglevel=INFO --concurrency 1 -P solo
10.	Go to localhost:3000 in your browser
