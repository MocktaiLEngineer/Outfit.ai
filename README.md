This section assumes that the reader has downloaded the code from /ka370/code/trunk/app. This application requires Redis to serve as a message broker, and this demands the need for a Subsystem for Linux.

For windows users, WSL (Windows Subsystem for Linux) must be enabled and installed.
For macOS users, Lima (Linux-on-mac) must be enabled and installed.
  
1.	Open a terminal and navigate to the root folder of the application.
2.	Change your current directory to /env/Scripts and run activate
3.	Change your current directory to the root folder of the application and run flask run
4.	Open Ubuntu and run sudo apt install redis-server
5.	In the same Ubuntu terminal, run sudo service redis-server start
6.	Open a second terminal and navigate to the root folder of the application.
7.	Change your current directory to /env/Scripts and run activate
8.	Change your current directory to the root folder of the application and run celery -A outfitai.celery worker --loglevel=INFO --concurrency 1 -P solo
9.	Go to localhost:3000 in your browser
