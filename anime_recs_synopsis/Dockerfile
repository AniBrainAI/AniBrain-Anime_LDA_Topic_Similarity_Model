# https://medium.com/@tasnuva2606/dockerize-flask-app-4998a378a6aa

# set base image (host OS)
FROM python:3.9

# set the working directory in the container
WORKDIR /code

# copy the content of the current directory to the working directory
ADD . /code

# install dependencies
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "python", "app.py" ]