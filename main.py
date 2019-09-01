from flask import Flask, jsonify, render_template, request, redirect, url_for, Response
import webbrowser
import time
import cv2
import sys
import os
import sqlite3
from PIL import Image
import numpy
import datetime
import re, itertools
from flask import *  
import shutil
import string
import random

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import update


from random import randint


project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, "User.db"))




app = Flask(__name__)


####################------Constants-----------------########################################################    
haar_file = 'haarcascade_frontalface_default.xml'
UPLOAD_FOLDER = os.path.join('static', 'datasets')
model = cv2.face.LBPHFaceRecognizer_create() 
trainingData = 'recognizer/trainingData.yml'
threshold = 500
idSize = 9
imgSrc = 8
stringLength=10
face_cascade = cv2.CascadeClassifier(haar_file) 
conn = sqlite3.connect('User.db' ,check_same_thread=False)
cur = conn.cursor()


webcam = cv2.VideoCapture(0)
##############################################################################################################

####################------DATABASE-----------------########################################################    
app.config["SQLALCHEMY_DATABASE_URI"] = database_file
db = SQLAlchemy(app) 
 

class Parent(db.Model):
    __tablename__ = 'parent'
    id = Column(Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    datapath = db.Column(db.String(80), nullable=False)
    status = db.Column(Integer)
    children = relationship("Child", back_populates="parent")

class Child(db.Model):
    __tablename__ = 'child'
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey('parent.id'))
    src = db.Column(db.String(80), nullable=False)

    parent = relationship("Parent", back_populates="children")

####################------HOME-----------------########################################################    
@app.route('/')
def index():  
    stuff()
    return render_template('index.html')
    
def get_frame():
    while True: 
        retval, im = webcam.read() 
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            
@app.route('/calc')
def calc():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/_stuff', methods = ['GET'])
def stuff():
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(UPLOAD_FOLDER): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(UPLOAD_FOLDER, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1

    model.read(trainingData)

    while True: 
        retval, im = webcam.read()  
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        for (x, y, w, h) in faces:  
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w]  
            prediction = model.predict(face) 
            name = '% s ' % (names[prediction[0]]) 
            if prediction[1]<threshold: 
                result = name
                print(result)
                return jsonify(result=result)
            else: 
                result = "no"
                return jsonify(result=result)



####################------Register-----------------########################################################    
@app.route('/register', methods =['GET', 'POST'])
def register():
    if request.method == 'POST':
        idnum = request.form['id']
        username = request.form['username']
        add_employee(idnum, username)
        return redirect(url_for('image'))
    return render_template('register.html')

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def randomStringDigits(stringLength):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(random.choice(lettersAndDigits) for i in range(stringLength))

def add_employee(idnum, username):

    sub_data = username 

    if not os.path.isdir(UPLOAD_FOLDER): 
        os.mkdir(UPLOAD_FOLDER) 

    path = os.path.join(UPLOAD_FOLDER, sub_data) 
    if not os.path.isdir(path): 
        os.mkdir(path) 

    me = Parent(id=idnum, name=sub_data, datapath=path)
    db.session.add(me)
         
    count = 1
    while count < 100:  
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            Cid = random_with_N_digits(idSize)
            src = '% s/% s.png' % (path, randomStringDigits(imgSrc))
            cv2.imwrite(src, face)
            images = Child(id=Cid, src=src ,parent_id=idnum)
            db.session.add(images)
        count += 1
        

    db.session.commit()

    
def get_image():
    while True: 
        retval, im = webcam.read() 
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            
@app.route('/image')
def image():
    return Response(get_image(),mimetype='multipart/x-mixed-replace; boundary=frame')


def add_images(idnum, username):

    sub_data = username 

    if not os.path.isdir(UPLOAD_FOLDER): 
        os.mkdir(UPLOAD_FOLDER) 

    path = os.path.join(UPLOAD_FOLDER, sub_data) 
    if not os.path.isdir(path): 
        os.mkdir(path) 

    
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0)  
    
    count = 1
    while count < 100:  
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            Cid = random_with_N_digits(idSize)
            src = '% s/% s.png' % (path, randomStringDigits(imgSrc))
            cv2.imwrite(src, face)
            images = Child(id=Cid, src=src ,parent_id=idnum)
            db.session.add(images)
        count += 1
        cv2.imshow('OpenCV', im)

    db.session.commit()


####################------Train-----------------########################################################    
@app.route('/train')
def train():
    row = Parent.query.all()
    return render_template("train.html", row=row)

@app.route('/train_employee')
def train_employee():
    
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')
     
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(UPLOAD_FOLDER): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(UPLOAD_FOLDER, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    
    (images, lables) = [numpy.array(lis) for lis in [images, lables]] 
    
    model.train(images, lables) 
    model.save(trainingData)
    return redirect('train')


####################-----------CRUD--------------------------###################
@app.route("/deleteUser", methods =['GET', 'POST'])
def deleteUser():
    if request.method == 'POST':
        Pid = request.form.get("id")
        parentsrc = request.form.get("path")
        id = Parent.query.filter_by(id=Pid).first()
        db.session.delete(id)
        db.session.commit()
        shutil.rmtree(parentsrc)   
        return redirect(url_for('train'))
    return redirect(url_for('train'))

@app.route("/updateImages", methods =['GET', 'POST'])
def updateImages():
    if request.method == 'POST':
        Pid = request.form.get("id")
        uname = request.form.get("name")
        add_images(Pid, uname)
        return redirect(url_for('train'))
    return redirect(url_for('train'))


@app.route('/view', methods =['GET', 'POST'])
def view(): 
    if request.method == 'POST':
        Pid = request.form.get("id")
        row = Child.query.filter_by(parent_id=Pid).all()
        return render_template("view.html", row=row)
    return render_template("view.html", row=row)


@app.route("/delete", methods =['GET', 'POST'])
def delete():
    if request.method == 'POST':
        Cid = request.form.get("id")
        childsrc = request.form.get("src")
        id = Child.query.filter_by(id=Cid).first()
        db.session.delete(id)
        db.session.commit()
        os.remove(childsrc) 

        Pid = request.form.get("Pid")
        row = Child.query.filter_by(parent_id=Pid).all()
        return render_template("view.html", row=row)
    return render_template("view.html", row=row)

##################main#############################################################
@app.route('/main')
def main(): 
    return render_template("main.html")

@app.route('/login', methods =['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        password = request.form['psw']
        if uname =="admin" and password== "admin" :
            return redirect(url_for('index'))
        else:
            print()
    return render_template('register.html')



@app.route('/test')
def test(): 
    row = Parent.query.all()
    return render_template("test.html", row=row)



if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
