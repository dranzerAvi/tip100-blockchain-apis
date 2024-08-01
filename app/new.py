# import main Flask class and request object
from flask import Flask, request, jsonify
from web3 import Web3
from flask_cors import CORS, cross_origin

import ssl
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import random
import json
# import urllib.request
# import re
import warnings
# Importing for blockchain database apis
import datetime
import hashlib

## ML Model Api Imports

# Import the required libraries.
import os
# import cv2
#import pafy
import math
import random
import numpy as np
import datetime as dt
# import tensorflow as tf
from collections import deque
# from tensorflow import keras

# from moviepy.editor import *
# %matplotlib inline




ssl._create_default_https_context = ssl._create_unverified_context


#translator=Translator()
#from translate import Translator
warnings.filterwarnings("ignore")

# create the Flask app
app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
cred = credentials.Certificate('./key.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()


LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()`~-=_+[]{}|;\':",./<>? '

def create_cypher_dictionary():
    numbers = [ '%02d' % i for i in range(100) ]
    random.shuffle( numbers )
    return { a : b for a,b in zip( LETTERS, numbers ) }

def encrypt( cypher, string ) :
    return ''.join( cypher[ch] for ch in string )

def decrypt( cypher, string ) :
    inverse_cypher = { b : a for a,b in cypher.items() }
    return ''.join( inverse_cypher[a+b] for a,b in zip(*[iter(string)]*2) )

cypher = {'a': '75', 'b': '35', 'c': '21', 'd': '07', 'e': '99', 'f': '00', 'g': '03', 'h': '79', 'i': '90', 'j': '50', 'k': '85', 'l': '22', 'm': '26', 'n': '36', 'o': '17', 'p': '08', 'q': '82', 'r': '69', 's': '73', 't': '16', 'u': '28', 'v': '29', 'w': '45', 'x': '05', 'y': '27', 'z': '95', 'A': '89', 'B': '66', 'C': '42', 'D': '54', 'E': '46', 'F': '67', 'G': '12', 'H': '25', 'I': '93', 'J': '96', 'K': '56', 'L': '94', 'M': '11', 'N': '53', 'O': '39', 'P': '34', 'Q': '77', 'R': '32', 'S': '13', 'T': '37', 'U': '61', 'V': '58', 'W': '15', 'X': '63', 'Y': '97', 'Z': '68', '1': '47', '2': '64', '3': '24', '4': '44', '5': '78', '6': '84', '7': '19', '8': '62', '9': '09', '0': '10', '!': '88', '@': '23', '#': '41', '$': '74', '%': '51', '^': '60', '&': '06', '*': '14', '(': '20', ')': '72', '`': '04', '~': '91', '-': '57', '=': '52', '_': '59', '+': '98', '[': '71', ']': '81', '{': '76', '}': '38', '|': '30', ';': '31', "'": '83', ':': '49', '"': '02', ',': '65', '.': '87', '/': '92', '<': '55', '>': '40', '?': '80', ' ': '18'}
@app.route('/')
@cross_origin()
def form_example():
    return 'Form Data Example'

@app.route('/register', methods=['POST'])
@cross_origin()
def register():


# use the key and encrypt pwd
    requestData = json.loads(request.data)
    # passkey = 'TIP100'
 
    # str_encoded  = encrypt( cypher, phrase)

    
    
    users_ref = db.collection('users')
    infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
    web3=Web3(Web3.HTTPProvider(infura_url))
    account=web3.eth.account.create()
    
    keystore=account.encrypt(requestData['phrase'])
    
    users_ref.document(keystore.get('address')).set({'keystore':keystore})

    return 'Registration Successful'

@app.route('/getUserID', methods=['POST'])
@cross_origin()
def getUserID():
    # return b'\xde\xad\xbe\xef'.hex()
    
    try: 
        
       
            # decoded=decrypt( cypher, doc.id )
            # if(decoded==phrase):
            #     userDoc=doc

        infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
        web3=Web3(Web3.HTTPProvider(infura_url))
        requestData = json.loads(request.data)
        phrase = requestData['phrase']
            
        
        docs = db.collection(u'users').stream()
        

            
        for doc in docs:
            try:
                acc=web3.eth.account.decrypt(doc.get('keystore'),phrase)
                print(acc.hex())
                if acc:
                    account=doc.get('keystore')
            except:
                continue


    
        # account=web3.eth.account.decrypt(userDoc.get('keystore'),phrase)
        return account.get("address")
        
    except:
        return 'Account Not Found'

@app.route('/getAllTippers', methods=['GET'])
@cross_origin()
def getAllTippers():
    # return b'\xde\xad\xbe\xef'.hex()
    
    try: 
        
       
            # decoded=decrypt( cypher, doc.id )
            # if(decoded==phrase):
            #     userDoc=doc

        # infura_url="https://mainnet.infura.io/v3/5e840ebea3974bb18f3783d9c5a5c559"
        # web3=Web3(Web3.HTTPProvider(infura_url))
        # requestData = json.loads(request.data)
        # phrase = requestData['phrase']
            
        
        docs = db.collection(u'users').stream()
        users=[]

            
        for doc in docs:
            try:
                users.append({"uid":doc.get('keystore')['address'],"score":doc.get('score')})
                # print(acc.hex())
                # if acc:
                    # account=acc
            except:
                print('Exception')
                continue

    #     response = {'message': 'A block is MINED',
    #             'index': block['index'],
    #             'timestamp': block['timestamp'],
    #             'proof': block['proof'],
    #             'previous_hash': block['previous_hash']}
    # #  blockchain.to_json(block)
    #     return jsonify(response), 200
    
        # account=web3.eth.account.decrypt(userDoc.get('keystore'),phrase)
        return users
        
    except:
        return 'Account Not Found'


 
class Blockchain:
   
    # This function is created
    # to create the very first
    # block and set its hash to "0"
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0',crimeType='',description='',mediaURL='',urgency='',crimeTime='',score='',uid='',address='',dateOfIncident='',userScore='')
        docs = db.collection(u'tips').stream()   
        
        for doc in docs:
            try:
                self.create_block(proof=doc.get('proof'), previous_hash=doc.get('previous_hash'),crimeType=doc.get('crimeType'),description=doc.get('description'),mediaURL=doc.get('mediaURL'),urgency=doc.get('urgency'),crimeTime=doc.get('crimeTime'),score=doc.get('score'),uid=doc.get('uid'),address=doc.get('address'),dateOfIncident=doc.get('dateOfIncident'),userScore=doc.get('userScore'))
                print('Tip Added')
            except:
                print('Block creation Error')
                continue
        # if len(self.chain)==0:
        #     self.create_block(proof=1, previous_hash='0',crimeType='',description='',mediaURL='',urgency='',crimeTime='',score='',uid='',address='')
 
    # This function is created
    # to add further blocks
    # into the chain
    def create_block(self, proof, previous_hash,crimeType,description,mediaURL,urgency,crimeTime,score,uid,address,dateOfIncident,userScore):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'crimeType': crimeType,
                 'description': description,
                 'mediaURL': mediaURL,
                 'urgency': urgency,
                 'crimeTime': crimeTime,
                 'score': score,
                 'uid': uid,
                 'address': address,
                 'dateOfIncident': dateOfIncident,
                 'userScore': userScore,
                 'previous_hash': previous_hash,'isAlert':0,'view':1,'useful':0,'fake':0,'suspectScore':0,'mentalScore':0,'mediaScore':0,'likelyhood':0}
        self.chain.append(block)

        return block
       
    # This function is created
    
    # to display the previous block
    def print_previous_block(self):
        return self.chain[-1]
       
    # This is the function for proof of work
    # and used to successfully mine the block
    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
         
        while check_proof is False:
            hash_operation = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:5] == '00000':
                check_proof = True
            else:
                new_proof += 1
                 
        return new_proof
 
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def to_json(self, block):
        blockJson = json.dumps(block)
        return blockJson
    
    # def upload_block(self,block):
    #     tips_ref = db.collection('tips')
    #     tips_ref.document(f"{len(self.chain)}").set(block)
  
    def chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
         
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
               
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()).hexdigest()
             
            if hash_operation[:5] != '00000':
                return False
            previous_block = block
            block_index += 1
         
        return True
 

# Create the object
# of the class blockchain
blockchain = Blockchain()
 
# Mining a new block
@app.route('/addTip', methods=['POST'])
@cross_origin()
def addTip():
    requestData = json.loads(request.data)
    crimeType = requestData['crimeType']
    description = requestData['description']
    mediaURL = requestData['mediaURL']
    urgency = requestData['urgency']
    crimeTime = requestData['crimeTime']
    score = requestData['score']
    uid = requestData['uid']
    address = requestData['address']
    dateOfIncident = requestData['dateOfIncident']
    userScore = requestData['userScore']
    previous_block = blockchain.print_previous_block()
    previous_proof = previous_block['proof']
    # isA = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    block = blockchain.create_block(proof, previous_hash,crimeType,description,mediaURL,urgency,crimeTime,score,uid,address,dateOfIncident,userScore)
    # blockchain.upload_block(blockchain.to_json(block))
    tips_ref = db.collection('tips')
    blockJson=blockchain.to_json(block)
    tips_ref.document(f"{len(blockchain.chain)}").set(json.loads(blockJson))
    print(blockchain.to_json(block))
    response = {'message': 'A block is MINED',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash']}
    #  blockchain.to_json(block)
    return jsonify(response), 200
 
# Display blockchain in json format
@app.route('/getAllTips', methods=['GET'])
@cross_origin()
def getAllTips():
    allTips=[]
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        # allTips.append(doc.to_dict())
        try:
            allTips.append(doc.to_dict())
            print('Tip Added')
        except:
            print('Block adding Error')
            continue
    response = {'chain': allTips,
                'length': len(allTips)}
    return jsonify(response), 200

@app.route('/getUserTips', methods=['GET'])
@cross_origin()
def getUserTips():
    userTips=[]
    # requestData = json.loads(request.data)
    # userID=requestData['uid']
    uid = request.args.get("uid")
    print(blockchain.chain)
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        try:
            if(doc.get("uid")==uid):
                userTips.append(doc.to_dict())
                print('Tip Added')
        except:
            print('Block adding Error')
            continue
    # response = {'chain': allTips,
    #             'length': len(allTips)}
    # for block in blockchain.chain:
    #     if block['uid']==uid:
    #         userTips.append(block)
    response = {'chain': userTips,
                'length': len(userTips)}
    return jsonify(response), 200
 
# Check validity of blockchain
@app.route('/valid', methods=['GET'])
@cross_origin()
def valid():
    valid = blockchain.chain_valid(blockchain.chain)
     
    if valid:
        response = {'message': 'The Blockchain is valid.'}
    else:
        response = {'message': 'The Blockchain is not valid.'}
    return jsonify(response), 200

@app.route('/getTipDetails', methods=['GET'])
@cross_origin()
def getTipDetails():


    n = request.args.get("index")

    
    # requestData = json.loads(request.data)
    # userID=requestData['uid']
    
    print(blockchain.chain)
    docs = db.collection(u'tips').stream()   
        
    for doc in docs:
        try:
            if(doc.get("index")==n):
                
                print('Tip Added')
                return doc.to_dict()
        except:
            print('Block adding Error')
            continue
    
    # requestData = json.loads(request.data)
    # print(blockchain.chain)
    # print(requestData['index'])
    # response = {'chain': userTips,
    #             'length': len(userTips)}
    return [block for block in blockchain.chain if block and str(block['index'])==n], 200

# LRCN_model = keras.models.load_model('./LRCNModel.h5')
# SEQUENCE_LENGTH = 20
# IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
# CLASSES_LIST = ["Abuse", "Arrest", "Assault", "Burglary", "Explosion", "Fighting", "Normal Videos"]
# def predict_single_action(video_file_path, SEQUENCE_LENGTH):
#     '''
#     This function will perform single action recognition prediction on a video using the LRCN model.
#     Args:
#     video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
#     SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
#     '''

#     # Initialize the VideoCapture object to read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)

#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Declare a list to store video frames we will extract.
#     frames_list = []
    
#     # Initialize a variable to store the predicted action being performed in the video.
#     predicted_class_name = ''

#     # Get the number of frames in the video.
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Calculate the interval after which frames will be added to the list.
#     skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

#     # Iterating the number of times equal to the fixed length of sequence.
#     for frame_counter in range(SEQUENCE_LENGTH):

#         # Set the current frame position of the video.
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

#         # Read a frame.
#         success, frame = video_reader.read() 

#         # Check if frame is not read properly then break the loop.
#         if not success:
#             break

#         # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
#         normalized_frame = resized_frame / 255
        
#         # Appending the pre-processed frame into the frames list
#         frames_list.append(normalized_frame)

#     # Passing the  pre-processed frames to the model and get the predicted probabilities.
#     predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

#     # Get the index of class with highest probability.
#     predicted_label = np.argmax(predicted_labels_probabilities)

#     # Get the class name using the retrieved index.
#     predicted_class_name = CLASSES_LIST[predicted_label]
    
#     # Display the predicted action along with the prediction confidence.
#     print()
        
#     # Release the VideoCapture object. 
    
#     return f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}'


# ## ML Model API
# @app.route('/getScore', methods=['POST'])
# @cross_origin()
# def getScore():




#     requestData = json.loads(request.data)
#     # Download the youtube video.
#     # video_title = download_youtube_videos('https://youtu.be/fc3w827kwyA', test_videos_directory)
   
#     # Construct tihe nput youtube video path
#     input_video_file_path = requestData['url']

#     # Perform Single Prediction on the Test Video.
#     score=predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
#     return score
#     # Display the input video.
#     # VideoFileClip(input_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()






if __name__ == '__main__':
#     # run app in debug mode on port 5000
    app.run(debug=True, port=5000)