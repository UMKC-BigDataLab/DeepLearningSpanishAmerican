'''Developed by Shivika Prasanna for Knowledge Graph for Spanish American Notarial Records (KGSAR).
Last updated on 10/05/2021.
If using conda:
    - Create conda env: % conda create -n flask_test1
    - Activate conda env: % conda activate flask_test1
    - Inside conda env:
        OpenCV: % conda install opencv
        Flask:  % conda install flask
        Flask-cors:  % conda install flask-cors
        Flask-restful:  % conda install flask-restful
        Pandas: % conda install pandas
        Pymantic: % pip install pymantic
        PIL: % conda install -c anaconda pillow
        FuzzyWuzzy: % pip install fuzzywuzzy
        Art: % pip install art
    - Run in terminal as: > java -server -Xmx4g -jar Downloads/blazegraph.jar
    - Run inside conda env as: > python3 /Downloads/Root/website/kgsar.py -t <root directory for .ttl files> -r <root directory for cleaned images>
        Subsequent runs, activate conda and then run the above command: > conda activate flask_test1
'''

from distutils.log import debug
from re import S
from this import s
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse

import pandas as pd
import os, sys, json
import operator
import datetime
import tempfile
import subprocess

import argparse

import ast
from pymantic import sparql
from PIL import Image as PImage
import base64
import cv2
import send_mail

from fuzzywuzzy import fuzz

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--ttl", required=True, help="root directory for .ttl files")
ap.add_argument("-r", "--root", required=True, help="root directory for cleaned images")
args = vars(ap.parse_args())

ttl_folder = str(args['ttl'])
root = str(args['root'])
root = "/images/"
# uncomment below to use locally
root = "D:/bigdataproject/DeepLearningSpanishAmerican/Search-Engine/Root/Images"
ttl_folder = "D:/bigdataproject/DeepLearningSpanishAmerican/Search-Engine/Root/Turtles"

# Print all versions here
print("Python: ", sys.version)
print("Pandas: ", pd.__version__)
print("OpenCV: ", cv2.__version__)

app = Flask(__name__)
api = Api(app)
CORS(app)

server = sparql.SPARQLServer('http://localhost:9999/blazegraph/sparql')

print("Copy http://localhost:5001 (check Docker Desktop for Port#) and paste it in your browser to start querying. \n Happy searching!")

class SearchWord(Resource):

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('word', required=True)
        parser.add_argument('display', required=True)

        args = parser.parse_args()
        limit = {'Default' : 15, 'Top 5' : 5, 'Top 10' : 10, 'All' : -1}
        result_limit = limit[args['display']]
        
        # Executing query
        result = server.query('select distinct ?page ?word ?wordVal ?boundingBox ?coordinateType ?coordinate ?score ?rank where {values ?coordinateType { <http://kgsar.org/botRightx> <http://kgsar.org/botRighty> <http://kgsar.org/topLeftx> <http://kgsar.org/topLefty> } . ?page <http://kgsar.org/hasWord> ?word . ?word <http://kgsar.org/wordValue> ?wordVal . ?wordVal bds:search "%s OR %s OR %s" .  ?wordVal bds:relevance ?score . ?wordVal bds:rank ?rank . ?word <http://kgsar.org/at> ?boundingBox .  ?boundingBox ?coordinateType ?coordinate . } order by  ?rank desc(?page)' % (args['word'].lower(), args['word'].lower(),args['word'].lower()))
        # print("Result: ", result)
        coordinates = {}
        group_pages = {}
        for a in result['results']['bindings']:
            page = a['page']['value']
            bb = a['boundingBox']['value'].split('/')[-1]

            if page not in group_pages:
                group_pages[page] = {"score": []}
            
            word_uri = a["word"]["value"]
            if word_uri not in group_pages[page]:
                group_pages[page][word_uri] = {"word" : "", "boundingbox" : {}}
                res_word = a['wordVal']['value']
                score = fuzz.ratio(args['word'], res_word)
                group_pages[page][word_uri]['word'] = res_word
                group_pages[page]['score'].append(score)

            group_pages[page][word_uri]['boundingbox'][a['coordinateType']['value'].split('/')[-1]] = int(float(a['coordinate']['value']))

        group_pages = sorted(group_pages.items(), key=lambda x : sum(x[1]['score']), reverse=True)

        imgPaths = []
        actual_images = []
        counter = 0
        for item in group_pages:
            actual_images_item = {'path': "", 'bounding_boxes': []}
            key, words = item
            i = key.split('document/')
            # i = key.split('page')
            tail = i[-1].split('/')
            folder = tail[-0]
            page = tail[1]
            page_num = page.split('page')[-1]
            imgPath = os.path.join(root, folder, page_num+'.jpg')
            actual_images_item['path'] = imgPath
            img_ID = folder + "/" + page_num + '.jpg'
            print("imgPath, score: ", imgPath, sum(words['score']))
            if os.path.exists(imgPath):
                im_CV = cv2.imread(imgPath)
                counter += 1
                for k, v in words.items():
                    if k != 'score':
                        boundingbox = v['boundingbox']
                        actual_images_item['bounding_boxes'].append(v)
                        cv2.rectangle(im_CV, (boundingbox['topLeftx'], boundingbox['topLefty']), (boundingbox['botRightx'], boundingbox['botRighty']), (0, 0, 255), 10)
                    
                data_uri = base64.b64encode(cv2.imencode('.jpg', im_CV)[1]).decode()
                imgPaths.append({'page': data_uri, "path": img_ID})
                actual_images.append(actual_images_item)
                if counter == result_limit:
                    break

        print("Returning data: ", len(imgPaths))
        return {'data':imgPaths, 'actualData': actual_images}, 200
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

api.add_resource(SearchWord, '/search')

data_loaded = 1
@app.route('/')
def index():
    global data_loaded
    if data_loaded == 0:
        load_kg()
        data_loaded = 1
    return render_template('index.html')

@app.route('/load')
def load_kg():
    # Loading data to Blazegraph
    server = sparql.SPARQLServer('http://localhost:9999/blazegraph/sparql')
    
    print("inside load_kg!")
    print("ttl_folder", ttl_folder)

    for file in os.listdir(ttl_folder):
        print("file", file)
        if file.endswith('.ttl'):
            ttl_file_dir = os.path.abspath(ttl_folder)
            ttl_file = os.path.join(ttl_file_dir, file)
            print("Loading: ", ttl_file)
            server.update('load <file://{}>'.format(ttl_file))
    return "Completed loading ttls!"

@app.route('/getimages', methods = ['POST'])
def get_base64_image():
    resp = request.get_json()
    l_paths = resp['paths'].split(',')
    result = []
    for path in l_paths:
        base64_image = base64.b64encode(cv2.imencode('.jpg', cv2.imread(path))[1]).decode()
        result.append({'path': path, 'b_image': base64_image})
    
    return {'data': result}, 200

@app.route('/annotate/<uuid>', methods = ['GET'])
def load_annotate_page(uuid):
    return render_template('annotate.html', data=uuid)

def send_email(data):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as annotation_file:
        actual_filename = '{}.json'.format(annotation_file.name)
        print(actual_filename)
        json.dump(data, open(actual_filename, 'w'))
        send_mail.send("Annotations update", "FYI", files=[actual_filename])
        # add to above to run locally: server='localhost'
        os.remove(actual_filename)


@app.route('/save',  methods = ['POST'])
def addAnnotations():
    data = request.get_json()
    data['datetime'] = str(datetime.datetime.now())
    data['ip'] = request.remote_addr
    send_email(data)
    return data

if __name__ == '__main__':
    app.run(host='localhost', port=5001,debug=True)
