from PIL import Image, ImageDraw
import numpy as np
#import cv2
from seetaface.api import *
from matplotlib import pyplot as plt
import torch
import os
import time
import json
import numpy as np
TARGET_DIR = 'imgs/'
files = [f for f in os.listdir(TARGET_DIR)]
known_face_encodings = []
known_face_names = []
known_face_sids = []
json_lists = []
i = 0
init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5

seetaFace = SeetaFace(init_mask)

def get_embeddings(image):
    detect_result1 = seetaFace.Detect(image)
    face_1 = detect_result1.data[0].pos
    points = seetaFace.mark5(image,face_1)
    feature = seetaFace.Extract(image,points)
    feature = seetaFace.get_feature_numpy(feature)
    return feature

def load_image_file(file, mode='RGB'):
    im = cv2.imread(file)
    return im

i = 0
for file in files:
    img_path = os.path.join(TARGET_DIR, file)
    obj_img = load_image_file(img_path)
    i += 1
    import time
    a1 = time.time()
    obj_face_encoding = get_embeddings(obj_img)
    print('time used:', time.time()-a1)
    if not len(obj_face_encoding):
        print('eee:', file)
        continue
    known_face_encodings.append(obj_face_encoding)
    name = file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    known_face_names.append(name)
    known_face_sids.append(i)
    json_lists.append({'img': img_path, 'name': name, 'sid': str(i)})
np.savez('all4.npz', encode=known_face_encodings, sids=known_face_sids, names=known_face_names)
with open('reg_student_json4.json', 'w') as f:
    json_list_str = json.dumps(json_lists)
    f.write(json_list_str)
npz = np.load('all4.npz')
names = npz['names']
print(names)
with open('reg_student_json4.json') as ff:
    abc = json.load(ff)
    print(abc)
