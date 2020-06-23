from PIL import Image, ImageDraw
import numpy as np
import os
import time
from seetaface.api import *
import torch

TARGET_DIR = 'imgs/'

files = [f for f in os.listdir(TARGET_DIR)]
npz = np.load('all4.npz')
known_face_encodings = npz['encode']
known_face_names = npz['names']
sids = npz['sids']

init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5

seetaFace = SeetaFace(init_mask)

def _load_image_file(file, mode='RGB'):
    #im = cv2.imread(file)
    im = Image.open(file)
    im = im.convert('RGB')
    return np.array(im)

def get_embeddings(image):
    detect_result1 = seetaFace.Detect(image)
    if detect_result1.size==0:
        return []
    face_1 = detect_result1.data[0].pos
    points = seetaFace.mark5(image, face_1)
    feature = seetaFace.Extract(image, points)
    feature = seetaFace.get_feature_numpy(feature)
    return feature

def _get_all_encoding():
    global known_face_encodings, known_face_names, sids
    if os.path.exists('new_data4.npz'):
        new_npz = np.load('new_data4.npz')
        new_encodings = new_npz['encode']
        new_names = new_npz['names']
        new_sids = new_npz['sids']
        num = known_face_encodings.shape[0]
        new_num = new_encodings.shape[0]
        flag = 0
        for j in range(new_num):
            for i in range(num):
                if str(sids[i])==new_sids[j]:
                    #known_face_encodings[i] = new_encodings[:,np.newaxis].T
                    known_face_encodings[i] = new_encodings[j]
                    known_face_names[i] = new_names[0]
                    sids[i] = new_sids[0]
                    flag = 1
                    break
        if flag==0:
            #known_face_encodings = np.vstack((known_face_encodings, new_encodings[:,np.newaxis].T))
            known_face_encodings = np.vstack((known_face_encodings, new_encodings))
            sids = np.hstack((sids, new_sids))
            known_face_names = np.hstack((known_face_names, new_names))
        os.remove('new_data4.npz')
    print(known_face_names)
    np.savez('all4.npz', encode=known_face_encodings, sids=sids, names=known_face_names)
    return known_face_encodings, sids, known_face_names


def _face_distance(known_face_encoding_list, face_encoding_to_check):
    if len(known_face_encoding_list) == 0:
        return np.empty((0))
    return np.linalg.norm(known_face_encoding_list - face_encoding_to_check, axis=1)

def _face_distance(known_face_encoding_list, face_encoding_to_check):
    if len(known_face_encoding_list) == 0:
        return np.empty((0))
    dot = np.sum(np.multiply(known_face_encoding_list, face_encoding_to_check), axis=1)
    norm = np.linalg.norm(known_face_encoding_list, axis=1) * np.linalg.norm(face_encoding_to_check)
    dist = dot / norm
    return dist


def _compare_faces(known_face_encoding_list, face_encoding_to_check, tolerance=0.6):
    return list(_face_distance(known_face_encoding_list, face_encoding_to_check) >= tolerance)


def recognition_name(img="lwx.jpg"):
    start = time.time()
    known_face_encodings, _, known_face_names =  _get_all_encoding()
    unknown_image = _load_image_file(img)
    unknown_face_encodings = face_encoding(img)
    if len(unknown_face_encodings)==0:
        return None
    pil_image = Image.fromarray(unknown_image)
    name = "Unknown"
    draw = ImageDraw.Draw(pil_image)
    matches = _compare_faces(known_face_encodings, unknown_face_encodings, tolerance=0.5)
    face_distances = _face_distance(known_face_encodings, unknown_face_encodings)
    print('face_distances: ', face_distances)
    best_match_index = np.argmax(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    print('name:', name)
    draw.text((100, 120), str(name), fill=(255, 255, 255, 255))

    end = time.time()
    print('耗时:', end-start)
    t_d = 'static/assets/img'
    up_path = os.path.join(t_d, 'aaa.jpg')
    pil_image.save(up_path, 'jpeg')
    return up_path


def _face_encodings(obj_img):
    embeddings = get_embeddings(obj_img)
    return embeddings


def face_encoding(img):
    obj_img = _load_image_file(img)
    obj_face_encoding = _face_encodings(obj_img)
    return obj_face_encoding

def identification_face(img="lwx.jpg"):
    known_face_encodings, _, known_face_names = _get_all_encoding()
    start = time.time()
    face_encodings = face_encoding(img)
    name = "Unknown"
    matches = _compare_faces(known_face_encodings, face_encodings, tolerance=0.5)
    face_distances = _face_distance(known_face_encodings, face_encodings)
    print('face_distances: ', face_distances)
    best_match_index = np.argmax(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    end = time.time()
    print('耗时:', end-start)
    print('name:', name)
    return name

#recognition_name(img='imgs/zhouxingchi.jpg')
