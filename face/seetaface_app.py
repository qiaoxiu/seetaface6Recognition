from flask import Flask
import time
import numpy as np
import os
import json
from seetaface_utils import recognition_name, identification_face, face_encoding
from flask import Flask,request, Response, render_template, make_response
app = Flask(__name__)
app.config.from_object('settings.Development')
app.config['SECRET_KEY'] = os.urandom(24)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/reg_html', methods=["GET"])
def reg_html():
    return render_template('reg.html')

@app.route('/api/face/html', endpoint="face_method", methods=["POST"])
def face_method():
    try:
        f = request.files.get('image')
        new_encodings = face_encoding(f)
        if len(new_encodings)==0:
            res = {'message':'pictrue not face area', 'code':1, 'data': []}
            return Response(json.dumps(res), content_type='application/json')

        upload_path = recognition_name(f)
        image_data = open(upload_path, "rb").read()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print('eeee:',e)
        res = {'message':'failth', 'code':1, 'data': []}
    return Response(json.dumps(res), content_type='application/json')


@app.route('/api/face/reg', endpoint="face_reg", methods=["POST"])
def face_reg():
    try:
        f = request.files.get('image')
        img_path = 'imgs_test/{}'.format(f.filename)
        f.save(img_path)
        name = request.form.get('name', '').strip()
        sid = request.form.get('id', '').strip()
        if sid=='' or name=='':
            res = {'message':'field exist null', 'code':1, 'data': []}
            return Response(json.dumps(res), content_type='application/json')
        new_encodings = face_encoding(f)
        if len(new_encodings) == 0:
            res = {'message':'pictrue not face area', 'code':1, 'data': []}
            return Response(json.dumps(res), content_type='application/json')

        with open('reg_student_json4.json') as f1:
            all_json_str = json.load(f1)
        with open('reg_student_json4.json', 'w') as f2:
            json_str = {'img': img_path, 'name': name, 'sid': sid}
            flag = 0
            for _json_str in all_json_str:
                if sid==_json_str.get('sid'):
                    _json_str.update(json_str)
                    print('ssss', _json_str)
                    flag = 1
                    break
            if flag==0:
                all_json_str.append(json_str)
            all_json_str = json.dumps(all_json_str)
            f2.write(all_json_str)
        #new_encodings = face_encoding(f)
        if os.path.exists('new_data4.npz'):
            npz = np.load('new_data4.npz')
            known_face_encodings = npz['encode']
            sids = npz['sids']
            known_face_names = npz['names']
            flag, num = 0, known_face_encodings.shape[0]
            for i in range(num):
                if str(sids[i]) == sid:
                    known_face_encodings[i] = new_encodings
                    known_face_names[i] = name
                    sids[i] = sid
                    flag = 1
                    break

            if flag == 0:
                known_face_encodings = np.vstack((known_face_encodings, new_encodings))
                sids = np.hstack((sids, [sid]))
                known_face_names = np.hstack((known_face_names, [name]))
        else:
            known_face_encodings = new_encodings[:, np.newaxis].T
            sids = [sid]
            known_face_names = [name]
        np.savez('new_data4.npz', encode=known_face_encodings, sids=sids, names=known_face_names)
        res = {'message':'success', 'code':0, 'data':[]}
    except Exception as e:
        print('eeee:',e)
        res = {'message':'failth', 'code':1, 'data': []}
    return Response(json.dumps(res), content_type='application/json')

@app.route('/api/face/identif', endpoint="face_identif", methods=["POST"])
def face_identif():
    try:
        f = request.files.get('image')
        new_encodings = face_encoding(f)
        if len(new_encodings) == 0:
            res = {'message':'pictrue not face area', 'code':1, 'data': []}
            return Response(json.dumps(res), content_type='application/json')

        name = identification_face(f)
        res = {'message':'success', 'code':0, 'data':[{'name': name}]}
    except Exception as e:
        print('eeeeeeeeeeee:',e)
        res = {'message':'failth', 'code':1, 'data':[{'name': ''}]}
    return Response(json.dumps(res), content_type='application/json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8894)
