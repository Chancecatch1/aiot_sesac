from flask import Flask, request, make_response, render_template, Response, session, redirect, url_for, send_file
import cv2
import numpy as np
import datetime
import os
import sys
from pathlib import Path
import tempfile
from animalcnn import predict_image, predict_batch
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# use image upload
import base64


app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'aiot'

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(minutes=5)
    session.modified = True


def send_file_data(data, mimetype='image/jpeg', filename='output.jpg'):
    response = make_response(data)
    response.headers.set('Content-Type', mimetype)
    response.headers.set('Content-Disposition', 'attachment', filename=filename)

    return response

def gen_frames(cap):
    # 프레임하나씩 예측 (스트리밍 방식으로 사용)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    filename = 'pred_video.mp4'
    out = cv2.VideoWriter('./static/'+filename, fourcc, 30, (w, h))

    cnt = 0
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        cnt+=1
        if cnt % 3 ==0:
            continue
        pred = predict_image(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (300, 100)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.putText(frame, pred, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    return filename


def batch_frames(cap):
    # 여러프레임 한번에 예측 (영상화 시킬때 사용)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    filename = 'pred_video.mp4'
    out = cv2.VideoWriter('./static/'+filename, fourcc, 30, (w, h))

    cnt = 0
    origins = []
    images = []
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            # del session['temp_filename']
            break
        cnt+=1
        if cnt % 3 ==0:
            continue
        im = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_AREA)
        origins.append(frame.copy())
        images.append(im)
        if len(images) > 10:
            preds = predict_batch(images)
            for ori, pred in zip(origins, preds):
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (150, 150)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                frame = cv2.putText(ori, pred, org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                out.write(frame)
            images = []
            origins = []
    cap.release()
    out.release()
    return filename


@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html',image = None, filename= None)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            # flash('No file part')
            return 'You forgot Snap!'
        fs = request.files.get('file')
        # print(fs.read(), file=sys.stderr)
        if 'jpg' in fs.filename:
            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            pred = predict_image(img)
            ret, buf = cv2.imencode('.jpeg', img)
            b64_img = base64.b64encode(buf).decode('utf-8')
            return render_template('index.html', image = b64_img, predict = pred, filename = None)
        elif 'mp4' in fs.filename:
            with tempfile.TemporaryDirectory() as td:
                temp_filename = Path(td) / 'uploaded_video'
                fs.save(temp_filename)
                vidcap = cv2.VideoCapture(str(temp_filename))
                filename = batch_frames(vidcap)
                return send_file('./static/{}'.format(filename), download_name = filename, as_attachment = True)
                # return render_template('index.html', image = None, filename = filename)
        else:
            return 'You forgot Snap!'

    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True, port=5000)