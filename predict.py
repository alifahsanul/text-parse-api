import cv2
import numpy as np
from keras.models import load_model
import requests
import io
from PIL import Image
import base64
import boto3
import json
import os
from imageio import imread


SECRET_JSON = json.load(open('secret.json'))
S3_SESSION = boto3.Session(
    aws_access_key_id=SECRET_JSON['accessKeyId'],
    aws_secret_access_key=SECRET_JSON['secretAccessKey'],
)


LOADED_MODEL = load_model(os.path.join('best_model.hdf5'))
print(LOADED_MODEL.summary())


def image_prediction(my_image):
    img_inv = 255 - my_image
    _, thresh = cv2.threshold(img_inv, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(contours, key=lambda contours: cv2.boundingRect(contours)[0])
    if len(cnt) < 1:
        raise ValueError('no bounding boxes found')
    if len(cnt) > 50:
        raise ValueError('too many bounding boxes found')
    w = 28
    h = 28
    train_data = []
    print('number of boxes found', len(cnt))
    rects=[]
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)
    bool_rect=[]
    for r in rects:
        l=[]
        for rec in rects:
            flag=0
            if rec!=r:
                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                    flag=1
                l.append(flag)
            if rec==r:
                l.append(0)
        bool_rect.append(l)
    if len(bool_rect) == 0:
        raise ValueError('length is zero')
  
    dump_rect=[]
    for i in range(0,len(cnt)):
        for j in range(0,len(cnt)):
            if bool_rect[i][j]==1:
                area1=rects[i][2]*rects[i][3]
                area2=rects[j][2]*rects[j][3]
                if(area1==min(area1,area2)):
                    dump_rect.append(rects[i])



    final_rect=[i for i in rects if i not in dump_rect]
    for r in final_rect:
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        im_crop =thresh[y:y+h+10,x:x+w+10]
        im_resize = cv2.resize(im_crop,(28,28))
        im_resize=np.reshape(im_resize,(28,28,1))
        train_data.append(im_resize)

    s=''
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i])
        train_data[i] = train_data[i].reshape(1, 28, 28, 1)
        predict_x = LOADED_MODEL.predict(train_data[i]) 
        result=np.argmax(predict_x,axis=1)
        if(result[0]==10):
            s=s+'-'
        if(result[0]==11):
            s=s+'+'
        if(result[0]==12):
            s=s+'*'
        if(result[0]==0):
            s=s+'0'
        if(result[0]==1):
            s=s+'1'
        if(result[0]==2):
            s=s+'2'
        if(result[0]==3):
            s=s+'3'
        if(result[0]==4):
            s=s+'4'
        if(result[0]==5):
            s=s+'5'
        if(result[0]==6):
            s=s+'6'
        if(result[0]==7):
            s=s+'7'
        if(result[0]==8):
            s=s+'8'
        if(result[0]==9):
            s=s+'9'  
    return s

def predict_from_base64_img(base64_text_input):
    b64_string = base64_text_input.split(',')[-1]
    decoded_text = base64.b64decode(b64_string)
    cv_im_raw = imread(io.BytesIO(decoded_text))
    cv_im = np.zeros((cv_im_raw.shape[0], cv_im_raw.shape[1]))
    cv_im = 255 - cv_im_raw[:, :, 3]
    cv_im = np.clip(cv_im, 0, 255).astype('uint8')
    parsed_text = image_prediction(cv_im)
    return parsed_text, eval(parsed_text)


def s3_connect(file_name):
    s3 = S3_SESSION.resource('s3')
    bucket = s3.Bucket(SECRET_JSON['S3_BUCKET_NAME'])
    obj = bucket.Object(file_name).get()
    body = obj.get('Body')
    string_result = body.read().decode('utf-8') 
    return string_result

def run_api(filename_in_s3):
    base64_str = s3_connect(filename_in_s3)
    parsed_text, eval_parsed_text = predict_from_base64_img(base64_str)
    return parsed_text, eval_parsed_text



if __name__ == '__main__':
    result = run_api('image_base64.txt')
    print(result)


