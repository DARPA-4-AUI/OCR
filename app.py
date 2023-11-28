# -*- coding: utf-8 -*-
import json
import time
import web
import numpy as np
import uuid
import tensorflow as tf
from flask import request, Flask
import os
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
web.config.debug = False

filelock = 'file.lock'
if os.path.exists(filelock):
    os.remove(filelock)

render = web.template.render('templates', base='base')
from config import *
from apphelper.image import union_rbox, adjust_box_to_origin, base64_to_PIL

if yoloTextFlag == 'keras' or AngleModelFlag == 'tf' or ocrFlag == 'keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K

        # tf.disable_eager_execution()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  ## GPU最大占用量
        config.gpu_options.allow_growth = True  ##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    else:
        ##CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag == 'opencv':
    scale, maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag == 'darknet':
    scale, maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag == 'keras':
    scale, maxScale = IMGSIZE[0], 2048
    from text.keras_detect import text_detect
else:
    print("err,text engine in keras\opencv\darknet")

from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase

    ocr = redisDataBase().put_values
else:
    from crnn.keys import alphabetChinese, alphabetEnglish

    if ocrFlag == 'keras':
        from crnn.network_keras import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True

    elif ocrFlag == 'torch':
        from crnn.network_torch import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense

        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag == 'opencv':
        from crnn.network_dnn import CRNN

        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print("err,ocr engine in keras\opencv\darknet")

    nclass = len(alphabet) + 1
    if ocrFlag == 'opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")

    ocr = crnn.predict_job

from main import TextOcrModel  # 接main.py

model = TextOcrModel(ocr, text_detect, angle_detect)

app=Flask(__name__)
@app.route("/ocr",methods=['GET','POST'])
def get_frame():
        if request.method=='POST':
            global res
            t = time.time()
            uidJob = uuid.uuid1().__str__()
            imgString=request.files['file']
            base64image = base64.b64encode(imgString.read())
            img = base64_to_PIL(base64image) # base64字符串转换为PIL图像对象
            imgname = imgString.filename  # 获取到图片名字

            if img is not None:
                img = np.array(img)

            while time.time() - t <= TIMEOUT:
                if os.path.exists(filelock):
                    continue
                else:
                    with open(filelock, 'w') as f:
                        f.write(uidJob)

                    result, angle = model.model(img,
                                                scale=scale,
                                                maxScale=maxScale,
                                                detectAngle=False,  ##是否进行文字方向检测，通过web传参控制
                                                MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并
                                                MIN_V_OVERLAPS=0.6,
                                                MIN_SIZE_SIM=0.6,
                                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                                LINE_MIN_SCORE=0.1,
                                                leftAdjustAlph=0.01,  ##对检测的文本行进行向左延伸
                                                rightAdjustAlph=0.01,  ##对检测的文本行进行向右延伸
                                                )
                    result = union_rbox(result, 0.2)
                    res = [{'text': x['text'],
                                'name': str(i),
                                'box': {'cx': x['cx'],
                                        'cy': x['cy'],
                                        'w': x['w'],
                                        'h': x['h'],
                                        'angle': x['degree']
                                        }
                            } for i, x in enumerate(result)]
                    res = adjust_box_to_origin(img, angle, res, imgname)  ##修正box
                    os.remove(filelock)
                    break
            with open('./result/'+imgname.split('.')[0] + ".json", "w") as f:
                json.dump(res, f, ensure_ascii=False,indent=4)
            return {'res':'success'}
        else:
            return {'res':'get'}


if __name__ == "__main__":
    app.run('127.0.0.1',port=8080)
