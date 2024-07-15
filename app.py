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
import threading

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

app = Flask(__name__)

@app.route("/ocr", methods=['GET', 'POST'])
def get_frame():
    if request.method == 'POST':
        # 原有的处理逻辑,但需要加上线程锁
        with threading.Lock():
            t = time.time()
            uidJob = uuid.uuid1().__str__()
            imgString = request.files['file']
            base64image = base64.b64encode(imgString.read())
            img = base64_to_PIL(base64image)
            imgname = imgString.filename

            if img is not None:
                img = np.array(img)

            while time.time() - t <= TIMEOUT:
                if os.path.exists(filelock):
                    continue
                else:
                    with open(filelock, 'w') as f:
                        f.write(uidJob)

                    result, angle = model.model(img, ...)
                    res = [{'text': x['text'], 'name': str(i), 'box': {...}}]
                    res, imgW, imgH = adjust_box_to_origin(img, angle, res)
                    os.remove(filelock)
                    break

            with open('./result/' + imgname.split('.')[0] + ".json", "w") as f:
                json.dump({'ui_id': imgname.split('.')[0], 'content': res}, f, ensure_ascii=False, indent=4)
            return jsonify({'ui_id': imgname.split('.')[0], 'imgW': imgW, 'imgH': imgH, 'content': res})
    else:
        return {'res': 'get'}

if __name__ == "__main__":
    app.run('127.0.0.1', port=8080, threaded=True)
