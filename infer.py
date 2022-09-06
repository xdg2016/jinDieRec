

from genericpath import isdir
import time
from model import  OcrHandle
import base64
from PIL import Image, ImageDraw,ImageFont
from io import BytesIO
import os
import time

ocrhandle = OcrHandle()
request_time = {}
now_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
from config import *

import cv2

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def infer(img,result_image_path):
    '''
    :return:
    报错：
    400 没有请求参数

    '''
    all_times = 0
    start_time = time.time()
    short_size = 960
    global now_time
    global request_time

    img = Image.fromarray(img)
    
    make_dirs(os.path.dirname(result_image_path))

    '''
    是否开启图片压缩
    默认为960px
    值为 0 时表示不开启压缩
    非 0 时则压缩到该值的大小
    '''
    res = []
    do_det = True
    compress_size = 960

    if compress_size is not None:
        try:
            compress_size = int(compress_size)
        except ValueError as ex:
            # logger.error(exc_info=True)
            res.append("短边尺寸参数类型有误，只能是int类型")
            do_det = False
            # self.finish(json.dumps({'code': 400, 'msg': 'compress参数类型有误，只能是int类型'}, cls=NpEncoder))
            # return

        short_size = compress_size
        if short_size < 64:
            res.append("短边尺寸过小，请调整短边尺寸")
            do_det = False

        short_size = 32 * (short_size//32)


    img_w, img_h = img.size
    if max(img_w, img_h) * (short_size * 1.0 / min(img_w, img_h)) > dbnet_max_size:
        # logger.error(exc_info=True)
        res.append("图片reize后长边过长，请调整短边尺寸")
        do_det = False
        # self.finish(json.dumps({'code': 400, 'msg': '图片reize后长边过长，请调整短边尺寸'}, cls=NpEncoder))
        # return

    if do_det:
        start_time = time.time()
        res = ocrhandle.text_predict(img,short_size)
        end_time = time.time()
        print("cost time: {:.4f}".format(end_time-start_time))
        cost = end_time-start_time
        all_times = cost
        img_detected = img.copy()
        img_draw = ImageDraw.Draw(img_detected)
        colors = ['red', 'green', 'blue', "purple"]

        for i, r in enumerate(res):
            rect, txt, confidence = r
            if confidence < 0.5:
                continue
            # print(txt,confidence)
            x1,y1,x2,y2,x3,y3,x4,y4 = rect.reshape(-1)
            size = max(min(x2-x1,y3-y2) // 2 , 20 )

            myfont = ImageFont.truetype(os.path.join(os.getcwd(), "仿宋_GB2312.ttf"), size=size)
            fillcolor = colors[i % len(colors)]
            img_draw.text((x1, y1 - size ), str(txt), font=myfont, fill=fillcolor)
            for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:
                img_draw.line(xy=xy, fill=colors[i % len(colors)], width=2)
            
        # print(type(img_draw))
        # 保存结果
        img_detected.save(result_image_path)
        
        # cv2.imwrite("result.jpg")
        output_buffer = BytesIO()
        img_detected.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        img_detected_b64 = base64.b64encode(byte_data).decode('utf8')
    
    else:
        output_buffer = BytesIO()
        img.save(output_buffer, format='JPEG') 
        byte_data = output_buffer.getvalue()
        img_detected_b64 = base64.b64encode(byte_data).decode('utf8')

    return res
