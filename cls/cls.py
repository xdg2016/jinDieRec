import onnxruntime as ort
import os
import numpy as np
import math
import cv2
from multiprocessing.dummy import Pool as ThreadPool
from crnn.util import get_rotate_crop_image

class TextClassifier(object):
    def __init__(self, model_path,in_names,out_names):
        self.in_names = in_names
        self.out_names = out_names
        self.cls_image_shape = 3,48,192
        self.label_list = ["text","ico"]

        if not os.path.exists(model_path): 
            raise ValueError("not find model file path {}".format(model_path))
        self.sess = ort.InferenceSession(model_path)
    
    def resize_norm_img(self, img):
        '''
        预处理
        '''
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def postprocess(self, preds):
        '''
        后处理
        '''
        label_list = self.label_list
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
     
        # label = [(label_list[idx], 1.0) for idx in label]
        return decode_out 

    def predict(self,data):
        '''
        单张预测
        '''
        box,im = data
        # 预处理
        im_data = self.resize_norm_img(im)
        # 增加维度
        im_data = im_data[np.newaxis, :]
        preds = self.sess.run(self.out_names, {self.in_names: im_data.astype(np.float32)}) # 12ms
        prob_out = preds[0]
        cls_result = self.postprocess(prob_out)
        label,score = cls_result[0]
        return box,im,label, score
    
    def __call__(self,img,boxes_list,process_num = 10):
        img_list = []
        for box in boxes_list:
            x,y,bb_w,bb_h = box 
            box_ = np.array([[x,y],[x+bb_w,y],[x,y+bb_h],[x+bb_w,y+bb_h]])
            # 裁剪
            partImg_array = get_rotate_crop_image(img, box_.astype(np.float32))
            img_list.append((box,partImg_array))

        img_num = len(img_list)
        cls_res = [['', 0.0]] * img_num
        
        # 多线程处理
        pool = ThreadPool(processes=process_num)
        results = pool.map(self.predict, img_list)
        pool.close()
        pool.join()
        # for rno in range(len(results)):
        #     label, score = results[rno]
        #     cls_res[rno] = [label, score]
        return results

