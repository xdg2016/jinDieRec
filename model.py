from cv2 import transform
from torch import batch_norm
from crnn.CRNN import CRNNHandle,PPrecHandle

from PIL import Image
import numpy as np
import cv2
import copy

import time
import traceback

class  OcrHandle(object):
    def __init__(self,model_path,target_h = 32):
        self.crnn_handle = CRNNHandle(model_path)
        self.pprec_handle = PPrecHandle(model_path)
        self.target_h = target_h

    def preprocess(self,im,max_wh_ratio = 3):
        target_h = self.target_h
        scale = im.shape[0] * 1.0 / target_h
        w = im.shape[1] / scale
        w = int(w)
        img = cv2.resize(im,(w, target_h),interpolation=cv2.INTER_AREA) # ,interpolation=cv2.INTER_AREA 在这里效果最好
       
        img -= 127.5
        img /= 127.5

        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        return transformed_image

    def cutbox(self,img):
        delta = 3
        h,w = img.shape[:2]
        return img[delta:h-delta,delta:w-delta]


    def crnnRecWithBox(self,im):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        partImg = self.preprocess(im[:,:,::-1].astype(np.float32))
        try:
            simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
        except Exception as e:
            print(traceback.format_exc())
        
        simPred = simPred.strip()
        return simPred

    def PPRecWithBox(self,im):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        partImg = self.preprocess(im[:,:,::-1].astype(np.float32))
        try:
            simPred,score = self.pprec_handle.predict_rbg(partImg)  ##识别的文本
        except Exception as e:
            print(traceback.format_exc())
        
        simPred = simPred.strip()
        return simPred,score

if __name__ == "__main__":
    pass
