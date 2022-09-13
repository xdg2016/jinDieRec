from importlib import import_module
from unittest import result
from cv2 import transform
from torch import batch_norm
from crnn.CRNN import CRNNHandle,PPrecHandle

from PIL import Image
import numpy as np
import cv2
import copy
from utils import get_rotate_crop_image, sorted_boxes
import time
import traceback

class  OcrHandle(object):
    def __init__(self,model_path,target_h = 32,batch_num = 8):
        self.crnn_handle = CRNNHandle(model_path)
        self.pprec_handle = PPrecHandle(model_path)
        self.target_h = target_h
        self.batch_num = batch_num

    def preprocess(self,im,max_wh_ratio = 3):
        target_h = self.target_h
        scale = im.shape[0] * 1.0 / target_h
        w = im.shape[1] / scale
        w = int(w)
        img = cv2.resize(im,(w, target_h),interpolation=cv2.INTER_AREA) # ,interpolation=cv2.INTER_AREA 在这里效果最好

        if self.batch_num > 1:
            # 最大
            max_resized_w = int(max_wh_ratio * target_h)
            padding_im = np.zeros((target_h,max_resized_w,3), dtype=np.float32)
            padding_im[:,0:w,:] = img
            img = padding_im

        img -= 127.5
        img /= 127.5

        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        return transformed_image

    def cutbox(self,img):
        '''
        将box裁小一圈
        '''
        delta = 3
        h,w = img.shape[:2]
        return img[delta:h-delta,delta:w-delta]
    
    def npbox2box(self,npbox):
        '''
        numpy格式转列表格式
        '''
        return npbox[0][0],npbox[0][1],npbox[3][0],npbox[3][1]


    def crnnRecWithBox(self,im,boxes_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        # boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []

        partImg = Image.fromarray(im).convert("RGB")
        line_imgs.append(partImg)

        count = 1
        if self.batch_num == 1:
            for box in boxes_list:
                tmp_box = copy.deepcopy(box)
                x,y,bb_w,bb_h = box 
                box = np.array([[x,y],[x+bb_w,y],[x,y+bb_h],[x+bb_w,y+bb_h]])
                # 裁剪
                partImg_array = get_rotate_crop_image(im, box.astype(np.float32))
                partImg = self.preprocess(partImg_array.astype(np.float32))
                try:
                    simPred,prob = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                results.append([self.npbox2box(box),simPred,prob])
                count += 1
        else:
            # # 批量测试
            width_list = []
            boxImgs_list = []
            box_lists =[]
            img_num = 0
            for box in boxes_list:
                tmp_box = copy.deepcopy(box)
                x,y,bb_w,bb_h = box 
                # box外扩
                # delta = 1
                # x -= delta
                # y -= delta 
                # bb_w += 2*delta
                # bb_h += 2*delta
                box = np.array([[x,y],[x+bb_w,y],[x,y+bb_h],[x+bb_w,y+bb_h]])
                partImg_array = get_rotate_crop_image(im, box.astype(np.float32))
                # partImg_array = self.cutbox(partImg_array)
                cv2.imwrite(f"tmp/{img_num}.png",partImg_array)
                img_num += 1
                h,w ,c = partImg_array.shape
                if h < 1 or h/w > 2 :
                    continue 
                box_lists.append(box)
                width_list.append(w / h)
                boxImgs_list.append(partImg_array)
            # 重新排序
            # Sorting can speed up the recognition process
            indices = np.argsort(np.array(width_list))
            boxes_list = np.array(box_lists)[indices]
            boxImgs_list = np.array(boxImgs_list)[indices]
            # 遍历
            box_num = len(boxes_list)
            for beg_img_no in range(0, box_num, self.batch_num):
                end_img_no = min(box_num, beg_img_no + self.batch_num)
                norm_img_batch = []
                batch_boxes = boxes_list[beg_img_no:end_img_no]
                max_wh_ratio = max([box.shape[1]/box.shape[0] for box in boxImgs_list[beg_img_no:end_img_no]])
                norm_img_batch = [self.preprocess(partImg,max_wh_ratio) for partImg in boxImgs_list[beg_img_no:end_img_no]]
                partImg = np.concatenate(norm_img_batch)

                try:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                    
                pred_num = self.batch_num
                if end_img_no % self.batch_num > 0:
                    pred_num = end_img_no % self.batch_num
                for i in range(pred_num):
                    results.append([self.npbox2box(batch_boxes[i]),simPred[i][0],simPred[i][1]])
                    count += 1   

        return results


    def PPRecWithBox(self,im,boxes_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []

        count = 1
        batch_num = self.batch_num
        if batch_num == 1:
            for box in boxes_list:
                x,y,bb_w,bb_h = box 
                box = np.array([[x,y],[x+bb_w,y],[x,y+bb_h],[x+bb_w,y+bb_h]])
                # tmp_box = copy.deepcopy(box)
                # 裁剪
                partImg_array = get_rotate_crop_image(im, box.astype(np.float32))
                partImg = self.preprocess(partImg_array.astype(np.float32))
                try:
                    result = self.pprec_handle.predict_rbg(partImg)
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                simPred = result[0][0]
                score = result[0][1]
                if simPred.strip() != '':
                    results.append([box,"{}、 ".format(count)+  simPred,score])
                    count += 1
        else:
            # # 批量测试
            width_list = []
            boxImgs_list = []
            box_lists =[]
            t1 = time.time()
            for box in boxes_list:
                x,y,bb_w,bb_h = box 
                box = np.array([[x,y],[x+bb_w,y],[x,y+bb_h],[x+bb_w,y+bb_h]])
                partImg_array = get_rotate_crop_image(im, box.astype(np.float32))
                # partImg_array = self.cutbox(partImg_array)
                h,w ,c = partImg_array.shape
                if h < 1  or  h/w > 1.5  :
                    continue 
                box_lists.append(box)
                width_list.append(w / h)    # 宽高比
                boxImgs_list.append(partImg_array)
            t2 = time.time()
            print("cost1: ",t2-t1)  # 约9ms
            # 重新排序
            # Sorting can speed up the recognition process
            indices = np.argsort(np.array(width_list))
            boxes_list = np.array(box_lists)[indices]
            boxImgs_list = np.array(boxImgs_list)[indices]
            t3 = time.time()
            print("cost2: ",t3-t2)
            # 遍历
            box_num = len(boxes_list)
            for beg_img_no in range(0, box_num, batch_num):
                end_img_no = min(box_num, beg_img_no + batch_num)   # 一批结束的索引
                norm_img_batch = []
                batch_boxes = boxes_list[beg_img_no:end_img_no]
                t4 = time.time()
                max_wh_ratio = max([box.shape[1]/box.shape[0] for box in boxImgs_list[beg_img_no:end_img_no]])
                t5 = time.time()
                print("求最大宽高比耗时：",t5-t4)
                norm_img_batch = [self.preprocess(partImg,max_wh_ratio) for partImg in boxImgs_list[beg_img_no:end_img_no]]
                t6 = time.time()
                print("预处理耗时: ",t6-t5)
                partImg = np.concatenate(norm_img_batch)

                try:
                    result = self.pprec_handle.predict_rbg(partImg)
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                t7 = time.time()
                print("预测耗时：",t7-t6)
                pred_num = batch_num
                if end_img_no % batch_num > 0:
                    pred_num = end_img_no % batch_num
                for i in range(pred_num):
                    simPred = result[i][0]
                    score = result[i][1]
                    if simPred.strip() != '':
                        results.append([batch_boxes[i],"{}、 ".format(count)+  simPred])
                        count += 1   
                print("batch time: " ,time.time()-t4)
            print("all cost: ",time.time()-t1)
        return results

if __name__ == "__main__":
    pass
