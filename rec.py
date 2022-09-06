import os
import time
import cv2
import numpy as np
# from infer import infer
from PIL import Image
import model
'''
金蝶财务软件文字和图标识别
实现思路：
    OCR得到所有文字的坐标（去除置信度很低的）
    根据文字位置mask，对剩余的非文字部分，做sobel,轮廓检测，记录位置
'''

def sobel(img,thresh = 10):
    '''
    边缘检测
    输入：
        img：       灰度图像
        thresh：    二值化阈值
    '''
    # 因为是从右到左做减法，因此有可能得到负值，如果输出为uint8类型，则会只保留灰度差为正的那部分，所以就只有右边缘会保留下来
    # grad_X = cv2.Scharr(img,cv2.CV_64F,1,0)
    # grad_Y = cv2.Scharr(img,cv2.CV_64F,0,1)

    grad_X = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    grad_Y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    grad_X = cv2.convertScaleAbs(grad_X)      
    grad_Y = cv2.convertScaleAbs(grad_Y)
    
    #求梯度图像
    grad = cv2.addWeighted(grad_X,0.5,grad_Y,0.5,0)
    edges = cv2.threshold(grad,thresh,255,cv2.THRESH_BINARY)[1]/255
    return edges.astype(np.uint8)

def remove_single_line(mask,line_idx,direction = 0 ):
    
    idxs = []
    if direction == 0:      # 竖线
        pos = np.where(mask[:,line_idx] > 0)[0]
    else:
        pos = np.where(mask[line_idx,:] > 0)[0]
    if len(pos) == 0:
        return 
    sub = pos[1:] - pos[0:len(pos)-1]
    seg = np.where(sub > 1)[0]
    seg_len = len(seg)
    sub2 = list(seg[1:] - seg[0:len(seg)-1]) 
    starts =  [0] + list(seg + 1)
    ends = list(seg) + [len(pos)-1]
    idxs = [[pos[starts[idx]],pos[ends[idx]]] for idx in range(len(starts))]
    idxs = sum(idxs,[])

    len_th = 50
    for i in range(0,len(idxs),2):
        seg_len = idxs[i+1]-idxs[i]
        if  seg_len > len_th:
            if direction == 0:
                mask[idxs[i]:idxs[i+1],line_idx] = 0
            else:
                mask[line_idx,idxs[i]:idxs[i+1]] = 0
    return mask

def remove_line(mask):
    '''
    去除长线
    '''
    h,w = mask.shape
    # 去除竖线
    [remove_single_line(mask,w_,0) for w_ in range(w)]
    # 去除横线
    [remove_single_line(mask,h_,1) for h_ in range(h)]
    return mask

def get_item_boxs(img):
    '''
    获取元素框，包括文字和图标
    '''
    t1 = time.time()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = sobel(gray)
    h,w = edges.shape
    t2 = time.time()
    print("sobel cost: ",t2-t1)
    edges = remove_line(edges)
    t3 = time.time()
    print("remove line cost: ",t3-t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # edges = cv2.dilate(edges,kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # edges = cv2.erode(edges,kernel)
    edges = cv2.morphologyEx(edges,op=cv2.MORPH_CLOSE,kernel=kernel,iterations=1)
    ed = 5
    edges[:ed,:] = 0
    edges[h-ed:,:] = 0
    edges[:,:ed] = 0
    edges[:,w-ed:] = 0
    t4 = time.time()
    print("morphology close cost: ",t4-t3)
    contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t5 = time.time()
    print("find contours cost: ",t5-t4)
    # draw_img = img.copy()
    # draw_img2 = img.copy()
    boxes = []
    for contour in contours:
        (x, y, bb_w, bb_h) = cv2.boundingRect(contour)
        whr_th = 1.5
        if bb_h > 80 or bb_h < 3 or bb_w < 3 or bb_h/bb_w > whr_th :
            continue
        box = (x, y, bb_w, bb_h)
        boxes.append(box)
        # cv2.drawContours(draw_img,[contour],-1,(0,255,0),2)
        # cv2.rectangle(draw_img2,(x,y),(x+bb_w,y+bb_h),255,1)
    print("get rect cost: ",time.time() - t1)
    return boxes

if __name__ == "__main__":
    
    data_home = "F:/Datasets/securety/页面识别/jindie/image1"
    imgs = [img for img in os.listdir(data_home) if os.path.splitext(img)[-1] in [".png",".webp"]]
    ocr_handle = model.OcrHandle("models/pprec.onnx",48)
    for item in imgs:
        image_path = os.path.join(data_home,item)
        img = np.array(Image.open(image_path).convert("RGB"))[:,:,::-1]    # 直接转RGB
        ori_h,ori_w = img.shape[:2]
        r = 1
        img = cv2.resize(img,(int(ori_w*r),int(ori_h*r)),cv2.INTER_LANCZOS4)
        t1 = time.time() 
        boxes = get_item_boxs(img)
        # 调用OCR做识别
        # result_image_path = os.path.join(data_home+"_result",os.path.splitext(item)[0]+".png")
        # res = infer(img,result_image_path)
        # print("OCR cost:",time.time()-t1)
        draw_img2 = img.copy()
        results = []
        icos = []
        texts = []
        score_th = 0.8
        for box in boxes:
            x,y,bb_w,bb_h = box
            ROI = img[y:y+bb_h,x:x+bb_w]
            t2 = time.time()
            result = ocr_handle.PPRecWithBox(np.array(ROI))
            print("OCR cost: ",time.time()-t2)
            text = result[0]
            score = result[1]
            if score > score_th:
                texts.append([text,box,score])
                cv2.rectangle(draw_img2,(x,y),(x+bb_w,y+bb_h),(0,255,0),1)
            else:
                icos.append(box)
                cv2.rectangle(draw_img2,(x,y),(x+bb_w,y+bb_h),(0,0,255),1)
        cv2.imshow("show", draw_img2)
        cv2.waitKey(0)
        print()