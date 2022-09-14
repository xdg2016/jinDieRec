import math
import os
import time
import cv2
import numpy as np
# from infer import infer
from PIL import Image
import model
from skimage import morphology

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
        img:        灰度图像
        thresh:     二值化阈值
    返回:
        edges:      二值化后的边缘图
    '''
    # 因为是从右到左做减法，因此有可能得到负值，如果输出为uint8类型，则会只保留灰度差为正的那部分，所以就只有右边缘会保留下来
    
    # grad_X = cv2.Sobel(img,cv2.CV_64F,1,0,3) # cv2.CV_64F
    # grad_Y = cv2.Sobel(img,cv2.CV_64F,0,1,3)

    # Robert算子边缘检测
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    grad_X = cv2.filter2D(img, cv2.CV_16S, kernelx)
    grad_Y = cv2.filter2D(img, cv2.CV_16S, kernely)

    grad_X = cv2.convertScaleAbs(grad_X)      
    grad_Y = cv2.convertScaleAbs(grad_Y)
    
    # #求梯度图像
    grad = cv2.addWeighted(grad_X,0.5,grad_Y,0.5,0)
    edges = cv2.threshold(grad,thresh,255,cv2.THRESH_BINARY)[1]/255

    # edges = cv2.Canny(img,10,200)

    return edges.astype(np.uint8)

def remove_single_line(mask,line_idx,direction = 0,len_th = 50):
    '''
    去除某一行或列中的长线段
    '''
    h,w = mask.shape
    idxs = []
    if direction == 0:      # 竖线
        pos = np.where(mask[:,line_idx] > 0)[0]
    else:
        pos = np.where(mask[line_idx,:] > 0)[0]
    if len(pos) == 0:
        return mask
    sub = pos[1:] - pos[0:len(pos)-1]
    seg = np.where(sub > 1)[0]
    seg_len = len(seg)
    starts =  [0] + list(seg + 1)
    ends = list(seg) + [len(pos)-1]
    idxs = [[pos[starts[idx]],pos[ends[idx]]] for idx in range(len(starts))]
    idxs = sum(idxs,[])

    for i in range(0,len(idxs),2):
        seg_len = idxs[i+1]-idxs[i]
        if  seg_len > len_th:
            if direction == 0 :
                mask[idxs[i]:idxs[i+1],line_idx] = 0
                # 判断左右两列是否也是线段
                if idxs[i] > 0 and idxs[i+1] < h-1 and mask[idxs[i]:idxs[i+1],line_idx-1].sum() == seg_len and mask[idxs[i]-1,line_idx-1] == 0 and mask[idxs[i+1]+1,line_idx-1] == 0:
                    mask[idxs[i]:idxs[i+1],line_idx-1] = 0
                if idxs[i] > 0 and idxs[i+1] < h-1 and mask[idxs[i]:idxs[i+1],line_idx+1].sum() == seg_len and mask[idxs[i]-1,line_idx+1] == 0 and mask[idxs[i+1]+1,line_idx+1] == 0:
                    mask[idxs[i]:idxs[i+1],line_idx+1] = 0
            else:
                mask[line_idx,idxs[i]:idxs[i+1]] = 0
                # 判断上下两行是否也是线段
                if idxs[i] > 0 and idxs[i+1] < w-1 and mask[line_idx-1,idxs[i]:idxs[i+1]].sum() == seg_len and mask[line_idx-1, idxs[i]-1] == 0 and mask[line_idx-1, idxs[i+1]+1] == 0:
                    mask[line_idx-1, idxs[i]:idxs[i+1]] = 0
                if idxs[i] > 0 and idxs[i+1] < w-1 and mask[line_idx+1,idxs[i]:idxs[i+1]].sum() == seg_len and mask[line_idx+1, idxs[i]-1] == 0 and mask[line_idx+1, idxs[i+1]+1] == 0:
                    mask[line_idx+1, idxs[i]:idxs[i+1]] = 0
    return mask

def remove_line(mask):
    '''
    删除横竖线,使用numpy数组判断
    '''
    h,w = mask.shape
    # 去除竖线
    [remove_single_line(mask,w_,0) for w_ in range(w)]
    # 去除横线
    [remove_single_line(mask,h_,1) for h_ in range(h)]
    return mask

def remove_line2(img,edges):
    '''
    删除横竖线,使用opencv的霍夫直线检测
    '''
    h,w = edges.shape
    draw_img = img.copy()
    # minLineLength = 20
    # maxLineGap = 5
    # lines = cv2.HoughLinesP(edges, 0.5, np.pi / 2, 10 ,minLineLength,maxLineGap)      # 概率直线检测
    # min_line_len = 20
    # straight_lines = []
    
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     # 过滤斜线，长度太短的线
    #     line_len = 0
    #     if x1==x2 and abs(y1-y2) > min_line_len:
    #         edges[min(y1,y2):max(y1,y2),x1] = 0 
    #         line_len = abs(y1-y2)
    #     elif y1 == y2 and abs(x1-x2) > min_line_len:
    #         edges[y1,min(x1,x2):max(x1,x2)] = 0 
    #         line_len = abs(x1-x2)
    #     if x1 !=x2 and y1!=y2 or line_len < min_line_len:
    #         continue  
    #     if line_len > 0:
    #         straight_lines.append(line[0])
    #     cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    edges[:,np.sum(edges,axis=0) == h] = 0
    edges[np.sum(edges,axis=1) == w,:] = 0

    lines  =  cv2.HoughLines(edges,1,np.pi/2,50)
    L = 1500
    min_len = 50
    for line in  lines:
        rho,theta = line[0]
        # 强制转成横和竖线
        a  =  0 if theta > 0 else 1
        b  =  1 if theta > 0 else 0
        x0  =  a*rho
        y0  =  b*rho
        x1  = min(w-1, max(0,int(x0  +  L*(-b))))
        y1  = min(h-1, max(0,int(y0  +  L*(a))))
        x2  = min(w-1, max(0,int(x0  -  L*(-b))))
        y2  = min(h-1, max(0,int(y0  -  L*(a))))
        # print( (x1, y1), (x2, y2))
        cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if theta == 0: # 竖直
            edges = remove_single_line(edges,x1,0,min_len)
        else:
            edges = remove_single_line(edges,y1,1,min_len)

    return edges

def dilate(edge,direc = 0,delta=1):
    '''
    上下各膨胀delta个像素
    '''
    idxs = np.where(edge > 0)
    h,w = edge.shape
    ys = idxs[0]
    xs = idxs[1] 
    if direc == 0:                       # 上下
        ys_up = ys -delta
        ys_down = ys + delta 
        ys_up[ys_up < 0] = 0
        ys_down[ys_down >= h] = h-1
        edge[(ys_up,xs)] = 1
        edge[(ys_down,xs)] = 1
    else:                                # 左右
        xs_left = xs -delta
        xs_right = xs + delta 
        xs_left[xs_left < 0] = 0
        xs_right[xs_right >= w] = w-1
        edge[(ys,xs_left)] = 1
        edge[(ys,xs_right)] = 1
    return edge

def remove_edge_line(mask,ed=5):
    '''
    去除图像边缘线
    '''
    h,w = mask.shape
    ed = 5
    mask[:ed,:] = 0
    mask[h-ed:,:] = 0
    mask[:,:ed] = 0
    mask[:,w-ed:] = 0
    return mask

def remove_connect_line(mask):
    h,w = mask.shape
    mask_ = mask.copy()
    r = 6
    for h_ in range(h):
        cond1 = h_ > 0 and h_ < h-1 
        if mask_[h_,:].sum() > w - r:
            mask[h_,:] = 0
    for w_ in range(w):
        if mask_[:,w_].sum() > h - r:
            mask[:,w_] = 0
    return mask

def remove_connectRegion(mask_):
    '''
    删除不符合条件的连通域
    '''
    t1 = time.time()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=4)
    h,w = mask_.shape
    print(time.time()-t1)
    num = 0
    for i in range(1,num_labels):
        label_width = stats[i][2]
        label_height = stats[i][3]
        wh_r = label_width/label_height
        area =  label_width*label_height
        cond1 = area > 30*30 or label_height > h/2 or label_width > w/2 or label_height <=3 and label_width >= 10 or label_width <=3 and label_height >=10
        if not cond1:
            continue
        num += 1
        label_x = stats[i][0]
        label_y = stats[i][1]
        label_mask = (labels[label_y:label_y + label_height, label_x:label_x + label_width]  == i).astype(np.uint8)
        area_rth = 0.3
        area_r = label_mask.sum() / area
        # 删除框线
        if area_r < area_rth and wh_r > 1.5:
            label_mask = remove_connect_line(label_mask)
        if  (area_r < area_rth or area_r > 0.9 or \
            ((label_height <=5 or label_width <=5) and (area_r < 0.7 or wh_r > 50))) and \
            cond1:
            # 空心情况下
            mask_[labels==i] = 0            # 将该轮廓置零
    
    return mask_

def get_item_boxs(img,r=1,ksize = 3,close = True):
    '''
    获取元素框，包括文字和图标
    输入：
        img:    输入图像
        r:      图像缩放比例
        ksize:  闭运算核大小
        close:  是否进行闭运算
    输出:
        boxes:  检测出的所有box

    '''
    t1 = time.time()
    # 灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape

    # 边缘检测
    edges = sobel(gray,10)
    t2 = time.time()
    print("sobel cost: ",t2-t1)
   
    # 连通域检测和去除
    edges = remove_connectRegion(edges)
    t3 = time.time()
    print("remove line cost: ",t3-t2)
    
    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))
    if close:
        edges = cv2.morphologyEx(edges,op=cv2.MORPH_CLOSE,kernel=kernel)

    t4 = time.time()
    contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t5 = time.time()
    # draw_img2 = img.copy()
    boxes = []
    # 过滤box
    for contour in contours:
        (x, y, bb_w, bb_h) = cv2.boundingRect(contour)
        hwr_th = 2
        whr_th = 10
        area_th = 5*5
        if bb_h > 50 or bb_h < 3 or bb_w < 2 or bb_h/bb_w > hwr_th or bb_w * bb_h < area_th or bb_h < 5 and bb_w / bb_h > whr_th:
            continue
        box = (int(x/r), int(y/r), int(bb_w/r), int(bb_h/r))
        boxes.append(box)
        # cv2.drawContours(draw_img2,[contour],-1,(0,255,0),2)
        # cv2.rectangle(draw_img2,(x,y),(x+bb_w,y+bb_h),(0,0,255),1)

    print("get rect cost: ",time.time() - t1)
    # cv2.imshow("result",draw_img2)
    # cv2.waitKey(0)
    return boxes

def page_items_rec(img,r=1,ksize = 3):
    '''
    页面元素识别
    输入：
        img:    待识别页面图像
        r:      图形缩放比例
        k_size: 闭运算的核大小
    返回：
        识别结果: box,text,prob
    '''
    # 获取所有的元素位置（文本+图标）
    ori_h,ori_w = img.shape[:2]
    if r != 1:
        img_resized = cv2.resize(img,(int(ori_w*r),int(ori_h*r)),cv2.INTER_LANCZOS4)
        boxes = get_item_boxs(img_resized,r,ksize = ksize)
    else:
        boxes = get_item_boxs(img,r,ksize = ksize)
    results = []
    t2 = time.time()
    # 调用OCR识别
    # results = ocr_handle.crnnRecWithBox(np.array(img),boxes)
    results = ocr_handle.PPRecWithBox(np.array(img),boxes)
    print("OCR cost: ",time.time()-t2)

    return results



if __name__ == "__main__":
    
    # data_home = "F:/Datasets/securety/页面识别/chrome"
    data_home = "F:/Datasets/securety/页面识别/jindie/image1"
    imgs = [img for img in os.listdir(data_home) if os.path.splitext(img)[-1] in [".png",".webp"]]
    
    # 初始化OCR模型
    ocr_handle = model.OcrHandle("models/pprec_v2.onnx",32,1)
    # ocr_handle = model.OcrHandle("models/crnn_lite_lstm.onnx",32,1)
    
    times = []
    for item in imgs:
        print("#"*200)
        print(item)
        image_path = os.path.join(data_home,item)
        img = np.array(Image.open(image_path).convert("RGB"))    # 直接转RGB
        ori_h,ori_w = img.shape[:2]
        r = 1
        t1 = time.time() 
        draw_img2 = img[:,:,::-1].copy()
        icos = []
        texts = []
        # 置信度阈值
        score_th = 0.8

        # 页面元素检测（文本+图标）
        results = page_items_rec(img,r,ksize = 3)
        trec = time.time()
        print("total cost: ",trec-t1)
        times.append(trec -t1)

        # 区分文字和图标
        for result in results:
            box,text,prob = result
            # print(result)
            if prob > score_th:
                texts.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
            else:
                icos.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)

        cv2.namedWindow('result',0)
        cv2.imshow("result",draw_img2)
        cv2.waitKey(0)
        # cv2.imwrite(f"result2/{item}",draw_img2)
        # cv2.imwrite(f"result_chrome/{item}",draw_img2)
        
    
    for t in times:
        print(t)
    print("平均耗时：",np.mean(times))