from concurrent.futures import process

import time
import cv2
import numpy as np
import logging
import crnn.model as model
import config

log = logging.getLogger()
log.setLevel("DEBUG")

'''
金蝶财务软件文字和图标识别
实现思路：
    OCR得到所有文字的坐标（去除置信度很低的）
    根据文字位置mask，对剩余的非文字部分，做sobel,轮廓检测，记录位置
'''

# 初始化模型
ocr_handle = model.OcrHandle(config.model_path,
                                 config.infer_h,
                                 config.batch,
                                 config.keys_txt_path,
                                 config.in_names,
                                 config.out_names)
ocr_predict = ocr_handle.PPRecWithBox



def edge_detect(img,thresh = 10):
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
        # logging.debug( (x1, y1), (x2, y2))
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
    # 连通域查找
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=8)
    h,w = mask_.shape
    num = 0
    for i in range(1,num_labels):
        # 连通域外接矩形左上角坐标和宽高
        label_x = stats[i][0]
        label_y = stats[i][1]
        label_width = stats[i][2]
        label_height = stats[i][3]
        wh_r = label_width/label_height
        area =  label_width*label_height
        cond1 = area > 30*30 or label_height > h/2 or label_width > w/2 or label_height <=3 and label_width >= 10 or label_width <=3 and label_height >=10
        if not cond1:
            continue
        
        # 连通域外接区域
        label_mask = (labels[label_y:label_y + label_height, label_x:label_x + label_width]  == i).astype(np.uint8)
        # 面积比阈值
        area_rth = 0.3
        # 面积比
        area_r = label_mask.sum() / area
        # 删除框线
        cond2 = area_r < area_rth and (((wh_r > 1.5 or wh_r < 1/1.5) and label_height > 10) or (label_height > 100 and label_width > 100) )
        cond3 = wh_r > 50 and label_height < 10 or wh_r < 0.2 and label_width < 10
        cond4 = area_r > 0.9 and (wh_r > 10 or wh_r < 0.2)
        if cond2 or cond3 or cond4:
            # 将该轮廓置零 
            mask_[label_y:label_y + label_height, label_x:label_x + label_width][labels[label_y:label_y + label_height, label_x:label_x + label_width]==i] = 0        
            num += 1
    # print(time.time()-t1)
    return mask_


def merge_boxes(boxes):
    '''
    合并检测框
    '''
    boxes = np.array(boxes)
    new_boxes = []
    delta = 5
    h_delta = 5
    wh_r_th = 1.5

    boxes_l = boxes[:,0]
    boxes_r = boxes[:,0] + boxes[:,2]
    boxes_t = boxes[:,1]
    boxes_h = boxes[:,3]
    boxes_whr = boxes[:,2]/boxes[:,3]

    for i in range(len(boxes)):
        box = boxes[i]
        l,r = box[0],box[0]+box[2]
        
        idxs_r = set(np.where((boxes_l - r < delta) & (boxes_l > l))[0])                                                        # 右侧相邻框
        idxs_l = set(np.where((l - boxes_r < delta ) & (boxes_r < r))[0])                                                       # 左侧相邻框
        idxs_samel = set(np.where((abs(box[1] - boxes_t) < h_delta) & (abs(box[3] - boxes_h) < h_delta))[0])                    # 同行框
        idxs_text = set(np.where( boxes_whr > wh_r_th)[0])                                                                      # 宽高比大于阈值，认为是文字行

        idxs = (idxs_l | idxs_r) & idxs_samel & idxs_text
        if len(idxs) == 0:                                  # 没有同行左或右相邻的框
            new_boxes.append(tuple(list(box))) 
        else:
            tmp_boxes = boxes[np.array(list(idxs)+[i])]
            x = tmp_boxes[:,0].min()
            y = tmp_boxes[:,1].min()
            bb_w = (tmp_boxes[:,0]+ tmp_boxes[:,2]).max() - x
            bb_h = (tmp_boxes[:,1]+ tmp_boxes[:,3]).max() - y
            new_boxes.append((x,y,bb_w,bb_h))
    new_boxes = list(set(new_boxes))
    return new_boxes


def get_item_boxs(img,r = 1,ksize = 3,close = True,mergebox = False):
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
    ori_h,ori_w = img.shape[:2]
    draw_img2 = img.copy()
    if r != 1:
        img = cv2.resize(img,(int(ori_w*r),int(ori_h*r)),cv2.INTER_LANCZOS4)
    # 灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = edge_detect(gray,10)
    t2 = time.time()
    logging.debug(f"edge_detect cost: {t2-t1}")
   
    # 连通域检测和去除
    edges = remove_connectRegion(edges)
    t3 = time.time()
    logging.debug(f"remove_connectRegion cost: {t3-t2}")
    
    # 闭运算连接相邻文字区域，减少块数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))
    if close:
        edges = cv2.morphologyEx(edges,op=cv2.MORPH_CLOSE,kernel=kernel)
    t4 = time.time()
    logging.debug(f"morph close cost: {t4-t3}")
    
    # 查找剩余轮廓
    contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t5 = time.time()
    logging.debug(f"findContours cost: {t5-t4}")

    boxes = []
    # 过滤box
    for contour in contours:
        (x, y, bb_w, bb_h) = cv2.boundingRect(contour)
        hwr_th = 2      # 高宽比阈值
        whr_th = 10     # 宽高比阈值
        area_th = 4*4   # 面积阈值
        if bb_h > 50 or bb_h < 2 or bb_w < 2 or bb_h/bb_w > hwr_th or bb_w * bb_h < area_th or bb_h < 5 and bb_w / bb_h > whr_th:
            continue
        # 映射回原始尺寸
        box = (int(x/r), int(y/r), int(bb_w/r), int(bb_h/r))
        boxes.append(box)
    
    t6 = time.time()
    logging.debug(f"filter boxes cost: {t6-t5}")

    # 拼接相邻的box
    if mergebox:
        old_boxes = boxes
        for i in range(10):
            boxes = merge_boxes(old_boxes)
            if len(boxes) == len(old_boxes):
                break
            old_boxes = boxes
        t7 = time.time()
        logging.debug(f"merge boxes cost: {t7-t6}")

    return boxes

def page_items_rec(img,r=config.r,ksize = 3,mergebox = config.merge_box, use_mp = config.use_mp, process_num = config.process_num):
    '''
    页面元素识别
    输入：
        img:            待识别页面图像
        r:              图形缩放比例，分辨率较高时可以缩放成较小的分辨率，减少耗时，常用缩放比例：3/4
        k_size:         闭运算的核大小
        mergebox:       是否合并检测框
        use_mp:         使用多线程
        process_num:    线程数
    返回：
        文本集合: texts
        图标集合: icos
    '''
    t1 = time.time()
    # 获取所有的元素位置（文本+图标）
    boxes = get_item_boxs(img, r, ksize = ksize, mergebox = mergebox)
    t2 = time.time()
    logging.debug(f"get {len(boxes)} boxes cost: {t2 - t1}")

    results = []
    # 调用OCR识别
    results = ocr_predict(np.array(img),boxes,use_mp, process_num)
    logging.debug(f"OCR cost: {time.time()-t2}")

    # 对结果进行分类,区分文字和图标
    texts = []
    icos = []

    for i,result in enumerate(results):
        box,text,prob = result
        if prob > config.score_th :
            texts.append(result)
        else :
            icos.append(result)

    return texts,icos
