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

    grad_X = cv2.Sobel(img,-1,1,0,ksize=3) # cv2.CV_64F
    grad_Y = cv2.Sobel(img,-1,0,1,ksize=3)

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

def remove_line2(img,edges):
    '''
    删除横竖线
    '''
    h,w = edges.shape
    draw_img = img.copy()
    # minLineLength = 20
    # maxLineGap = 5
    # lines = cv2.HoughLinesP(edges, 0.5, np.pi / 2, 10 ,minLineLength,maxLineGap)
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
    
    lines  =  cv2.HoughLines(edges,1,np.pi/2,100)
    L = 1500
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
        print( (x1, y1), (x2, y2))
        cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if theta == 0: # 水平
            edges = remove_single_line(edges,x1,0)
        else:
            edges = remove_single_line(edges,y1,1)


    return edges

def dilate(edge):
    idxs = np.where(edge > 0)
    h,w = edge.shape
    ys = idxs[0]
    xs = idxs[1] 
    ys_up = ys -1
    ys_down = ys + 1 
    ys_up[ys_up <0] = 0
    ys_down[ys_down>=h] = h-1
    edge[(ys_up,xs)] = 1
    edge[(ys_down,xs)] = 1
    return edge

def get_item_boxs(img):
    '''
    获取元素框，包括文字和图标
    '''
    t1 = time.time()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2.blur(gray, (3,3))#模糊降噪
    # gray = cv2.medianBlur(gray, 3)#模糊降噪
    edges = sobel(gray)
    
    h,w = edges.shape
    t2 = time.time()
    print("sobel cost: ",t2-t1)
    for i in range(1):
        # edges = remove_line2(img,edges)
        edges = remove_line2(img,edges)
    t3 = time.time()
    print("remove line cost: ",t3-t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # edges = cv2.dilate(edges,kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # edges = cv2.erode(edges,kernel)
    # edges = cv2.morphologyEx(edges,op=cv2.MORPH_CLOSE,kernel=kernel,iterations=1)

    # 上下各扩一像素
    edges = dilate(edges)

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
    draw_img2 = img.copy()
    boxes = []
    for contour in contours:
        (x, y, bb_w, bb_h) = cv2.boundingRect(contour)
        whr_th = 1.5
        if bb_h > 80 or bb_h < 3 or bb_w < 3 or bb_h/bb_w > whr_th :
            continue
        box = (x, y, bb_w, bb_h)
        boxes.append(box)
        # cv2.drawContours(draw_img,[contour],-1,(0,255,0),2)
        cv2.rectangle(draw_img2,(x,y),(x+bb_w,y+bb_h),(0,0,255),1)
    print("get rect cost: ",time.time() - t1)
    cv2.imshow("result",draw_img2)
    cv2.waitKey(0)
    return boxes

if __name__ == "__main__":
    
    data_home = "F:/Datasets/securety/页面识别/jindie/image1"
    imgs = [img for img in os.listdir(data_home) if os.path.splitext(img)[-1] in [".png",".webp"]]
    # ocr_handle = model.OcrHandle("models/pprec.onnx",48,32)
    ocr_handle = model.OcrHandle("models/crnn_lite_lstm.onnx",32,8)
    for item in imgs:
        image_path = os.path.join(data_home,item)
        img = np.array(Image.open(image_path).convert("RGB"))[:,:,::-1]    # 直接转RGB
        ori_h,ori_w = img.shape[:2]
        r = 1
        img = cv2.resize(img,(int(ori_w*r),int(ori_h*r)),cv2.INTER_LANCZOS4)
        t1 = time.time() 
        
        # 获取所有的元素
        boxes = get_item_boxs(img)
        draw_img2 = img.copy()
        results = []
        icos = []
        texts = []
        # 置信度阈值
        score_th = 0.8
        t2 = time.time()
        # results = ocr_handle.PPRecWithBox(np.array(img),boxes)
        results = ocr_handle.crnnRecWithBox(np.array(img),boxes)
        print("OCR cost: ",time.time()-t2)
        for result in results:
            box,text,prob = result
            if prob > score_th:
                texts.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
            else:
                icos.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)
        cv2.imshow("result",draw_img2)
        cv2.waitKey(0)
        print()