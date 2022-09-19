from rec import page_items_rec,log
import os
import time
import cv2
import numpy as np
from PIL import Image 
import config
'''
测试页面文字和图标检测识别
'''
if __name__ == "__main__":
    
    # 测试图片路径
    data_home = "F:/Datasets/securety/页面识别/chrome"
    # data_home = "F:/Datasets/securety/页面识别/jindie/image8"
    # data_home = "Y:/zx-AI_lab/数据集/亚马逊页面识别/aws_gt"
    # data_home = "F:/Datasets/securety/tmp"
    imgs = [img for img in os.listdir(data_home) if os.path.splitext(img)[-1] in [".png",".webp"]]
    
    
    # 统计耗时
    times = []
    start = 14
    boxes = []
    for i,item in enumerate(imgs[start:]):
        print("#"*200)
        print(f"{i} {item}")
        image_path = os.path.join(data_home,item)
        img = np.array(Image.open(image_path).convert("RGB"))           # 直接转RGB
        draw_img2 = img[:,:,::-1].copy()                                # 转成BGR为了显示和保存结果
        ori_h,ori_w = img.shape[:2]
        
        t1 = time.time() 
        icos = []
        texts = []

        # 页面元素检测（文本+图标）
        results = page_items_rec(img,
                                r = config.r,
                                mergebox = config.merge_box,
                                use_mp =config.use_mp,
                                process_num =config.process_num)
        trec = time.time()
        print(f"API cost: {trec-t1}")
        times.append(trec - t1)
        boxes.append(len(results))
        
        # 区分文字和图标
        for result in results:
            box,text,prob = result
            # print(result)
            if prob > config.score_th:
                texts.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
            else:
                icos.append(result)
                cv2.rectangle(draw_img2,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)

        cv2.namedWindow(f'result',0)
        cv2.imshow(f"result",draw_img2)
        cv2.waitKey(0)
        # cv2.imwrite(f"result2/{item}",draw_img2)
        # cv2.imwrite(f"result_chrome/{item}",draw_img2)
        
    print(f"平均耗时：{np.mean(times)}")
    print(f"框数：{boxes}")