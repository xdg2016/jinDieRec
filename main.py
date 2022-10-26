from pageItemRec import page_items_rec,log
import os
import time
import cv2
import numpy as np
from PIL import Image 

# os.environ["OMP_NUM_THREADS"] = '16'

'''
测试页面文字和图标检测识别
'''

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    
    # 测试图片路径
    # data_home = "F:/Datasets/securety/PageRec/test_data/test/imgs"
    # data_home = "F:/Datasets/securety/页面识别/jindie/image1"
    # data_home = "Y:/zx-AI_lab/数据集/亚马逊页面识别/aws_gt"
    # data_home = "Y:/zx-AI_lab/数据集/页面识别截图/速卖通全屏"
    # data_home = "Y:/zx-AI_lab/数据集/页面识别截图/亚马逊窗口"
    # data_home = "F:/Datasets/securety/tmp"
    data_home = "F:/Datasets/securety/PageRec/原始标注数据/测试/jindie/image1"
    data_home = "F:/Datasets/securety/PageRec/原始标注数据/测试/chrome"

    imgs = [img for img in os.listdir(data_home) if os.path.splitext(img)[-1] in [".jpg",".png",".webp"]]
    
    # 统计耗时
    times = []
    start = 0
    boxes = []
    for i,item in enumerate(imgs[start:]):
        print("#"*100)
        print(f"{i} {item}")
        image_path = os.path.join(data_home,item)
        name = os.path.splitext(os.path.basename(image_path))[0]
        img = np.array(Image.open(image_path).convert("RGB"))           # 直接转RGB
        draw_img2 = img[:,:,::-1].copy()                                # 转成BGR为了显示和保存结果
        ori_h,ori_w = img.shape[:2]
        
        t1 = time.time() 
        icos = []
        texts = []

        # 页面元素检测（文本+图标）
        results = page_items_rec(img,
                                 r = 1,
                                 use_mp = True,
                                 process_num = 10
                                )
        trec = time.time()
        print(f"API cost: {trec-t1}")
        times.append(trec - t1)
        boxes.append(len(results["texts"])+len(results["icos"]))
        
        img_save_dir = "F:/Datasets/OCR/cls/ori_imgs2"

        # 显示文字和图标
        for i,result in enumerate(results["texts"]):
            box,text,prob = result
            print(text)
            cv2.rectangle(draw_img2,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,0,255),1)
        for i,result in enumerate(results["icos"]):
            box = result 
            cv2.rectangle(draw_img2,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),1)

        cv2.namedWindow(f'result',0)
        cv2.imshow(f"result",draw_img2)
        cv2.waitKey(0)
        # cv2.imwrite(f"result2/{item}",draw_img2)
        # cv2.imwrite(f"result_chrome/{item}",draw_img2)
        
    print(f"平均耗时：{np.mean(times)}")
    print(f"框数：{boxes}")