from infer import *
import numpy as np
times = []

def get_mask(mask_path):
    mask = np.ones_like(img[:,:,0])*255
    if not os.path.exists(mask_path):
        return mask
    maskimg = cv2.imread(mask_path,0)
    # 闭运算
    kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(5,5))
    mask=cv2.morphologyEx(maskimg.copy(),op=cv2.MORPH_CLOSE,kernel=kernel,iterations=1)

    # 找外部轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)  # 计算点集外接矩形
        # cv2.rectangle(maskimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mask[y:y+h,x:x+w] = 255
    return mask

for i in range(1):
    all_costs = []
    # data_home = "E:\workspace\AI\SecurityAudit/PPOCRLabel/data/all"
    data_home = "E:\workspace\AI/SecurityAudit/test"
    imgs = [img for img in os.listdir(data_home) if "mask" not in img and os.path.splitext(img)[-1] in [".png",".webp"]]
    for item in imgs:
        image_path = os.path.join(data_home,item)
        img = cv2.imread(image_path)[:,:,::-1]    # 直接转RGB 
        # mask_path = os.path.join(data_home,"mask"+str(int(item.split('.')[0][-1:]))+".webp")
        
        # mask = get_mask(mask_path)
        # # mask = np.zeros_like(img[:,:,0])
        # # mask[431:475,117:577] = 255
        # img[mask == 0] = 0
        result_image_path = os.path.join(data_home+"_result",os.path.splitext(item)[0]+".png")
        cost = infer(img,result_image_path)
        all_costs.append(cost)
    times.append(all_costs)

print("real time:")
times = np.array(times)
print(times.mean(axis=0))
time_ = times.mean(axis=0)
print("avg time:", time_.mean(axis=0))
