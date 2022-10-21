from operator import pos
import os
import random
import shutil
from paddle import rand
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image,ImageFont,ImageDraw
import numpy as np
import cv2
import math
import xml.dom.minidom

'''
页面元素解析数据处理脚本
'''

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def del_imgs_no_xml():
    '''
    删除没有xml的图片
    '''
    data_home = "F:/Datasets/securety/PageRec/原始标注数据/2022-09-29-全场景-labeled-1021"
    dirs = os.listdir(data_home)
    for dir in dirs:
        dir_path = os.path.join(data_home,dir)
        imgs_path = os.path.join(dir_path,"imgs")
        xmls_path = os.path.join(dir_path,"xmls")
        imgs = os.listdir(imgs_path)
        for img in tqdm(imgs):
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(imgs_path,img)
            xml_path = os.path.join(xmls_path,img_name+".xml")
            if not os.path.exists(xml_path):
                os.remove(img_path)

def gen_train_val():
    '''
    生成训练验证数据集
    '''
    data_home = "F:/Datasets/securety/PageRec"
    origin_data_home = data_home+"/原始标注数据/2022-09-29-全场景-labeled-1021"
    trainval_data_home = data_home+"/trainval_data/text_ico_2020-10-21"
    test_data_home = data_home+"/test_data"
    make_dirs(trainval_data_home)
    make_dirs(test_data_home)

    f_train = open(trainval_data_home+"/train.txt","w",encoding="utf-8")
    f_val = open(trainval_data_home+"/val.txt","w",encoding="utf-8")
    f_test = open(test_data_home+"/test.txt",'w',encoding="utf-8")

    dirs = os.listdir(origin_data_home)
    for dir in tqdm(dirs):
        dir_path = os.path.join(origin_data_home,dir)
        imgs = os.listdir(os.path.join(dir_path,"imgs"))
        for img in imgs:
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(dir_path,"imgs",img)
            xml_path = os.path.join(dir_path,"xmls",img_name+".xml")
            # 训练验证数据集
            make_dirs(trainval_data_home+"/train_val/imgs")
            make_dirs(trainval_data_home+'/train_val/xmls')
            dst_img_path = os.path.join(trainval_data_home+"/train_val/imgs/"+img)
            dst_xml_path = os.path.join(trainval_data_home+'/train_val/xmls/'+img_name+".xml")
            # 测试集
            make_dirs(test_data_home+"/test/imgs")
            make_dirs(test_data_home+"/test/xmls")
            dst_test_img_path = os.path.join(test_data_home+"/test/imgs/"+img)
            dst_test_xml_path = os.path.join(test_data_home+'/test/xmls/'+img_name+".xml")
            if random.random() > 0.2:
                shutil.copy(img_path,dst_img_path)
                shutil.copy(xml_path,dst_xml_path)
                f_train.write(f"{dst_img_path} {dst_xml_path}\r")
            elif random.random() > 0.5:
                shutil.copy(img_path,dst_img_path)
                shutil.copy(xml_path,dst_xml_path)
                f_val.write(f"{dst_img_path} {dst_xml_path}\r")
            else:
                shutil.copy(img_path,dst_test_img_path)
                shutil.copy(xml_path,dst_test_xml_path)
                f_test.write(f"{dst_test_img_path} {dst_test_xml_path}\r")

    f_train.close()
    f_val.close()
    f_test.close()

def read_xml(xml_path):
    '''
    读取xml模板
    '''
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return tree,root
    except:
        return None

def calculate_calss_nums():

    '''
    统计训练数据中每个类别的个数
    '''
    data_home = "F:/Datasets/securety/PageRec/test_data"

    trainf = open(os.path.join(data_home,"test.txt"))
    lines = trainf.readlines()

    class_nums = {}
    for line in tqdm(lines):
        img_path ,xml_path = line.split(" ")
        xml_path = os.path.join(data_home,xml_path.strip())
        tree,root = read_xml(xml_path)
        objs = tree.findall('object')
        for obj in objs:
            cls_name = obj.find('name').text
            if cls_name in class_nums.keys():
                class_nums[cls_name] += 1
            else:
                class_nums[cls_name] = 1
    print("class nums:")
    for k,v in class_nums.items():
        print(k,v)


def char_to_picture(text="ABCD",
                    font_dir="./fonts",
                    font_name="msyhbd.ttc",
                    background_color=(0, 0, 0, 255),
                    text_color=(255, 255, 255),
                    pictrue_size=1000,
                    font_height=150,
                    width_min=512):

    if type(pictrue_size) == tuple or type(pictrue_size) == list:
        pictrue_shape = [pictrue_size[0], pictrue_size[1]]  # w,h
    else:
        pictrue_shape = [pictrue_size, pictrue_size]

    text_position = (10, 10)
    # 取得字体文件的位置
    font_path = os.path.join(font_dir, font_name)
    font_exist = False

    # 检查字体是否存在
    for ext in [".TTF", ".ttf", ".TTC", ".ttc", ".OTF", ".otf"]:
        if os.path.exists(font_path[:-4] + ext):
            font_exist = True
            font_path = os.path.join(font_dir, font_name[:-4] + ext)
            break
    if not font_exist:
        # tqdm.write("font_path:{} not found!".format(font_path))
        return
    font_size = font_height

    font = ImageFont.truetype(font_path, int(font_size))
    w_idth, h_eight = font.getsize(text)
    w_idth = w_idth + 20 if w_idth > width_min else width_min
    pictrue_shape = (w_idth + 20, h_eight + 20)

    # 重新绘制图像
    # if len(background_color) == 3:  # 构造的边缘带文字部分透明
    #     background_color += (0,)
    im = Image.new("RGBA", pictrue_shape, background_color)
    dr = ImageDraw.Draw(im)

    dr.text(text_position, text, font=font, fill=text_color)
    im_array = np.array(im)
    im_array[:, :, 3] = 0
    return im_array

def draw_png_test(fontPath, text):
    try:
        font = ImageFont.truetype(fontPath, 50)
        t1 = text()
        t2 = text()
        while t1 == t2:
            t2 = text()
        image = Image.new(mode='RGBA', size=(100, 100))
        draw_table = ImageDraw.Draw(im=image)
        draw_table.text(xy=(0, 0), text=t1, fill='#000000', font=font)
        img = np.array(image)
        n1 = img.sum()
        if n1 == 0:
            return False

        image2 = Image.new(mode='RGBA', size=(100, 100))
        draw_table = ImageDraw.Draw(im=image2)
        draw_table.text(xy=(0, 0), text=t2, fill='#000000', font=font)
        img2 = np.array(image2)
        n2 = img2.sum()
        if n2 == 0 or n1 == n2:
            return False
        return True
    except:
        return False

# 随机生成GBK2312字符
def GBK2312():
    i = 0
    while (True):
        head = random.randint(0xb0, 0xf7)
        body = random.randint(0xa1, 0xfe)
        val = f'{head:x}{body:x}'
        try:
            str = bytes.fromhex(val).decode('gb2312')
            break
        except:
            i += 1
            if i > 20:
                return bytes.fromhex('b0a1').decode('gb2312')
            continue
    return str


# 随机生成字母数字
def ABC():
    return chr(random.randint(32, 126))


def gen_rand_text(fontPath):
    '''
    随机生成文本
    '''
    # 生成文字检查
    b_GBK2312 = False
    b_ABC = False
    if draw_png_test(fontPath, GBK2312):
        b_GBK2312 = True
    if draw_png_test(fontPath, ABC):
        b_ABC = True

    # 随机生成文字（长度、内容随机）
    font_num = random.sample(list(range(1, 30)), 1)[0]
    text = ''
    for i in range(font_num):
        if b_ABC and b_GBK2312:
            if random.random() > 0.5:
                text += GBK2312()
            else:
                text += ABC()
        elif b_ABC:
            text += ABC()
        elif b_GBK2312:
            text += GBK2312()
    if text.strip() == "":
        text = "ABCD"
    return text

def gen_rand_text2(texts):
    l = random.randint(1,30)
    text = random.choice(texts).strip()
    while len(text) < 1:
        text = random.choice(texts).strip()
    return text[:l]

def resizeByShort(img, size, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[0], img.shape[1]
    r = size * 1.0 / min(height, width)
    height = math.ceil(height * r)
    width = math.ceil(width * r)
    out = cv2.resize(img, (width, height), interpolation)
    return out

def bbox2(img):
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax

def num2hex(num):
    a = hex(int(int(num)/16))[-1]# 商
    b = hex(int(int(num)%16))[-1]# 余数
    return(a+b)

def get_text_img(text,font):
    image = Image.new(mode='RGBA', size=(300, 300),color=(0,0,0,0))
    draw_table = ImageDraw.Draw(im=image)
    r,g,b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
    fill = num2hex(r)+num2hex(g)+num2hex(b)
    draw_table.text(xy=(0, 0), text=text, fill=f'#{fill}', font=font)
    ymin, ymax, xmin, xmax = bbox2(np.array(image)[:,:,3])
    image = image.crop(box=(xmin,ymin, xmax, ymax))
    return image

def check_valid(boxes,box):
    '''
    box=[x,y,x+w,y+h]
    boxes = [[x,y,x+w,y+h],
             [x,y,x+w,y+h],
             ...]
    '''
    if len(boxes) == 0:
        return True
    boxes = np.array(boxes)
    box = np.array(box)
    box = np.tile(box,(len(boxes),1))

    x1 = np.max(np.concatenate([box[:,0][:,np.newaxis],boxes[:,0][:,np.newaxis]],1),1)
    x2 = np.min(np.concatenate([box[:,2][:,np.newaxis],boxes[:,2][:,np.newaxis]],1),1)
    y1 = np.max(np.concatenate([box[:,1][:,np.newaxis],boxes[:,1][:,np.newaxis]],1),1)
    y2 = np.min(np.concatenate([box[:,3][:,np.newaxis],boxes[:,3][:,np.newaxis]],1),1)

    inter_area = (x2-x1)*(y2-y1)
    box_area = (box[:,2]-box[:,0])* (box[:,3]-box[:,1])
    boxes_area = (boxes[:,2]-boxes[:,0])* (boxes[:,3]-boxes[:,1])
    iou = inter_area / (box_area + boxes_area - inter_area)
    if (iou>0).any():
        return False
    else:
        return True

def write_xml(folder: str, img_name: str, path: str, img_width: int, img_height: int, tag_num: int, tag_names: str, box_list:list,save_path:str):
    '''
    VOC标注xml文件生成函数
    :param folder: 文件夹名
    :param img_name:
    :param path:
    :param img_width:
    :param img_height:
    :param tag_num: 图片内的标注框数量
    :param tag_name: 标注名称
    :param box_list: 标注坐标,其数据格式为[[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2]....]
    :return: a standard VOC format .xml file, named "img_name.xml"
    '''
    # 创建dom树对象
    doc = xml.dom.minidom.Document()
 
    # 创建root结点annotation，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)
 
    # 创建结点并加入到根结点
    folder_node = doc.createElement("folder")
    folder_value = doc.createTextNode(folder)
    folder_node.appendChild(folder_value)
    root_node.appendChild(folder_node)
 
    filename_node = doc.createElement("filename")
    filename_value = doc.createTextNode(img_name)
    filename_node.appendChild(filename_value)
    root_node.appendChild(filename_node)
 
    path_node = doc.createElement("path")
    path_value = doc.createTextNode(path)
    path_node.appendChild(path_value)
    root_node.appendChild(path_node)
 
    source_node = doc.createElement("source")
    database_node = doc.createElement("database")
    database_node.appendChild(doc.createTextNode("Unknown"))
    source_node.appendChild(database_node)
    root_node.appendChild(source_node)
 
    size_node = doc.createElement("size")
    for item, value in zip(["width", "height", "depth"], [img_width, img_height, 3]):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        size_node.appendChild(elem)
    root_node.appendChild(size_node)
 
    seg_node = doc.createElement("segmented")
    seg_node.appendChild(doc.createTextNode(str(0)))
    root_node.appendChild(seg_node)
 
    for i in range(tag_num):
        obj_node = doc.createElement("object")
        name_node = doc.createElement("name")
        name_node.appendChild(doc.createTextNode(tag_names[i]))
        obj_node.appendChild(name_node)
 
        pose_node = doc.createElement("pose")
        pose_node.appendChild(doc.createTextNode("Unspecified"))
        obj_node.appendChild(pose_node)
 
        trun_node = doc.createElement("truncated")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)
 
        trun_node = doc.createElement("difficult")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)
 
        bndbox_node = doc.createElement("bndbox")
        for item, value in zip(["xmin", "ymin", "xmax", "ymax"], box_list[i]):
            elem = doc.createElement(item)
            elem.appendChild(doc.createTextNode(str(value)))
            bndbox_node.appendChild(elem)
        obj_node.appendChild(bndbox_node)
        root_node.appendChild(obj_node)
 
    with open(os.path.join(save_path,img_name.split('.')[-2] + ".xml"), "w", encoding="utf-8") as f:
        # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")



def gen_text_ico_imgs():
    '''
    合成图标文本图片
    '''
    # 图标库
    icos_dir = "F:/Datasets/securety/PageRec/generate/app_icos"
    icos_dir2 = "F:/Datasets/securety/PageRec/generate/Decoration"
    icos = [os.path.join(icos_dir,ico) for ico in os.listdir(icos_dir) if ico.endswith(".ico")]
    icos2 = [os.path.join(icos_dir2,ico) for ico in os.listdir(icos_dir2) if ico.endswith(".png")]
    icos = icos # + icos2
    # 背景图片
    bg_path = "Y:/zx-AI_lab/OCR/background"
    bg_imgs = []
    for home, dirs, files in os.walk(bg_path):
        for filename in files:
            if os.path.splitext(filename)[-1] in [".jpg",".JPG"]:
                # 文件名列表，包含完整路径
                bg_imgs.append(os.path.join(home, filename))

    # 字体
    font_dir = "F:/Datasets/securety/PageRec/generate/fonts"
    fonts = os.listdir(font_dir)
    # 总合成图片数
    total_imgs = 1000
    short = 1080

    # 文本
    text_dir = "F:/Datasets/securety/PageRec/generate/corpus"
    txts = [os.path.join(text_dir ,txt) for txt in os.listdir(text_dir) if txt.endswith("txt")]
    texts = []
    for txt in txts:
        f_txt = open(txt,"r",encoding="utf-8")
        lines = f_txt.readlines()
        texts.extend(lines)
        f_txt.close()

    # 创建目录
    img_save_home = "F:/Datasets/securety/PageRec/原始标注数据/生成3"
    make_dirs(img_save_home)
    imgs_path = os.path.join(img_save_home,"train_val","imgs")
    make_dirs(imgs_path)
    xmls_path = os.path.join(img_save_home,"train_val","xmls")
    make_dirs(xmls_path) 

    f_train = open(os.path.join(img_save_home,"train.txt"),"w",encoding="utf-8")

    # 合成图片
    for i in tqdm(range(total_imgs)):
        # 生成背景
        if random.random() > 1:
            bg_img_path = random.choice(bg_imgs)
            bg_img = np.array(Image.open(bg_img_path).convert("RGB"))
            bg_img = resizeByShort(bg_img,short)
            bg_img = Image.fromarray(bg_img)
        else:
            color = (255,255,255)
            if random.random() > 0.5:
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            bg_img = Image.new(mode='RGB', size=(1920, 1080),color=color)
        w,h = bg_img.size
        bg_img_np = np.array(bg_img)
        positions = []
        # 生成文字和图标
        random_text_num = random.randint(200,500)
        label_names = []
        box_list = []
        # 最大最小尺寸
        max_thresh = 1/8
        min_thresh = 15
        for j in range(random_text_num):

            if random.random() > 0.8:
                # 图标
                channel = 3
                while channel==3:
                    ico_path = random.choice(icos)
                    ico_im = np.array(Image.open(ico_path))
                    channel = ico_im.shape[2]
                item_img_np = ico_im
                if ico_im.shape[2] < 4:
                    a = np.ones_like(ico_im[:,:,0])*255
                    ico_im = np.concatenate([ico_im,a[:,:,np.newaxis]],2)
                a = item_img_np[:,:,3]
                ymin, ymax, xmin, xmax = bbox2(np.array(a))
                item_img_np = item_img_np[ymin:ymax,xmin:xmax]
                item_h,item_w,c = item_img_np.shape
                label_names.append("ico")
            else:
                # 文字
                font_path = os.path.join(font_dir, random.choice(fonts))
                font_size = random.randint(20,50)
                font = ImageFont.truetype(font_path, font_size)
                text = gen_rand_text2(texts)
                text_img = get_text_img(text,font)
                item_img_np = np.array(text_img)
                item_h,item_w,c = item_img_np.shape
                label_names.append("text")
            if item_h ==0 or item_w ==0 :
                label_names.pop()
                continue
            # 缩放
            r = 1
            if item_h / h > max_thresh:
                r = h*max_thresh / item_h
            elif item_w / w > max_thresh:
                r = w*max_thresh / item_w
            elif item_h < min_thresh :
                r = min_thresh/item_h
            elif item_w < min_thresh:
                r = min_thresh/item_w
            if r != 1:
                item_w,item_h = int(item_w*r),int(item_h*r) 
                item_img_np = cv2.resize(item_img_np,(item_w,item_h))
            
            # 随机贴图
            w_,h_ = item_w,item_h
            x = random.randint(0,w)
            y = random.randint(0,h)
            # 超出边界
            if x+item_w > w:
                w_ = w - x
            if y+item_h > h:
                h_ = h - y

            # 只漏出一半，不要
            if h_ < item_h/2 or w_ < item_w/2:
                label_names.pop()
                continue
            # 与已经贴过的做iou,无交叉才贴图
            if len(positions)==0 or check_valid(positions,[x,y,x+w_,y+h_]):
                item_img_np = item_img_np[:h_,:w_]
                alpha = (item_img_np[:,:,3]/255)[:,:,np.newaxis]
                bg_img_np[y:y+h_,x:x+w_] = (bg_img_np[y:y+h_,x:x+w_] * (1-alpha) + item_img_np[:,:,:3] * alpha).astype(np.uint8)
                positions.append([x,y,x+w_,y+h_])
                box_list.append([x,y,x+w_,y+h_])
                # cv2.namedWindow("show",0)
                # cv2.imshow("show",bg_img_np)
                # cv2.waitKey(0)
            else:
                label_names.pop()
        print("box nums: ",len(box_list))

        img_name = f"{i}.jpg"   
        img_save_path = imgs_path+f"/{img_name}"
        xml_save_path = xmls_path+f"/{i}.xml"
        
        # 保存结果
        Image.fromarray(bg_img_np).save(img_save_path)
        write_xml(folder=img_save_home,
                img_name=img_name,
                path=img_save_path,
                img_width=w,
                img_height=h,
                tag_num=len(label_names),
                tag_names=label_names,
                box_list=box_list,
                save_path = xmls_path,
                )
        f_train.write(f"{img_save_path} {xml_save_path}\r")
        # print()
    f_train.close()



if __name__ == "__main__":
    # del_imgs_no_xml()
    # gen_train_val()
    calculate_calss_nums()
    # gen_text_ico_imgs()