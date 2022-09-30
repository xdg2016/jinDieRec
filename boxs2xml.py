from logging import shutdown
import os
import shutil
from unittest import makeSuite
from urllib.request import DataHandler
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pageItemRec import get_text_icos
import numpy as np
from PIL import Image
import xml.dom.minidom

'''
用第一版的方法，检测出文本和图标框，然后用分类模型做分类，
将分好的信息，生成目标检测的xml格式标注文件，用于后续训练检测模型
'''

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

 
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

def rename_img_xml(data_home,dir):
    imgs_path = os.path.join(data_home,dir,"imgs")
    xmls_path = os.path.join(data_home,dir,"xmls")
    xmls = os.listdir(xmls_path)
    for xml in tqdm(xmls):
        old_path = os.path.join(xmls_path,xml)
        new_path = os.path.join(xmls_path,dir+"_"+xml)
        os.rename(old_path,new_path)
        # old_img_path = os.path.join(imgs_path,xml.replace("xml","png"))
        # new_img_path = os.path.join(imgs_path,dir+"_"+xml.replace("xml","png"))
        # os.rename(old_img_path,new_img_path)


if __name__ == "__main__":
    
    # data_home = "F:/Datasets/securety/页面识别/全场景图标和文本检测"
    data_home = "F:/Datasets/securety/PageRec/all/origin"
    rename_img_xml(data_home,"wps")

    # dirs = os.listdir(data_home)
    # for dir in dirs:
    #     dir_path = os.path.join(data_home,dir)
    #     imgs_path = os.path.join(dir_path,"imgs")
    #     xmls_path = os.path.join(dir_path,"xmls")
    #     make_dirs(imgs_path)
    #     make_dirs(xmls_path)
    #     imgs = [f for f in os.listdir(dir_path) if os.path.splitext(f)[-1] in [".jpg",".png"]]
    #     for img in tqdm(imgs):
    #         img_name = dir+"_"+os.path.splitext(img)[0]
    #         # 原始图片路径
    #         src_im_path = os.path.join(dir_path,img)
    #         # 目标图片和xml保存路径
    #         img_path = os.path.join(imgs_path,img_name+".jpg")
    #         xml_path = os.path.join(xmls_path,img_name+".xml")
    #         try:
    #             im = np.array(Image.open(src_im_path).convert("RGB"))           # 直接转RGB
    #         except:
    #             continue
    #         h,w,c = im.shape

    #         # 调用第一版的检测和分类
    #         texts,icos = get_text_icos(im)
    #         tag_names = ["text"]*len(texts) + ["ico"] * len(icos)
    #         box_list = [[box[0],box[1],box[0]+box[2],box[1]+box[3]] for box,_ in texts] + [[box[0],box[1],box[0]+box[2],box[1]+box[3]] for box in icos]
    #         try:
    #             # 保存进xml文件
    #             write_xml(folder=dir,
    #                     img_name=img_name+".jpg",
    #                     path=img_path,
    #                     img_width=w,
    #                     img_height=h,
    #                     tag_num=len(texts)+len(icos),
    #                     tag_names=tag_names,
    #                     box_list=box_list,
    #                     save_path = xmls_path,
    #                     )
    #             # 移动图片
    #             shutil.move(src_im_path,img_path)
    #         except:
    #             continue
            




