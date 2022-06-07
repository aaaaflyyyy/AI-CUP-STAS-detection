# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
##############################################################

import random
import cv2
import os
import math
import numpy as np

class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5, 
                crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, 
                cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
    
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, bbox[4]])
        
        return rot_img, rot_bboxes

    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- 裁剪boundingbox ----------------------
        #裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min, bbox[4]])
        
        return crop_img, crop_bboxes

    # def dataAugment(self, img, bboxes):
    #     '''
    #     图像增强
    #     输入:
    #         img:图像array
    #         bboxes:该图像的所有框坐标
    #     输出:
    #         img:增强后的图像
    #         bboxes:增强后图片对应的box
    #     '''

    #     # Rotate
    #     angle = random.sample([15, 30, 330,345],1)[0]
    #     scale = 1
    #     img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
    #     # Crop
    #     # img, bboxes = self._crop_img_bboxes(img, bboxes)

    #     return img, bboxes
            

if __name__ == '__main__':

    ### test ###

    from xml_helper import *

    dataAug = DataAugmentForObjectDetection()

    source_pic_root_path = './Train_Images'
    source_xml_root_path = './Train_Annotations'

    rotate_pic_root_path = './Rotate_Images'
    rotate_xml_root_path = './Rotate_Annotations'

    crop_pic_root_path = './Crop_Images'
    crop_xml_root_path = './Crop_Annotations'

    if not os.path.exists(rotate_pic_root_path):
        os.makedirs(rotate_pic_root_path)

    if not os.path.exists(rotate_xml_root_path):
        os.makedirs(rotate_xml_root_path)

    if not os.path.exists(crop_pic_root_path):
        os.makedirs(crop_pic_root_path)

    if not os.path.exists(crop_xml_root_path):
        os.makedirs(crop_xml_root_path)
    
    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')

            coords = parse_xml(xml_path)
            img = cv2.imread(pic_path)

            # auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

            # Rotate
            angle = random.sample([15, 30, 330,345],1)[0]
            rotate_img, rotate_bboxes = dataAug._rotate_img_bbox(img, coords, angle, 1)

            generate_xml(file,rotate_bboxes,rotate_img.shape,rotate_xml_root_path)
            cv2.imwrite(f'{rotate_pic_root_path}/{file}',rotate_img)

            # Crop
            crop_img, crop_bboxes = dataAug._crop_img_bboxes(img, coords)

            generate_xml(file,crop_bboxes,crop_img.shape,crop_xml_root_path)
            cv2.imwrite(f'{crop_pic_root_path}/{file}',crop_img)

