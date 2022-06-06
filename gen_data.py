import cv2
import xml.etree.ElementTree as ET
from glob import glob
import random

def overlap_area(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    iou_area = iou_w * iou_h
    # all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    # return max(iou_area/all_area, 0)
    return iou_area



Images = sorted(glob('Train_Images/*.jpg'))
Annotations = sorted(glob('Train_Annotations/*.xml'))


for img_file, anno_file in zip(Images,Annotations):
    print(img_file, anno_file)

    img = cv2.imread(img_file)
    new_img = img.copy()

    bbox_list = []
    new_bbox_list = []
    obj_queue = []

    tree = ET.parse(anno_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin

        bbox_list.append([xmin, ymin, width, height])
        new_bbox_list.append([xmin, ymin, width, height])
        # cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+height),(0,255,0),3)


    for bbox in bbox_list:

        [xmin, ymin, width, height] = bbox

        new_xmin = random.randint(0, 1716 - width)
        new_ymin = random.randint(0, 942 - height)

        addbbox = True
        for ori_bbox in new_bbox_list:
            if overlap_area([new_xmin,new_ymin,width,height],ori_bbox) / (width*height)  > 0.5:
                addbbox = False
                break

        if addbbox:
            new_img[new_ymin:new_ymin+height,new_xmin:new_xmin+width,:] = img[ymin:ymin+height,xmin:xmin+width,:]
            new_bbox_list.append([new_xmin,new_ymin,width,height])
            # cv2.rectangle(img,(new_xmin,new_ymin),(new_xmin+width,new_ymin+height),(255,0,0),3)

            new_obj_bndbox = ET.Element("bndbox")
            ET.SubElement(new_obj_bndbox,"xmin")
            ET.SubElement(new_obj_bndbox,"ymin")
            ET.SubElement(new_obj_bndbox,"xmax")
            ET.SubElement(new_obj_bndbox,"ymax")
            new_obj = ET.Element("object")
            ET.SubElement(new_obj,"name")
            ET.SubElement(new_obj,"pose")
            ET.SubElement(new_obj,"truncated")
            ET.SubElement(new_obj,"difficult")
            new_obj.append(new_obj_bndbox)

            new_obj.find('name').text = 'stas'
            new_obj.find('pose').text = 'unspecified'
            new_obj.find('truncated').text = '0'
            new_obj.find('difficult').text = '0'

            new_obj.find('bndbox').find('xmin').text = str(new_xmin)
            new_obj.find('bndbox').find('ymin').text = str(new_ymin)
            new_obj.find('bndbox').find('xmax').text = str(new_xmin + width)
            new_obj.find('bndbox').find('ymax').text = str(new_ymin + height)

            obj_queue.append(new_obj)


    new_tree = ET.parse(anno_file)
    new_root = new_tree.getroot()

    for new_obj in obj_queue:
        new_root.append(new_obj)
        
    new_tree.write(f"{anno_file.replace('Train','Gen')}", encoding='UTF-8')
    cv2.imwrite(f"{img_file.replace('Train','Gen')}", new_img)

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
