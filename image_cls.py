from glob import glob

import cv2
import numpy as np

from sklearn.cluster import KMeans
import shutil
import os


# ===========================================================
#                          Step.1
# ===========================================================

# Train_Images = sorted(glob('Train_Images/*.jpg'))

# fw = open('image_cls.csv','w')

# for img_path in Train_Images:
#     img = cv2.imread(img_path)
#     hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#     str = f'{img_path},{np.median(hsv_img[:,:,0])},{np.median(hsv_img[:,:,1])},{np.median(hsv_img[:,:,2])}\n'
#     fw.write(str)
    
# fw.close()

# ===========================================================
#                          Step.2
# ===========================================================

# n_clusters = 2

# for k in range(n_clusters):
#     os.makedirs('cls_'+str(k),exist_ok=True)

# lines = []
# with open('image_cls.csv','r') as fr:
#     lines = fr.readlines()
#     fr.close

# X = []
# for line in lines:
#     data = [float(x) for x in line.split(',')[2:]]
#     X.append(data)
# X = np.asarray(X)

# kmeans = KMeans(n_clusters=n_clusters)
# kmeans.fit(X)
# new_dy = kmeans.predict(X)

# for line ,Y in zip(lines,new_dy):
#     src = line.split(',')[0]
#     dst = src.replace('Train_Images','cls_'+ str(Y))
#     shutil.copy(src,dst)

# ===========================================================
#                          Step.3
# ===========================================================
    
# cls_0 = sorted(glob('cls_0/*.jpg'))
# cls_1 = sorted(glob('cls_1/*.jpg'))

# def count_normalize(img_paths):

#     mean = [0,0,0]

#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#         mean[0] += np.mean(hsv_img[:,:,0])
#         mean[1] += np.mean(hsv_img[:,:,1])
#         mean[2] += np.mean(hsv_img[:,:,2])

#     return np.array(mean)/len(img_paths)

# mean_0 = count_normalize(cls_0)
# print(mean_0)

# mean_1 = count_normalize(cls_1)
# print(mean_1)

# ===========================================================
#                          Step.4
# ===========================================================

cls_0 = sorted(glob('cls_0/*.jpg'))
cls_1 = sorted(glob('cls_1/*.jpg'))

mean_0 = [119.04420972,  54.90843678, 219.68134062]
mean_1 = [146.90407024, 108.11722206, 201.23112486]

def cvt_img(img_paths, mean_A, mean_B):

    for img_path in img_paths:
        img = cv2.imread(img_path)
        
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv_img = hsv_img.astype(np.int32)

        fg_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        bg_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        fg_mask[(hsv_img > (0, 64, 0)).all(axis=2)] = 1
        bg_mask = 1 - fg_mask

        bg_mask_3 = cv2.merge([bg_mask,bg_mask,bg_mask])
        fg_mask_3 = cv2.merge([fg_mask,fg_mask,fg_mask])


        h = hsv_img[:,:,0]
        s = hsv_img[:,:,1] + int(mean_B[1] - mean_A[1])
        v = hsv_img[:,:,2] + int(mean_B[2] - mean_A[2])

        s[s>255] = 255
        s[s<0] = 255

        v[v>255] = 255
        v[v<0] = 255

        new_hsv_img = cv2.merge([h,s,v])

        new_hsv_img = new_hsv_img.astype(np.uint8)
        hsv_img = hsv_img.astype(np.uint8)

        new_hsv_img = np.bitwise_or(hsv_img * bg_mask_3,new_hsv_img * fg_mask_3)

        new_img = cv2.cvtColor(new_hsv_img, cv2.COLOR_HSV2BGR)
 
        # cv2.imshow('img',new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("Extend_3_Images/"+str(img_path.split('\\')[-1]), new_img)

cvt_img(cls_0, mean_0, mean_1)
cvt_img(cls_1, mean_1, mean_0)