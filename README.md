# STAS_detection

## 1. 安裝PaddlePaddle

根據作業系統選擇安裝PaddlePaddle，詳細內容請參考[PaddlePaddle安裝流程](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)

## 2. 安裝PaddleDetection

```
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
pip install -r requirements.txt
```

```
STAS_Detection
├── datasets/STAS
│   ├── ColorAdj.py
│   ├── DataAugmentForObejctDetection.py
│   ├── gen_train_list.py
│   ├── stas.yml
│   ├── STAS2stas.py
│   ├── SynStas.py
│   ├── xml_helper.py
│   |   ...
├── PaddleDetection
├── 2ans.py
|   ...
```

## 3. 準備訓練資料
安裝環境
```
pip install opencv-python scikit-learn
```

下載資料集放在datasets/STAS下
```
datasets/STAS/
├── Train_Annotation
│   ├── 00000000.xml
│   ├── 00000001.xml
│   |   ...
├── Train_Images
│   ├── 00000000.jpg
│   ├── 00000001.jpg
│   |   ...
|   ...
```
統一標記大小寫
```
cd datasets/STAS/
python STAS2stas.py
```
旋轉，裁切
參考 https://github.com/maozezhong/CV_ToolBox
```
python DataAugmentForObejctDetection.py
```
顏色深淺調整
```
python ColorAdj.py
```
圖片合成
```
python SynStas.py
```

產生train.txt,val.txt,label_list.txt
```
python gen_train_list.py
```

```
>>cat train.txt
Train_Images/00000000.jpg Train_Annotations/00000000.xml
Train_Images/00000001.jpg Train_Annotations/00000001.xml
Train_Images/00000002.jpg Train_Annotations/00000002.xml
...

>>cat val.txt
Train_Images/00000842.jpg Train_Annotations/00000842.xml
Train_Images/00000843.jpg Train_Annotations/00000843.xml
Train_Images/00000844.jpg Train_Annotations/00000844.xml
...

>>cat label_list.txt
stas
```

改寫stas.yml
```yaml
_BASE_: [
  '/path/to/PaddleDetection/configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_2x_coco.yml'
]
save_dir: output/cascade_rcnn_r50_vd_fpn_ssld_2x_stas

snapshot_epoch: 1

metric: VOC
map_type: 11point
num_classes: 1

TrainDataset:
  !VOCDataSet
    dataset_dir: /path/to/datasets/STAS
    anno_path: train.txt
    label_list: label_list.txt
    data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

EvalDataset:
  !VOCDataSet
    dataset_dir: /path/to/datasets/STAS
    anno_path: val.txt
    label_list: label_list.txt
    data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

TestDataset:
  !ImageFolder
    dataset_dir: /path/to/datasets/STAS
    anno_path: label_list.txt
```

## 4. Train
```
cd ../../PaddleDetection
python tools/train.py -c /path/to/datasets/STAS/stas.yml --eval
```

## 5. Test
```
# 執行前須先將stas.yml，metric: VOC 該行註解。

python tools/infer.py -c /path/to/datasets/STAS/stas.yml -o weights=output/cascade_rcnn_r50_vd_fpn_ssld_2x_stas/stas/best_model.pdparams --infer_dir /path/to/datasets/STAS/Private_Image/ --save_results True

cd ../
python 2ans.py
```
