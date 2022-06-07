from glob import glob

data = 'val'

Train_Annotations = sorted(glob('Train_Annotations/*.xml'))
Train_Images = sorted(glob('Train_Images/*.jpg'))

Rotate_Annotations = sorted( glob('Rotate_Annotations/*.xml'))
Rotate_Images = sorted(glob('Rotate_Images/*.jpg'))

Crop_Annotations = sorted( glob('Crop_Annotations/*.xml'))
Crop_Images = sorted(glob('Crop_Images/*.jpg'))

SynStas_Annotations = sorted( glob('SynStas_Annotations/*.xml'))
SynStas_Images = sorted(glob('SynStas_Images/*.jpg'))

ColorAdj_Images = sorted(glob('ColorAdj_Images/*.jpg'))

TRAIN_DATA = int(len(Train_Annotations)*0.8)

print(f'gen {data} data txt file.')

if data == 'train':
    train_file = open('train.txt','w')

    for img,anno in zip(Train_Images[:TRAIN_DATA],  Train_Annotations[:TRAIN_DATA]):
        train_file.write(f'{img} {anno}\n'.replace("\\","/"))

    # for img,anno in zip(SynStas_Images[:TRAIN_DATA],   SynStas_Annotations[:TRAIN_DATA]):
    #     train_file.write(f'{img} {anno}\n'.replace("\\","/"))

    # for img,anno in zip(ColorAdj_Images[:TRAIN_DATA],   Train_Annotations[:TRAIN_DATA]):
    #     train_file.write(f'{img} {anno}\n'.replace("\\","/"))

    # for img,anno in zip(Crop_Images[:TRAIN_DATA],   Crop_Annotations[:TRAIN_DATA]):
    #     train_file.write(f'{img} {anno}\n'.replace("\\","/"))

    for img,anno in zip(Rotate_Images, Rotate_Annotations):
        train_file.write(f'{img} {anno}\n'.replace("\\","/"))
    train_file.close()

# val
elif data == 'val':
    val_file = open('val.txt','w')

    for img,anno in zip(Train_Images[TRAIN_DATA:],Train_Annotations[TRAIN_DATA:]):
        val_file.write(f'{img} {anno}\n'.replace("\\","/"))
    val_file.close()

label_list = open('label_list.txt','w')
label_list.write('stas')
label_list.close()