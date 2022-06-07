from glob import glob 

Annotations = sorted(glob('Rotate_Annotations/*.xml'))

for annofile in Annotations:
    with open(f'{annofile}','r') as fr :
        data = fr.read()
        with open(f'{annofile}','w') as fw :
            fw.write(data.replace('STAS','stas'))
    
    