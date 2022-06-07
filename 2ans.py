import json
import datetime


date = f'{datetime.date.today().month:02d}{datetime.datetime.today().day:02d}'
modelname = 'cascade_rcnn_r50_vd_fpn_ssld_2x_stas'
threshold = 0.05


with open(f'PaddleDetection/output/bbox.json','r') as fr:
    json_data = json.load(fr)


output = {}
for i in range(131):
    output[f'Public_{i:08d}.jpg'] = []

for data in json_data:
    image_id = data['image_id']
    [x_min,y_min,box_width,box_height] = [int(x) for x in data['bbox']]

    score = float(f"{data['score']:.05f}")
    
    if score < threshold:
        print(f'Public_{image_id:08d}.jpg {[x_min,y_min,x_min+box_width,y_min+box_height,score]}')
        continue

    output[f'Public_{image_id:08d}.jpg'].append([x_min,y_min,x_min+box_width,y_min+box_height,score])

with open(f'upload/{date}_{modelname}.json','w') as fw:
    json.dump(output,fw)

# output = {}
# for i in range(184):
#     output[f'Private_{i:08d}.jpg'] = []

# for data in json_data:
#     image_id = data['image_id']
#     [x_min,y_min,box_width,box_height] = [int(x) for x in data['bbox']]

#     score = float(f"{data['score']:.05f}")
    
#     if score < threshold:
#         print(f'Private_{image_id:08d}.jpg {[x_min,y_min,x_min+box_width,y_min+box_height,score]}')
#         continue

#     output[f'Private_{image_id:08d}.jpg'].append([x_min,y_min,x_min+box_width,y_min+box_height,score])

# with open(f'upload/{date}_{modelname}.json','w') as fw:
#     json.dump(output,fw)