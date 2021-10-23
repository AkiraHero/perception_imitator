import os
import pickle
from typing import Counter

with open("D:/1Pjlab/ModelSimulator/data/fp_difficult.pkl", 'rb') as f:
    fp_difficult = pickle.load(f)
print(fp_difficult.keys())
print(fp_difficult['image'])
print(fp_difficult['dtbox_id'])
print(fp_difficult['difficult'])
print('-'*10)

all_img_fp_difficult = []
for img_id in range(7481):   # 图片序号
    have_fp = 0
    num_all_fp = 0
    num_easy_fp = 0
    num_hard_fp = 0

    dtbox_id_index = [i for i,v in enumerate(fp_difficult['image']) if v==img_id]   # 获取该图中的所有fp对应的索引号
    if len(dtbox_id_index) != 0:
        have_fp = 1
        num_all_fp = len(dtbox_id_index)

        difficult = fp_difficult['difficult'][dtbox_id_index]
        for j in Counter(difficult).keys():
            if j in [1, 2]:
                num_hard_fp = num_hard_fp + Counter(difficult)[j]
            elif j in [0]:
                num_easy_fp = num_easy_fp + Counter(difficult)[j]
    else:
        pass
    
    img_fp_difficult = {'image': img_id, 'have_fp': have_fp, 'all_fp': num_all_fp, 'easy_fp': num_easy_fp, 'hard_fp': num_hard_fp}
    all_img_fp_difficult.append(img_fp_difficult)

with open("D:/1Pjlab/ModelSimulator/data/img_fp_difficult.pkl", "wb") as f:
    pickle.dump(all_img_fp_difficult, f)
