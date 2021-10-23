import os
import pickle

fp_class_path = "D:/1Pjlab/ModelSimulator/output/fp_diff"
cls_1_path = os.path.join(fp_class_path, 'fp_clss_1.pkl')
cls_2_path = os.path.join(fp_class_path, 'fp_clss_2.pkl')

with open(cls_1_path, 'rb') as f1:
    fp_clss_1 = pickle.load(f1)
with open(cls_2_path, 'rb') as f2:
    fp_clss_2 = pickle.load(f2)

fp_difficult = 2 - fp_clss_1['cls_result'] - fp_clss_2['cls_result']    # difficult：0表示fp最容易本分辨出来（两个MLP得分都是1）；1表示fp不太容易被分辨出来（两个MLP得分一个1，一个0）；2表示fp很难被分辨出来（两个MLP得分都为0）

fp_diff = {'image': fp_clss_1['image'], 'dtbox_id': fp_clss_1['dtbox_id'], 'difficult': fp_difficult}

print(fp_diff)

with open("D:/1Pjlab/ModelSimulator/data/fp_difficult.pkl", "wb") as f:
    pickle.dump(fp_diff, f)