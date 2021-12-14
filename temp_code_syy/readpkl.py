import pickle

with open('D:/1Pjlab/Datasets/kitti-pvrcnn-epch8369-392dropout/default/dropout-iter-0-20211211-140118/gt_dt_matching_res.pkl', 'rb') as f:
    gt_dt_matching_res = pickle.load(f)
with open('D:/1Pjlab/Datasets/kitti-pvrcnn-epch8369-392dropout/default/dropout-iter-0-20211211-140118/result.pkl', 'rb') as f:
    result = pickle.load(f)

id = 4
print(gt_dt_matching_res['bev'][id])
print(gt_dt_matching_res['gt_annos'][id])
print(result[id]['boxes_lidar'])

