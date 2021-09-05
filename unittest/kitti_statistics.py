
import os

dir = "/home/xlju/data/kitti_3d_object/data_object_label_2/training/label_2"
nums = []
interested_nums_sum = []

interested_cls = ['Car', 'Pedestrian', 'Cyclist']

for i in os.listdir(dir):
    p = os.path.join(dir, i)
    with open(p, 'r') as f:
        ls = f.readlines()
        nums.append(len(ls))
        interested_nums_dict_frm = {
            'Car': 0,
            'Pedestrian': 0,
            'Cyclist': 0
        }
        for i in ls:
            for cls in interested_cls:
                if cls in i:
                    interested_nums_dict_frm[cls] += 1

        interested_nums_sum.append(sum(interested_nums_dict_frm.values()))


print("max obj in same frm:", max(nums))
print("max interested obj in same frm:", max(interested_nums_sum))


