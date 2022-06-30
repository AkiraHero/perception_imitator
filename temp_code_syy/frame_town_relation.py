import os
import pickle

path = 'E:/PJLAB_Experiment/Data/carla/carla_new/Town_frame_all'
all_town_file = os.listdir(path)
save = {}

print(all_town_file)
for town_file in all_town_file:
    town_name = town_file.replace('.txt', '')
    print(town_name)

    id_file = open(os.path.join(path, town_file))
    for id in id_file.readlines():
        id = id.replace('\n', '')
        save['%s' %id] = town_name
    
with open("./data/frame2town.pkl", "wb") as f:
    pickle.dump(save, f)