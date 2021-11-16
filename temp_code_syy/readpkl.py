import pickle

with open('D:/1Pjlab/ADModel_Pro/data/fp_difficult.pkl', 'rb') as f:
    fp_difficult = pickle.load(f)

print(fp_difficult.keys())
print(fp_difficult['image'][25], fp_difficult['dtbox_id'][25], fp_difficult['difficult'][25])
