import pickle

with open('D:/1Pjlab/ADModel_Pro/data/img_fp_difficult.pkl', 'rb') as f:
    fp_difficult = pickle.load(f)

print(fp_difficult[0])