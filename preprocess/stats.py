import os
import csv
import shutil



csv_path = '/data/hdd1/by/tmp_folder/datasets/dataset_csv/brac/brac.csv'
duoyu_img_path = '/data/smb/syh/WSI_cls/brac/img_duoyu'
img_path = '/data/smb/syh/WSI_cls/brac/img'


slides = []
with open(csv_path, 'r') as csv_file:
    for row in csv.DictReader(csv_file):
        slide_id = row['slide_id']
        slides.append(slide_id)



for img in os.listdir(img_path):
    if img not in slide_id:
        src_dir = os.path.join(img_path, img)
        dest_dir = os.path.join(duoyu_img_path, img)
        # shutil.move()
        print(src_dir, dest_dir)
