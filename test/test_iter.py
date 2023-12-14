# # path = '/home/baiyu/miniconda3/envs/torch1.13/lib/python3.10/site-packages/torch/'

# import os
# import time


# for root, dirs, files in os.walk(path, topdown=False):
#     # print(root, dirs)
#     # print(root, dirs)
#     # print(root, dirs, files)
#     for f in files:
#         if f.split('.')[-1] == 'py':
#             # print(f)
#             file_path = os.path.join(root, f)
#             st_time = os.stat(file_path).st_mtime
#             #print(time.ctime(os.stat(os.path.join(root, f)).st_mtime))
#             print(time.ctime(st_time), file_path)


    # break



# import glob
# from multiprocessing import Pool, TimeoutError
# from PIL import Image

# def read_img(img_path):
#     image = Image.open(img_path).convert('RGB')
#     return image


# # data_dir = '/data/ssd1/by/folowler'
# data_dir = '/data/ssd1/xuxinan/flower102jpg'


#     # with Pool(processes=4) as pool:
# # lists = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
# lists = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)

# import time
# for num_proc in [1, 4]:
#     t1 = time.time()
#     with Pool(processes=num_proc) as pool:
#         pool.map(read_img, lists)

#     t2 = time.time()
#     print(t2 - t1)


# with open('sss.txt') as f:
#     for line in f.readlines():
#         # print(line)
#         if 'The following' in line:
#             # print(line)
#             continue
#         if line.strip():
#             pkg = line.split()[0].replace('~', '')
#             # print(pkg,
#             os.system('conda list | grep {}'.format(pkg))
        # print(line.strip().split())