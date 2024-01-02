import argparse
import os
import csv
# import glob
import sys
import json
import io
sys.path.append(os.getcwd())
# from PIL import Image
from functools import partial
import multiprocessing as mp
from multiprocessing import Process, Queue
import time


import lmdb
import openslide
# from torch.utils.dataloader

# from conf import camlon16
# from conf.brac import settings as brac_settings



# def tif2json(tif_path):
    # tif_path.replace('training', 'training_json').replace('.')
    # if 'tumor' in


# def write
# class Dataset:
    # d


# def write_single_wsi(wsi_path, json_path, settings):
def write_single_wsi(path, settings, q):
# def write_to_lmdb(path, settings):
# def write_to_lmdb(settings):
# def write_single_wsi(path, q):

        wsi_path, json_path = path
        t1 = time.time()
        count = 0
        # db_size = 1 << 40
        # env = lmdb.open(settings.patch_dir, map_size=db_size)
    # for wsi_path, json_path in get_file_path(settings):
        # data = json.load(open(json_path, 'r'))
        # count += len(data['coords'][0])
        # print(len(data['coords'][0]))

    # wsi_path, json_path = path

        basename = os.path.basename(wsi_path)
        # json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
        # print(wsi_path, json_path)

        # with env.begin(write=True) as txn:
            # txn.put()

        json_data = json.load(open(json_path))
        # filename = json_data['filename']
        # txn.put(json_data.encode(), img_buff)
        # print(filename)
        # print(json_data.keys())
        coords = json_data['coords']
        # print(len(coords))

        wsi = openslide.OpenSlide(wsi_path)

        print('reading image {}'.format(wsi_path))


        # with env.begin(write=True) as txn:
        # t1 = time.time()
        # count = 0
        for coord in coords[0]:
            # count += 1
            # print(coord)
            (x, y), level, (patch_size_x, patch_size_y) = coord
            coord = (int(x), int(y)), int(level), (int(patch_size_x), int(patch_size_y))
            # print(coord)
            # print(x, y, level, patch_size_x, patch_size_y)
            patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                basename=basename,
                x=x,
                y=y,
                level=level,
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y)
            # print(basename)
            # print(patch_id)
            try:
                patch = wsi.read_region(*coord).convert('RGB')
            except Exception as e:
                print(coord, wsi_path)
                print(e)
                raise ValueError('some thing wrong')

            # print(patch.size)
            # patch.save('tmp/{}.jpg'.format(count))
            img_byte_arr = io.BytesIO()
            patch.save(img_byte_arr, format='jpeg')
            img_byte_arr = img_byte_arr.getvalue()
            # print(type(img_byte_arr))
            # ss = Image.open(io.BytesIO(img_byte_arr))
            # print(ss)
            # ss.save('tmp')
            # ss.save('tmp/{}_after.jpg'.format(count))
            # print(patch_id.encode())
            # print(patch_id, count)
            # db_size = 1 << 40
            # env = lmdb.open(settings.patch_dir, map_size=db_size)
            # with env.begin(write=True) as txn:
                # txn.put(patch_id.encode(), img_byte_arr)
            # print(os.path.basename(wsi_path))
            # print(patch_id)
            q.put((patch_id.encode(), img_byte_arr))
            count += 1
            # if count % 100 == 0:
                # print((time.time() - t1) / count, wsi_path)


        print(time.time() - t1, len(coords[0]))

#def create_lmdb(data_set, lmdb_path):
#    #wsi_path = os.path.join()
#
#    # for i in glob.iglob(os.path.join(wsi_path, "**", '*.tif'), recursive=True):
#        # print(i)
#
#
#
## def camlon16_wsis(data_set, direction=-1):
#    if data_set == 'train':
#        dirs = camlon16.train_dirs
#    else:
#        dirs = camlon16.test_dirs
#
#
#
#
#    # wsis = []
#    # txn = lmdb.open(lmdb_path)
#
#    count = 0
#    for wsi_dir, json_dir in zip(dirs['wsis'], dirs['jsons']):
#        for wsi_path in glob.iglob(os.path.join(wsi_dir, '**', '*.tif'), recursive=True):
#            # print(wsi_path)
#            basename = os.path.basename(wsi_path)
#            json_path = os.path.join(json_dir, basename.replace('.tif', '.json'))
#            print(wsi_path, json_path)
#
#            # with env.begin(write=True) as txn:
#                # txn.put()
#
#            json_data = json.load(open(json_path))
#            # filename = json_data['filename']
#            # txn.put(json_data.encode(), img_buff)
#            # print(filename)
#            # print(json_data.keys())
#            coords = json_data['coords']
#            # print(len(coords))
#
#            wsi = openslide.OpenSlide(wsi_path)
#
#            print('saving image {}'.format(wsi_path))
#
#            # return
#
#            # with env.begin(write=True) as txn:
#            for coord in coords[0]:
#                count += 1
#                # print(coord)
#                (x, y), level, (patch_size_x, patch_size_y) = coord
#                # print(x, y, level, patch_size_x, patch_size_y)
#                patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
#                    basename=basename,
#                    x=x,
#                    y=y,
#                    level=level,
#                    patch_size_x=patch_size_x,
#                    patch_size_y=patch_size_y)
#                # print(basename)
#                # print(patch_id)
#                patch = wsi.read_region(*coord).convert('RGB')
#                # print(patch.size)
#                # patch.save('tmp/{}.jpg'.format(count))
#                img_byte_arr = io.BytesIO()
#                patch.save(img_byte_arr, format='jpeg')
#                img_byte_arr = img_byte_arr.getvalue()
#                # print(type(img_byte_arr))
#                # ss = Image.open(io.BytesIO(img_byte_arr))
#                # print(ss)
#                # ss.save('tmp')
#                # ss.save('tmp/{}_after.jpg'.format(count))
#                # print(patch_id.encode())
#                # print(patch_id, count)
#                # db_size = 1 << 40
#                # env = lmdb.open(lmdb_path, map_size=db_size)
#                # with env.begin(write=True) as txn:
#                #     txn.put(patch_id.encode(), img_byte_arr)
#
#
#
#
#
#            # import sys; sys.exit()
#
#
#
#            # import sys; sys.exit()
#            # print(set(coords[3]) - set(coords[6]))
#            # print(coor)
#            # for i in coords[0]:
#                # print(i)
#
#            # for c in coords[0]:
#            #     print(coords[3].index(c))
#
#            # print(len(coords[0]))
#            # import sys; sys.exit()


        # with env.begin(write=True) as txn:
        #     txn.put(patch_id.encode(), img_byte_arr)

# def write_to_lmdb(env, q):
#         # db_size = 1 << 40
#         # env = lmdb.open(lmdb_path, map_size=db_size)

#     with env.begin(write=True) as txn:
#         # for q
#         while True:
#             record = q.get()
#             if record is None:
#                 # time.sleep(1)
#                 # record = q.get()
#                 print('no records')
#                 return


#             txn.put(*record)
#             # if numsq is None:
#                 # return



def get_file_path(settings):
    # json_dir = brac_settings.train_dirs['jsons'][0]
    # wsi_dir = brac_settings.train_dirs['wsis'][0]
    csv_path = settings.file_list_csv
    with open(csv_path, 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            slide_id = row['slide_id']
            name = os.path.splitext(slide_id)[0]
            json_path = os.path.join(settings.json_dir, name + '.json')
            wsi_path = os.path.join(settings.wsi_dir, slide_id)

            yield wsi_path, json_path


    # for json_file in glob.iglob(os.path.join(json_dir, '**', '*.json'), recursive=True):

    #     basename = os.path.basename(json_file)
    #     wsi_path = os.path.join(wsi_dir, basename.replace('.json', '.svs'))
    #     # print(wsi_path, json_file)
    #     yield wsi_path, json_file


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True, default=None)

    return parser.parse_args()



if __name__ == '__main__':
    # lmdb_path = '/data/smb/syh/WSI_cls/TCGA_BRCA/patch_lmdb/'
    args = get_args_parser()
    if args.dataset == 'brac':
        from conf.brac import settings

    patch_path = settings.patch_dir
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    # pool = mp.Pool(processes=8)
    # pool.map(fn, get_file_path(settings))
    # m = mp.Manager()
    # q = m.Queue()

    # count = 0
    # for wsi_path, json_path in get_file_path(settings):
    #     data = json.load(open(json_path, 'r'))
    #     count += len(data['coords'][0])
    #     print(len(data['coords'][0]))
    # write_to_lmdb(settings)

    # print(count)

    q = Queue()
    fn = partial(write_single_wsi, settings=settings, q=q)
    # pool.apply_async(fn, get_file_path(settings))
    # pool.map(fn, get_file_path(settings))

    db_size = 1 << 40
    env = lmdb.open(settings.patch_dir, map_size=db_size)
    count = 0
    t1 = time.time()
    num_process = 16
    with env.begin(write=True) as txn:

        proc = []
        for path in get_file_path(settings):
            proc.append(
                Process(target=fn, args=(path,))
            )

            if len(proc) == num_process:
                for p in proc:
                    p.start()

                time.sleep(20)
                while True:
                    try:
                        # if after 10 seconds, still no data
                        # then means the process ends
                        record = q.get(timeout=10)
                        # with
                        txn.put(*record)
                        count += 1
                        # if num is None:
                            # break
                        # print(num[0].decode())
                        if count % 1000 == 0:
                            print('time', (time.time() - t1) / count, 'total', count)
                    except:
                        print('end of reading {} processes'.format(len(proc)))
                        break


                # wait untill process ends
                for p in proc:
                    p.join()

                # clear prc
                proc = []

        # if last p less len(proc)
        for p in proc:
            p.start()

        time.sleep(10)
        while True:
            try:
                # if after 10 seconds, still no data
                # then means the process ends
                record = q.get(timeout=10)
                # if num is None:
                    # break
                txn.put(*record)
                # print(num[0].decode())
                if count % 1000 == 0:
                    print('time', (time.time() - t1) / count, 'total', count)
            except:
                print('end of reading {} processes'.format(len(proc)))
                break

        for p in proc:
            p.join()


        # num = q.get()
        # print(num)
        # q.join()
        # proc = []






    # pool.close()
    # pool.join()
    # print(pool, 'cccccc')
    # while True:
        # print(q.get())

    # q = Queue()
    # producer = partial(write_single_wsi, q=q)
    # pathes = list(get_file_path_brac())[:10]

    # producers = [Process(target=producer, args=(path,)) for path in get_file_path_brac()]
    # producers = [Process(target=producer, args=(path,)) for path in pathes]
    # for p in producers:
    #     p.start()

    # db_size = 1 << 40
    # env = lmdb.open(lmdb_path, map_size=db_size)
    # print('???')
    # consumer = Process(target=write_to_lmdb, args=(env, q))
    # consumer.start()

    # consumer.join()

    # print("All done")



# wsi_path = '/data/ssd1/by/CAMELYON16/training/'
# json_path = '/data/ssd1/by/CAMELYON16/training_json/'
# lmdb_path = '/data/ssd1/by/CAMELYON16/training_lmdb'
# lmdb_path = '/data/ssd1/by/CAMELYON16/testing_lmdb'
# json_dir = brac_settings.train_dirs['json'][0]


# import time
# t1 = time.time()
# for count, path in enumerate(get_file_path_brac()):
#     write_single_wsi(path, lmdb_path)
#     print((time.time() - t1) / (count + 1))




# task executed in a worker process
# def task(identifier, value):
#     # report a message
#     print(f'Task {identifier} executing with {value}', flush=True)
#     # block for a moment
#     # sleep(value)
#     # return the generated value
#     return (identifier, value)

# items = [(i, i) for i in range(10)]


# def howmany_within_range_rowonly(row, minimum=4, maximum=8):
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count += 1
#     return count
# import time

# def howmany_within_range(row, minimum, maximum):
#     count = 0
#     for n in range(row):
#         if minimum <= n <= maximum:
#             count += 1

#     time.sleep(1)
#     # print(row)
#     return count

# print(mp.cpu_count())
# pool = mp.Pool(mp.cpu_count())

# results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in range(100)])
# pool.close()

# pool = mp.Pool(processes=4)
# pool.starmap(howmany_within_range, )

# fn = partial(write_single_wsi, lmdb_path=lmdb_path)
# # mp.set_start_method('fork')
# pool = mp.Pool(processes=16)
# pool.map(fn, get_file_path_brac())
# pool.close()
# for w, f in get_file_path_brac():
    # fn(w, f)

# create_lmdb(wsi_path, json_path, lmdb_path)
# create_lmdb('train', lmdb_path)
# lmbd_path = '/data/smb/syh/WSI_cls/TCGA_BRCA/patch_lmdb/'
# create_lmdb('test', lmdb_path)

# import multiprocessing
# import time

# def find_area_sq(x, data):
#     time.sleep(3)
#     # data = [data[i] + 1 for i in range(2)]
#     # for d in data:
#         # d += 1
#     data[0] += 100
#     print("AA", x*x)
#     print(data[:])

# def find_vol_cube(x, data):
#     print("AAA", x*x*x)

# data = multiprocessing.Array('i', 3)

# p1 = multiprocessing.Process(target=find_area_sq, args=(5, data))
# p2 = multiprocessing.Process(target=find_area_sq, args=(5, data))
# print(p1, p2)
# t1 = time.time()
# p1.start()
# p2.start()
# print(time.time() - t1)
# p1.join()
# p2.join()
# # print(time.time() - t1)
# print(p1.is_alive())
# print(data[:])

# from multiprocessing import Process, Value, Array

# def f(n, a):
#     n.value = 3.1415927
#     for i in range(len(a)):
#         a[i] = a[i] + 1

#     print(a[:])
# def add_a_cube(newlist, q):

#     for i in newlist:
#         q.put(i ** 3)
#         print(i ** 2)
#         time.sleep(2)

# def print_queue(q):
#     while not q.empty():
#         # print())
#         print(q.get(), 'get')

#     print('queue has empty!')



# def f(x):
#     return x*x

# from multiprocessing import Pool

# if __name__ == '__main__':
#     with Pool(processes=4) as pool:         # start 4 worker processes
#         result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously in a single process
#         print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow

#         print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"

#         it = pool.imap(f, range(10))
#         print(next(it))                     # prints "0"
#         print(next(it))                     # prints "1"
#         print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow

#         result = pool.apply_async(time.sleep, (10,))
#         print(result.get(timeout=1))

# if __name__ == '__main__':

#     mylist = [1,2,3,4]
#     q = multiprocessing.Queue()

#     p1 = multiprocessing.Process(target=add_a_cube, args=(mylist, q))
#     p2 = multiprocessing.Process(target=print_queue, args=(q, ))

#     p1.start()

#     p2.start()
#     p2.join()
#     p1.join()
    #num = Value('d', 0.0)
    #arr = Array('i', range(10))

    ## p = Process(target=f, args=(num, arr))
    ## p1 = Process(target=f, args=(num, arr))
    #p = multiprocessing.Process(target=find_area_sq, args=(5, arr))
    ## p1 = multiprocessing.Process(target=find_area_sq, args=(5, arr))
    #p.start()
    ## p1.start()
    #p.join()
    ## p1.join()

    #print(num.value)
    #print(arr[:])