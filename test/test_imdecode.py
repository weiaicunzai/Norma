import cv2
import time
import tqdm
import numpy

if __name__ == "__main__":
    # img = cv2.imread('image.jpg')
    img = cv2.imread('/data/hdd1/by/tmp_folder/lmdb_files/Ldbm_task/cat.jpeg')
    # img = cv2.resize(img, (768, 1280))
    # img = cv2.resize(img, (512, 512))
    img = cv2.resize(img, (256, 256))
    _, imgbuf = cv2.imencode('.jpg', img)
    n_try = 500

    times = numpy.zeros((n_try,), dtype=numpy.float32)
    for i in tqdm.tqdm(range(n_try)):
        begin = time.time()
        _ = cv2.imdecode(imgbuf, 1)
        times[i] = time.time() - begin

    times = numpy.sort(times)[1:-1]   # exclude minimum and maximum
    avg = 1000 * numpy.mean(times)
    std = 1000 * numpy.std(times)
    print("mean={}ms, stdev={}ms".format(avg, std))