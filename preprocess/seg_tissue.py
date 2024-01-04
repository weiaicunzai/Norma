import openslide
import cv2
import numpy as np
import glob
import os
import csv


def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]


def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

# def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False,
def segment_tissue(wsi, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False,
                            filter_params={'a_t':100, 'a_h': 16, 'max_n_holes': 8}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
            filter_params:  a_t : area filter for tissue, default 100
                            a_h : area filter for holes,  default 16
                            max_n_holes: maximum number of holes to consider for each tissue contourarea filter for holes: default 8
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            # print('hhhhh', hierarchy_1.shape)
            all_holes = []

            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # print('h', holes.shape, holes)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # print(hole_areas, a)
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                # print(11, a, filter_params)
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)


            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    # print(hole.shape)
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        # img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        # print(wsi.level_dimensions)
        img = np.array(wsi.read_region((0,0), seg_level, wsi.level_dimensions[seg_level]))

        # cv2.imwrite('img.png', img)

        # converted from RGB to the HSV colour space.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space


        # A binary mask for the tissue regions (foreground) is computed
        # based on thresholding the saturation channel of the image after median blurring to
        # smooth the edges and is followed by additional morphological closing to fill small
        # gaps and holes.
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)



        #return img
        # return img_otsu
        scale = (wsi.level_downsamples[seg_level], wsi.level_downsamples[seg_level])
        # print(scale)  64  64
        # print(scale, wsi.level_downsamples)
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # at / scale * ref_patch

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours
        # print(contours.shape)
        # print(len(contours))
        #for i in contours:
        #     print(i.shape)
        #  [num_points, 1, 2]   2 is (x, y)

        # print(type(hierarchy))
        #print(hierarchy.shape) (1, 501, 4)
        # print(contours)

        # return img_otsu
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
        # print(foreground_contours)

        contours_tissue = scaleContourDim(foreground_contours, scale)
        holes_tissue = scaleHolesDim(hole_contours, scale)

        #exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

        contours_tissue = [contours_tissue[i] for i in contour_ids]
        holes_tissue = [holes_tissue[i] for i in contour_ids]

        # print(contours_tissue)
        # for i in contours_tissue:
        #     print(i.shape)
        mask =  np.zeros(img.shape[:2])

        mask = cv2.drawContours(mask, foreground_contours, -1, 255, -1)

        return mask



def segment_tissues(src_dir, dest_dir):
    for wsi_path in glob.iglob(os.path.join(src_dir, '**', '*.tif'), recursive=True):
        print(wsi_path)
        wsi = openslide.OpenSlide(wsi_path)
        mask = segment_tissue(wsi, seg_level=6, use_otsu=True)

        if not mask.sum():
            # for scale in [2, 4, 8, 16, 32, 64]:
            # control the granularity of segementation, some wsis are small
            # so we need to use small scales like 2 to segmenta some thing
            for scale in (2**i for i in range(1, 6)):
                filter_params={'a_t':100 // scale, 'a_h': 16, 'max_n_holes': 8}
                mask = segment_tissue(wsi, seg_level=6, use_otsu=True, filter_params=filter_params)

                if mask.sum() > 4000000:
                    break

        assert mask.sum() > 0

        base_name = os.path.basename(wsi_path)
        # print(os.path.join(dest_dir, base_name))

        if not os.path.exists(os.path.join(dest_dir)):
             os.makedirs(dest_dir)

        print('write {}'.format(os.path.join(dest_dir, base_name.replace('.tif', '.png'))))
        cv2.imwrite(os.path.join(dest_dir, base_name.replace('.tif', '.png')), mask)


# def segment_a_single_wsi(wsi_path, dest_dir, seg_level):
def segment_a_single_wsi(wsi_path, dest_dir):
    # wsi_name

    wsi = openslide.OpenSlide(wsi_path)
    print(wsi.level_dimensions)
    seg_level = 0
    for idx, dims in enumerate(wsi.level_dimensions):
        if max(dims) < 10000:
            seg_level = idx
            break

    mask = segment_tissue(wsi, seg_level=seg_level, use_otsu=True)

    if not mask.sum():
        # for scale in [2, 4, 8, 16, 32, 64]:
        # control the granularity of segementation, some wsis tissue areas
        # are small in the wsi so we need to use small scales
        # like 2 to segmenta tissue
        for scale in (2**i for i in range(1, 6)):
            filter_params={'a_t':100 // scale, 'a_h': 16, 'max_n_holes': 8}
            mask = segment_tissue(wsi, seg_level=seg_level, use_otsu=True, filter_params=filter_params)

            # if masked pixel is larger than 400000
            # if mask.sum() > 4000000:
            if mask.sum() / (mask.shape[0] * mask.shape[1]) > 0.1:
                break

    assert mask.sum() > 0

    base_name = os.path.basename(wsi_path)
    # print(os.path.join(dest_dir, base_name))

    if not os.path.exists(os.path.join(dest_dir)):
         os.makedirs(dest_dir)

    file_name = os.path.join(dest_dir,  os.path.splitext(base_name)[0] + '.png')
    # print('write {}'.format(os.path.join(dest_dir, base_name.replace('.tif', '.png'))))
    print('write {}'.format(file_name))
    # cv2.imwrite(os.path.join(dest_dir, base_name.replace('.tif', '.png')), mask)
    cv2.imwrite(file_name, mask)


def level_0_mag(wsi):
    if 'aperio.AppMag' in wsi.properties.keys():
        level_0_magnification = int(float(wsi.properties['aperio.AppMag']))
    elif 'openslide.mpp-x' in wsi.properties.keys():
        level_0_magnification = 40 if int(float(wsi.properties['openslide.mpp-x']) * 10) == 2 else 20
    else:
        # print('ccccccc????')
        # raise ValueError('no information????')
        level_0_magnification = 40

    return level_0_magnification

def get_filename(source_csv_file, wsi_dir):
    with open(source_csv_file, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            # print(', '.join(row))
            # print(row[2])
            # print(row['slide_id'])
            # print(row)
            yield os.path.join(wsi_dir, row['']+ '.svs')
            # row[]

if __name__ == '__main__':


    path = '/data/smb/syh/WSI_cls/TCGA_BRCA/img'
    save_path = '/data/smb/syh/WSI_cls/TCGA_BRCA/mask'


    # for i in glob.iglob(os.path.join(path, '**', '*.svs'), recursive=True):
    # for i in get_filename(source_csv_file='/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/splits/10foldcv_subtype/tcga_brca/splits_0_bool.csv', wsi_dir=path):
    wsi_lists = get_filename(source_csv_file='/data/hdd1/by/HIPT/2-Weakly-Supervised-Subtyping/splits/10foldcv_subtype/tcga_brca/splits_0_bool.csv', wsi_dir=path)
    # print(len(list(wsi_lists)))
    # import sys; sys.exit()
        # segment_a_single_wsi()
        # print(i)
        # i = os.path.join(path, i)
        # print(i)
        # wsi = openslide.OpenSlide(i)
        # print(wsi.level_dimensions)
        # print(wsi.properties['Mag'])
        # print()
        # print(wsi.level_count)
        # print(wsi.level_downsamples)
        # print(wsi.level_downsamples)
        # print(wsi.level_count)
        # print(wsi.level_dimensions)
        # wsi.read_region()
        # print(level_0_mag(wsi))
        # segment_a_single_wsi(wsi_path=i, dest_dir=save_path, seg_level=)

        # import sys; sys.exit()
        # print(i)
    # pool = multiprocessing.Pool(processes=32)
    # pool.map(segment_a_single_wsi, itertools.izip(wsi_lists, itertools.repeat(save_path)))
    # pool.map(partial(segment_a_single_wsi, dest_dir=save_path), wsi_lists)


    # segment_a_single_wsi(wsi_path=i, dest_dir=save_path)


        # import sys; sys.exit()

# from random import randint

# path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/test_{:03d}.tif'.format(randint(1, 100))

    # path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/tumor/'
    # dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/tumor/'

    # segment_tissues(path, dest_dir)

    # path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training/normal/'
    # dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/training_mask/normal/'

    # segment_tissues(path, dest_dir)

    # path = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/images/'
    # dest_dir = '/data/yunpan/syh/PycharmProjects/CGC-Net/data_baiyu/CAMELYON16/testing/masks'

    # segment_tissues(path, dest_dir)


# wsi = openslide.OpenSlide(path)

# mask = segment_tissue(wsi, seg_level=6, use_otsu=True)

# cv2.imwrite('test.png', mask)
