import os
import glob
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
import openslide






def parse_anno(xml_path):
    annos = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(root)
    print(xml_path)
    for anno in root.iter('Annotation'):
        res = []
        for coord in anno.iter('Coordinate'):
            res.append([float(coord.attrib['X']), float(coord.attrib['Y'])])

        annos.append(res)
    return annos

def xml_files(anno_dir):
    for xml_path in glob.iglob(os.path.join(anno_dir, '**', '*.xml'), recursive=True):
        yield xml_path

def xml2json(wsi_dir, xml_path):
    basename = os.path.basename(xml_path).replace('xml', 'tif')
    return os.path.join(wsi_dir, basename)


def xml2tif(wsi_dir, xml_path):
    basename = os.path.basename(xml_path).replace('xml', 'tif')
    return os.path.join(wsi_dir, basename)

def draw_lesion_mask(image_shape, coords):
    coords = np.array(coords)
    coords.reshape(-1, 1, 2)
    cv2.polylines()

def xml2json(json_dir, xml_path):
    basename = os.path.basename(xml_path).replace('xml', 'json')
    return os.path.join(json_dir, basename)


#anno_dir = '/data/ssd1/by/CAMELYON16/training/lesion_anno'
#wsi_dir = '/data/ssd1/by/CAMELYON16/training/tumor/'
#json_dir = '/data/ssd1/by/CAMELYON16/training_json/tumor/patch_size_512_at_mag_20/'
## dest_file = '/data/ssd1/by/CAMELYON16/training_json/patch_label_patch_size_512_at_mag_20.json'
#dest_file = '/data/ssd1/by/CAMELYON16/training_json/tumor/patch_size_512_at_mag_20_patch_label'
#mask_dir = '/data/ssd1/by/CAMELYON16/training_mask/tumor/'

anno_dir = '/data/ssd1/by/CAMELYON16/testing/lesion'
wsi_dir = '/data/ssd1/by/CAMELYON16/testing/images/'
json_dir = '/data/ssd1/by/CAMELYON16/testing/jsons/patch_size_512_at_mag_20/'
dest_file = '/data/ssd1/by/CAMELYON16/testing/jsons/patch_size_512_at_mag_20_patch_label'
mask_dir = '/data/ssd1/by/CAMELYON16/testing/masks/'

def xml2mask(xml_path):
    basename = os.path.basename(xml_path).replace('xml', 'png')
    return os.path.join(mask_dir, basename)

def draw_mask(wsi_dir, xml_dir):
    level = 4
    count = 82
    avg_area = 0
    for idx, xml_path in enumerate(xml_files(xml_dir)):
        if count != idx + 1:
            continue

        wsi_path = xml2tif(wsi_dir, xml_path)
        wsi = openslide.OpenSlide(wsi_path)
        dims = wsi.level_dimensions[level]
        downsample_factor = wsi.level_downsamples[level]
        img = wsi.read_region((0,0), level, dims)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        mask_path = xml2mask(xml_path)
        mask = cv2.imread(mask_path, -1)
        print('mask', np.unique(mask))

        resized_mask = cv2.resize(mask, dims, interpolation=cv2.INTER_NEAREST)
        print(np.unique(resized_mask))


        coords = parse_anno(xml_path)

        tmp = []
        for region in coords:
            region = np.array(region)
            region = region.reshape(-1, 1, 2)
            region = region / downsample_factor
            tmp.append(region.astype(np.int32))

        # img = cv2.polylines(cv_img, tmp, False, (0, 255, 255))
        img = cv2.polylines(cv_img, tmp, False, (0, 255, 255))

        alpha = 0.8
        mask_bgr = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        # img = cv2.addWeighted(img, alpha, resized_mask, 1 - alpha, 0)
        img = cv2.addWeighted(img, alpha, mask_bgr, 1 - alpha, 0)

        # template = cv2.polylines(template, tmp, False, 255)
        # # print(template.sum() / 255, resized_mask.sum() / 255)
        # # print(np.unique(template),  np.unique(resized_mask))


        template = np.zeros(resized_mask.shape[:2])
        # template = cv2.polylines(template, tmp, isClosed=True, color=255)
        template = cv2.fillPoly(template, tmp, color=255)
        print(np.unique(template), np.unique(resized_mask))
        intersection = template * resized_mask / 255
        # # print(template.shape, resized_mask.shape)
        # # print(intersection.max(), resized_mask.max())
        # print(
        area_cc = (intersection == 255).sum() / (resized_mask == 255).sum()
        avg_area += area_cc

        cv2.imwrite('tmp/interseciong.png', intersection)
        cv2.imwrite('tmp/mask.png', resized_mask)
        cv2.imwrite('tmp/template.png', template)
        cv2.imwrite('tmp/test.jpg', img)


        # coords = np.array(coords)
        # coords.reshape(-1, 1, 2)



    print(avg_area / count)

def write_patch_label(wsi_dir, xml_dir, json_dir, dest_dir):
    res = {}

    # for xml_path in xml_files(xml_dir):
    #     wsi_path = xml2tif(wsi_dir, xml_path)

    level = 1
    count = 110
    avg_area = 0
    for idx, xml_path in enumerate(xml_files(xml_dir)):
        num_patch = 0
        #if count != idx:
        #    continue

        wsi_path = xml2tif(wsi_dir, xml_path)
        wsi = openslide.OpenSlide(wsi_path)
        dims = wsi.level_dimensions[level]
        downsample_factor = wsi.level_downsamples[level]
        # img = wsi.read_region((0,0), level, dims)
        # cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # mask_path = xml2mask(xml_path)
        # mask = cv2.imread(mask_path, -1)
        # print('mask', np.unique(mask))

        # resized_mask = cv2.resize(mask, dims, interpolation=cv2.INTER_NEAREST)
        # print(np.unique(resized_mask))


        coords = parse_anno(xml_path)

        tmp = []
        for region in coords:
            region = np.array(region)
            region = region.reshape(-1, 1, 2)
            region = region / downsample_factor
            tmp.append(region.astype(np.int32))

        # img = cv2.polylines(cv_img, tmp, False, (0, 255, 255))


        template = np.zeros(dims[::-1])
        template = cv2.fillPoly(template, tmp, color=255)

        ############
        # template_draw = template.copy()
        ####################

        # img = cv2.polylines(cv_img, tmp, False, (0, 255, 255))

        # alpha = 0.8
        # mask_bgr = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        # img = cv2.addWeighted(img, alpha, resized_mask, 1 - alpha, 0)
        # img = cv2.addWeighted(img, alpha, mask_bgr, 1 - alpha, 0)

        # template = cv2.polylines(template, tmp, False, 255)
        # # print(template.sum() / 255, resized_mask.sum() / 255)
        # # print(np.unique(template),  np.unique(resized_mask))


        # template = np.zeros(resized_mask.shape[:2])
        # template = cv2.polylines(template, tmp, isClosed=True, color=255)
        # print(np.unique(template), np.unique(resized_mask))
        # intersection = template * resized_mask / 255
        # # print(template.shape, resized_mask.shape)
        # # print(intersection.max(), resized_mask.max())
        # print(
        # area_cc = (intersection == 255).sum() / (resized_mask == 255).sum()
        # avg_area += area_cc
        # print('tmpshape', template.shape)
        # cv2.imwrite('tmp/t1.png', template)

        # for region in coords
        json_path = xml2json(json_dir=json_dir, xml_path=xml_path)
        # print(json_path)

        json_data = json.load(open(json_path, 'r'))

        basename = os.path.basename(json_path).replace('json', 'tif')

        res = {}

        sss = 0
        # count = 0
        for coord  in json_data['coords'][0]:
            # count += 1
            (x, y), level, (patch_size_x, patch_size_y) = coord
                    # print(x, y, level, patch_size_x, patch_size_y)
            patch_id = '{basename}_{x}_{y}_{level}_{patch_size_x}_{patch_size_y}'.format(
                basename=basename,
                x=x,
                y=y,
                level=level,
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y)

            num_patch += 1
            # print(template.max())
            # ratio = template[y:y+patch_size_y, x:x+patch_size_x].sum() / 255 / (patch_size_y * patch_size_x)
            # y = 20992
            # x = 28160
            scaled_x = int(x / downsample_factor)
            scaled_y = int(y / downsample_factor)
            ratio = template[scaled_y:scaled_y+patch_size_y, scaled_x:scaled_x+patch_size_x].sum() / 255 / (patch_size_y * patch_size_x)
            # print(ratio)
            # x = int(x / downsample_factor)
            # y = int(y / downsample_factor)

            #======================================================
            #template_draw = cv2.rectangle(template_draw, (scaled_x , scaled_y ), (scaled_x+patch_size_x, scaled_y+patch_size_y), 255, 10)
            #======================================================

            # print(x, y)
            # print(ratio)
            # if ratio > 0:
                # print(ratio)
            if ratio > 0.0:
                res[patch_id] = 1
                # print(ratio, 'ccc')
                sss += 1
            else:
                 res[patch_id] = 0
            # break


        #======================================================
        #template_draw = cv2.resize(template_draw, (0, 0), fx=0.5, fy=0.5)
        ## # print(template.shape)
        #cv2.imwrite('tmp/real.png', template_draw)
        #mask = cv2.imread(xml2mask(xml_path), -1)
        #cv2.imwrite('tmp/real1.png', mask)
        #======================================================

        # print(sss, count, idx, num_patch)
        print(sss, num_patch, idx)
        # import sys; sys.exit()
        # cccccc = 0
        # for row in range(0, template.shape[0], 512):
        #     for col in range(0, template.shape[1], 512):
        #         cc = template[row:row+512, col:col+512].sum()
        #         if cc != 0:
        #             cccccc += 1
        #             # print(row, col)
        #             print('1111', cccccc, cc / 255, 'coords', row, col)
        # if sss == 0:
        # print(111, sss)
        assert sss > 0

        json_filename = os.path.basename(json_path)
        print(dest_dir)
        json.dump(res, open(os.path.join(dest_dir, json_filename), 'w'))







# draw_mask(wsi_dir=wsi_dir, xml_dir=anno_dir)
write_patch_label(wsi_dir=wsi_dir, xml_dir=anno_dir, json_dir=json_dir, dest_dir=dest_file)




# for xml_path in xml_files(anno_dir=anno_dir):

#     cc = xml2tif(wsi_dir, xml_path)
#     print(xml_path, cc)
