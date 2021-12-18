import spectral as sp
import numpy as np
from PIL import Image
import random


def read_and_calibrate():
    header_paths = ['data\Tray1_SWIR_384_SN3154_6056us_2021-04-08T154417_raw_rad_ref_float32.hdr', 'data\Tray2_SWIR_384_SN3154_6056us_2021-04-08T155241_raw_rad_ref_float32.hdr']
    image_paths = ['data\Tray1_SWIR_384_SN3154_6056us_2021-04-08T154417_raw_rad_ref_float32.img', 'data\Tray2_SWIR_384_SN3154_6056us_2021-04-08T155241_raw_rad_ref_float32.img']
    roi_mask_paths = ['data\\Tray1_mask.png', 'data\\Tray2_mask.png']

    mask_value_to_consentration = {
        1: 0.0,
        5: 0.5,
        10: 1.0,
        20: 2.0,
        30: 3.0,
        40: 4.0,
        50: 5.0,
        60: 6.0,
        70: 7.0,
        80: 8.0,
        90: 9.0,
        100: 10.0,
        150: 15.0,
        200: 20.0,
        250: 25.0,
        255: 100.0
    }

    rois = [None]*len(mask_value_to_consentration)

    for i in range(len(header_paths)):
        img = sp.envi.open(header_paths[i], image_paths[i])

        white_panel = img[1500:,:,:]
        white_panel_spectrum = np.average(white_panel, axis=(0,1))

        roi_mask_rgba = np.array(Image.open(roi_mask_paths[i]))
        roi_mask = roi_mask_rgba[:,:,0]

        for j, mask_value in enumerate(mask_value_to_consentration.keys()):
            corners = np.where(roi_mask == mask_value)
            if len(corners[0]):
                rois[j] = 0.5*img[corners[0][0]:corners[0][1], corners[1][0]:corners[1][1], :]/white_panel_spectrum

    return 1e-9*np.array(img.bands.centers), rois, np.array(list(mask_value_to_consentration.values()))


def average_over_regions(R_data, region_size):
    frame_n, frame_m = region_size
    samples = [None]*len(R_data)
    for i, roi in enumerate(R_data):
        frame_n = frame_n if frame_n else roi.shape[0]
        frame_m = frame_m if frame_m else roi.shape[1]
        n = int(roi.shape[0]/frame_n)
        m = int(roi.shape[1]/frame_m)
        samples[i] = []
        for x in range(n):
            for y in range(m):
                samples[i].append(np.average(roi[frame_n*x:frame_n*(x+1),frame_m*y:frame_m*(y+1)], axis=(0,1)))
        samples[i] = np.array(samples[i])
    return samples


def random_equal_split(samples):
    train_set = []
    test_set = []

    for c, spectra_c in enumerate(samples):
            half = int(len(spectra_c)/2)
            random.shuffle(spectra_c)
            train_set.append([])
            test_set.append([])
            for spectrum in spectra_c[0:half]:
                train_set[c].append(spectrum)
            for spectrum in spectra_c[half:]:
                test_set[c].append(spectrum)
    return train_set, test_set