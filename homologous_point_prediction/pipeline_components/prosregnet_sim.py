import numpy as np
import cv2


def prn_extrude_hist(hist, hist_points):
    points = np.argwhere(hist[:, :, 0] != 0)
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    y, x, h, w = cv2.boundingRect(points)  # create a rectangle around those points
    crop = hist[x:x + w, y:y + h, :]

    if h > w:
        y_offset = int(h * 0.15)
        x_offset = int((h - w + 2 * y_offset) / 2)
    else:
        y_offset = int(h * 0.2)
        x_offset = int((h - w + 2 * y_offset) / 2)

    # pad image
    h = h + 2 * y_offset
    w = w + 2 * x_offset

    padHist = np.zeros((w, h, 3))

    padHist[x_offset:crop.shape[0] + x_offset, y_offset:crop.shape[1] + y_offset, :] = crop

    # Round pixels to ints
    padHist = (padHist * 255).astype(np.uint8)
    hist_points = np.copy(hist_points)
    hist_points[:, 0] = hist_points[:, 0] - x + x_offset
    hist_points[:, 1] = hist_points[:, 1] - y + y_offset
    return padHist, hist_points

def prn_extrude_mri(mri, mri_points):
    points = points = np.argwhere(mri[:, :, 0] != 0)
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    y, x, h, w = cv2.boundingRect(points)  # create a rectangle around those points

    mri = mri * 255

    if h > w:
        y_offset = int(h*0.15)
        x_offset = int((h - w + 2*y_offset)/2)
    else:
        y_offset = int(h*0.2)
        x_offset = int((h - w + 2*y_offset)/2)

    crop = mri[x - x_offset:x+w+x_offset, y - y_offset:y+h +y_offset]

    crop = crop*25.5/(np.max(crop)/10)
    mri_points = np.copy(mri_points)
    mri_points[:, 1] = mri_points[:, 1] - y + y_offset
    mri_points[:, 0] = mri_points[:, 0] - x + x_offset
    return(crop.astype(int).astype(float), mri_points)