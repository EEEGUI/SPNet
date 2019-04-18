import scipy
import skimage
import numpy as np
from pypardiso import spsolve
from PIL import Image
from tqdm import tqdm
import pandas as pd


def decode_depthmap(disparity):
    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256

    disparity[disparity > 0] = (0.209313 * 2262.52) / disparity[disparity > 0]
    return disparity

def img2depth(filename):

    depth = decode_depthmap(np.array(Image.open(filename), dtype=np.float32))

    return depth


def fill_depth_colorization(rgb_filename, depth_filename, alpha=1):
    imgRgb = np.array(Image.open(rgb_filename), dtype=int)
    imgRgb = imgRgb / np.max(imgRgb)
    imgDepthInput = img2depth(depth_filename)

    imgIsNoise = imgDepthInput == 0.0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window
    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output

def depth2img(depth, filename):
    depth = depth/depth.max() * 255
    depth = (depth * 255).astype('int16')
    depth_png = Image.fromarray(depth)
    depth_png.save(filename)

def city_depth2img(filename):
    depth2img(img2depth(filename), filename.split('/')[-1])



if __name__ == '__main__':
    rgb = '/home/lin/Documents/dataset/Cityscapes/leftImg8bit/train/aachen/aachen_000015_000019_leftImg8bit.png'
    depth_img = '/home/lin/Documents/dataset/Cityscapes/disparity/train/aachen/aachen_000015_000019_disparity.png'
    depth = fill_depth_colorization(rgb, depth_img)
    depth2img(depth, 'assets/%s' % (depth_img.split('/')[-1]))
    # city_depth2img(depth_img)