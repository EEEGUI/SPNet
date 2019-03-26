import numpy as np
from PIL import Image



if __name__ == '__main__':
    depth = Image.open('/home/lin/Documents/dataset/Cityscapes/disparity/train/aachen/aachen_000000_000019_disparity.png')


    def decode_depthmap(disparity):
        disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256

        disparity[disparity > 0] = (0.209313 * 2262.52) / disparity[disparity > 0]

        return disparity

    depth = decode_depthmap(np.array(depth, dtype=np.float32))
    print(depth[270, 106])