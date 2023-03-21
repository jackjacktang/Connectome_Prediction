import math
import nibabel as nib
import numpy as np

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # print(err)
    err /= len(imageA)

    return err


def psnr2(img1, img2, valid):
    mae = np.sum(img1 - img2) / valid
    mse = (np.sum((img1 - img2) ** 2)) / valid
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    return mae, mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def pad(b_slice, mask_slice, dim1, dim2):
    pad_width = round((256 - dim2) / 2)
    pad_width1 = round((256 - dim1) / 2)

    if dim2 > 256:
        b_temp = b_slice[:, pad_width * -1:256 - pad_width]
        m_temp = mask_slice[:, pad_width * -1:256 - pad_width]
    else:
        b_temp = np.pad(b_slice, ((0, 0), (pad_width, 256 - pad_width - dim2)), 'constant',
                        constant_values=((0, 0), (0, 0)))
        m_temp = np.pad(mask_slice, ((0, 0), (pad_width, 256 - pad_width - dim2)), 'constant',
                        constant_values=((0, 0), (0, 0)))

    if dim1 > 256:
        b_temp = b_temp[pad_width1 * -1:256 - pad_width1, :]
        m_temp = m_temp[pad_width1 * -1:256 - pad_width1, :]
    else:
        b_temp = np.pad(b_temp, ((pad_width1, 256 - pad_width1 - dim1), (0, 0)), 'constant',
                        constant_values=((0, 0), (0, 0)))
        m_temp = np.pad(m_temp, ((pad_width1, 256 - pad_width1 - dim1), (0, 0)), 'constant',
                        constant_values=((0, 0), (0, 0)))

    return b_temp, m_temp


def unpad(comp, dim1, dim2):
    pad_width = round((256 - dim2) / 2)
    pad_width1 = round((256 - dim1) / 2)

    if dim2 > 256:
        temp = np.pad(comp[:, :], ((0, 0), (pad_width * -1, pad_width - 256 + dim2)), 'constant',
                      constant_values=((0, 0), (0, 0)))
    else:
        temp = comp[:, pad_width:pad_width + dim2]

    if dim1 > 256:
        temp = np.pad(temp, ((pad_width1 * -1, pad_width1 - 256 + dim1), (0, 0)), 'constant',
                              constant_values=((0, 0), (0, 0)))
    else:
        temp = temp[pad_width1:pad_width1 + dim1, :]

    return temp
