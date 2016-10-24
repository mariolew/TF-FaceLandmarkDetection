import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

TXT = 'testImageList.txt'

def read_data_from_txt(TXT):
    with open(TXT, 'r') as fid:
        lines = fid.readlines()
    result = []
    for line in lines:
        components = line.strip().split(' ')
        imgName = components[0].replace('\\', '/')
        bbx = map(int, components[1:5])
        landmarks = map(float, components[5:])
        landmarks = np.asarray(landmarks).reshape([-1, 2])
        result.append([imgName, BBox(bbx), landmarks])
    return result

def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    # Make sure that the flipped landmarks are in the right order #
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)

def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = (bbox.x+bbox.w/2, bbox.y+bbox.h/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.y:bbox.y+bbox.h,bbox.x:bbox.x+bbox.w]
    return (face, landmark_)

def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m) / s
    return imgs

def generate_hdf5(data, output='shit.h5'):
    lines = []
    dst = 'tf_test/'
    imgs = []
    labels = []
    for (imgPath, bbx, landmarks) in data:
        im = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        imgName = imgPath.split('/')[-1][:-4]
        
        bbx_sc = bbx.bbxScale(im.shape, scale=1.1)
        #print bbx_sc.x, bbx_sc.y, bbx_sc.w, bbx_sc.h
        im_sc = im[bbx_sc.y:bbx_sc.y+bbx_sc.h, bbx_sc.x:bbx_sc.x+bbx_sc.w]
        im_sc = cv2.resize(im_sc, (39, 39))
        imgs.append(im_sc.reshape(39, 39, 1))
        name = dst+imgName+'sc.jpg'
        lm_sc = bbx_sc.normalizeLmToBbx(landmarks)
        labels.append(lm_sc.reshape(10))
        lines.append(name + ' ' + ' '.join(map(str, lm_sc.flatten())) + '\n')
    imgs, labels = np.asarray(imgs), np.asarray(labels)
    imgs = processImage(imgs)
    with h5py.File('shit.h5', 'w') as h5:
        h5['data'] = imgs.astype(np.float32)
        h5['landmark'] = labels.astype(np.float32)

def data_augmentation(data, output='tfboy.txt', is_training=False):
    lines = []
    dst = 'tfvae_test/'
    for (imgPath, bbx, landmarks) in data:
        im = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        imgName = imgPath.split('/')[-1][:-4]
        
        bbx_sc = bbx.bbxScale(im.shape, scale=1.1)
        #print bbx_sc.x, bbx_sc.y, bbx_sc.w, bbx_sc.h
        im_sc = im[bbx_sc.y:bbx_sc.y+bbx_sc.h, bbx_sc.x:bbx_sc.x+bbx_sc.w]
        im_sc = cv2.resize(im_sc, (64, 64))
        name = dst+imgName+'sc.png'
        cv2.imwrite(name, im_sc)
        lm_sc = bbx_sc.normalizeLmToBbx(landmarks)
        lines.append(name + ' ' + ' '.join(map(str, lm_sc.flatten())) + '\n')

        if not is_training:
            continue

        

        origin = im[bbx.y:bbx.y+bbx.h, bbx.x:bbx.x+bbx.w]
        origin = cv2.resize(origin, (64, 64))
        name = dst+imgName+'origin.png'
        cv2.imwrite(name, origin)
        lm_o = bbx.normalizeLmToBbx(landmarks)
        lines.append(name + ' ' + ' '.join(map(str, lm_o.flatten())) + '\n')

        bbx_sf = bbx_sc.bbxShift(im.shape)
        im_sf = im[bbx_sf.y:bbx_sf.y+bbx_sf.h, bbx_sf.x:bbx_sf.x+bbx_sf.w]
        im_sf = cv2.resize(im_sf, (64, 64))
        name = dst+imgName+'sf.png'
        cv2.imwrite(name, im_sf)
        lm_sf = bbx_sf.normalizeLmToBbx(landmarks)
        lines.append(name + ' ' + ' '.join(map(str, lm_sf.flatten())) + '\n')

        im_rotate, lm_rotate = rotate(im, bbx_sc, landmarks, 5)
        im_rotate = cv2.resize(im_rotate, (64, 64))
        name = dst+imgName+'rotate.png'
        cv2.imwrite(name, im_rotate)
        lm_rotate = bbx_sc.normalizeLmToBbx(lm_rotate)
        lines.append(name + ' ' + ' '.join(map(str, lm_rotate.flatten())) + '\n')
        # bbx_sf2 = bbx_sc.bbxShift(im.shape)
        # im_sf2 = im[bbx_sf2.y:bbx_sf2.y+bbx_sf2.h, bbx_sf2.x:bbx_sf2.x+bbx_sf2.w]
        # im_sf2 = cv2.resize(im_sf2, (39, 39))
        # name = dst+imgName+'sf2.png'
        # cv2.imwrite(name, im_sf2)
        # lm_sf2 = bbx_sf2.normalizeLmToBbx(landmarks)
        # lines.append(name + ' ' + ' '.join(map(str, lm_sf2.flatten())) + '\n')

        flipo, lm_flipo = flip(origin, lm_o)
        name = dst+imgName+'flipo.png'
        cv2.imwrite(name, flipo)
        lines.append(name + ' ' + ' '.join(map(str, lm_flipo.flatten())) + '\n')

        flipsc, lm_flipsc = flip(im_sc, lm_sc)
        name = dst+imgName+'flipsc.png'
        cv2.imwrite(name, flipsc)
        lines.append(name + ' ' + ' '.join(map(str, lm_flipsc.flatten())) + '\n')

        flipsf, lm_flipsf = flip(im_sf, lm_sf)
        name = dst+imgName+'flipsf.png'
        cv2.imwrite(name, flipsf)
        lines.append(name + ' ' + ' '.join(map(str, lm_flipsf.flatten())) + '\n')

        # flipsf2, lm_flipsf2 = flip(im_sf2, lm_sf2)
        # name = dst+imgName+'flipsf2.png'
        # cv2.imwrite(name, flipsf2)
        # lines.append(name + ' ' + ' '.join(map(str, lm_flipsf2.flatten())) + '\n')

    with open(output, 'w') as fid:
        fid.writelines(lines)






class BBox(object):

    def __init__(self, bbx):
        self.x = bbx[0]
        self.y = bbx[2]
        self.w = bbx[1] - bbx[0]
        self.h = bbx[3] - bbx[2]


    def bbxScale(self, im_size, scale=1.3):
        # We need scale greater than 1 #
        assert(scale > 1)
        x = np.around(max(1, self.x - (scale * self.w - self.w) / 2.0))
        y = np.around(max(1, self.y - (scale * self.h - self.h) / 2.0))
        w = np.around(min(scale * self.w, im_size[1] - x))
        h = np.around(min(scale * self.h, im_size[0] - y))
        return BBox([x, x+w, y, y+h])

    def bbxShift(self, im_size, shift=0.03):
        direction = np.random.randn(2)
        x = np.around(max(1, self.x - self.w * shift * direction[0]))
        y = np.around(max(1, self.y - self.h * shift * direction[1]))
        w = min(self.w, im_size[1] - x)
        h = min(self.h, im_size[0] - y)
        return BBox([x, x+w, y, y+h])

    def normalizeLmToBbx(self, landmarks):
        result = []
        # print self.x, self.y, self.w, self.h
        # print landmarks
        lmks = landmarks.copy()
        for lm in lmks:
            lm[0] = (lm[0] - self.x) / self.w
            lm[1] = (lm[1] - self.y) / self.h
            result.append(lm)
        result = np.asarray(result)
        
        return result


if __name__ == '__main__':
    data = read_data_from_txt(TXT)
    # generate_hdf5(data)
    data_augmentation(data, output='tftest_vae.txt', is_training=False)

