# -*- coding: utf-8 -*-

# AUTHOR DEBORAH BAMBIL 

import numpy as np
from skimage import feature as ftr, measure as msr
from skimage.filters import prewitt_h as p_h, prewitt_v as p_v
import cv2
import os
import tempfile

_N = 'numeric'
_C_MIN, _C_MAX = 100, 200
_G_L, _L_R, _B_S = 256, 2, 18
_SI = False

class Extratores(object):
    def __init__(self):
        self.i, self.i_g, self.i_b, self.i_e, self.i_f = [None]*5
        self.s_q, self.t_d = 1, tempfile.gettempdir()

    def _f1(self):
        h, l = cv2.cvtColor(self.i, 40), cv2.cvtColor(self.i, 44)
        c_s = cv2.split(self.i) + cv2.split(h) + cv2.split(l)
        n = [f'c_{j}_{k}' for j in range(9) for k in ['a','b','c','d']]
        v = []
        for c in c_s:
            v.extend([float(np.min(c)), float(np.max(c)), float(np.mean(c)), float(np.std(c))])
        return n, [_N]*len(n), v

    def _f2(self):
        m = msr.moments(self.i_g)
        if m[0,0] == 0: return [f'h_{j}' for j in range(7)], [_N]*7, [0.0]*7
        r, c = int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])
        r, c = max(0, min(r, self.i_g.shape[0]-1)), max(0, min(c, self.i_g.shape[1]-1))
        h = msr.moments_hu(msr.moments_normalized(msr.moments_central(self.i_g, (r, c))))
        return [f'h_{j}' for j in range(7)], [_N]*7, [float(x) for x in h]

    def _f3(self):
        g = ftr.graycomatrix(self.i_g, [1, 2], [0, np.pi/4, np.pi/2], _G_L, True, True)
        p = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
        v = []
        for x in p: v.extend(ftr.graycoprops(g, x).flatten().tolist())
        n = [f'g_{p[i//6]}_{i%6}' for i in range(36)]
        return n, [_N]*len(n), [float(x) for x in v]

    def _f4(self):
        v = ftr.hog(self.i_f, orientations=8, pixels_per_cell=(32,32), cells_per_block=(1,1), block_norm='L1')
        return [f'og_{j}' for j in range(len(v))], [_N]*len(v), [float(x) for x in v]

    def _f5(self):
        l = ftr.local_binary_pattern(self.i_g, 8*_L_R, _L_R, 'uniform')
        h, _ = np.histogram(l, density=True, bins=_B_S, range=(0, _B_S))
        return [f'lb_{j}' for j in range(_B_S)], [_N]*_B_S, [float(x) for x in h]

    def _f6(self):
        k = cv2.getGaborKernel((10,10), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        f = cv2.filter2D(self.i_g, cv2.CV_8UC3, k)
        return ['gb_1', 'gb_2'], [_N]*2, [float(np.mean(f)), float(np.std(f))]

    def _f7(self):
        f = np.fft.fftshift(np.fft.fft2(self.i_g))
        m = 20 * np.log(np.abs(f) + 1e-9)
        return [f'ft_{j}' for j in range(4)], [_N]*4, [float(np.mean(m)), float(np.std(m)), float(np.max(m)), float(np.min(m))]

    def _f8(self):
        h, _ = np.histogram(self.i_e, bins=2, range=(0, 256))
        return [f'cn_{j}' for j in range(2)], [_N]*2, [float(x) for x in h]

    def _f9(self):
        x, y = cv2.Scharr(self.i_g, cv2.CV_64F, 1, 0), cv2.Scharr(self.i_g, cv2.CV_64F, 0, 1)
        m = cv2.normalize(np.sqrt(x**2 + y**2), None, 0, 1, cv2.NORM_MINMAX)
        h, _ = np.histogram(m, density=True, bins=10, range=(0,1))
        return [f'sc_{j}' for j in range(10)], [_N]*10, [float(z) for z in h]

    def _f10(self):
        m = cv2.normalize(np.absolute(cv2.Laplacian(self.i_g, cv2.CV_64F)), None, 0, 1, cv2.NORM_MINMAX)
        h, _ = np.histogram(m, bins=10, range=(0,1), density=True)
        return [f'lp_{j}' for j in range(10)], [_N]*10, [float(z) for z in h]

    def _f11(self):
        x, y = cv2.Sobel(self.i_g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(self.i_g, cv2.CV_64F, 0, 1, ksize=3)
        m = cv2.normalize(np.sqrt(x**2 + y**2), None, 0, 1, cv2.NORM_MINMAX)
        h, _ = np.histogram(m, bins=10, range=(0,1), density=True)
        return [f'sb_{j}' for j in range(10)], [_N]*10, [float(z) for z in h]

    def _f12(self):
        m = cv2.normalize(np.sqrt(p_h(self.i_g)**2 + p_v(self.i_g)**2), None, 0, 1, cv2.NORM_MINMAX)
        h, _ = np.histogram(m, density=True, bins=10, range=(0,1))
        return [f'pw_{j}' for j in range(10)], [_N]*10, [float(z) for z in h]

    def extrai_todos(self, img):
        if img is None or img.size == 0: return None, None, None
        self.i = img
        self.i_g = cv2.cvtColor(self.i, 6)
        self.i_e = cv2.Canny(self.i_g, _C_MIN, _C_MAX)
        # FIX: Removed 16 (Triangle) to avoid Assertion Failed error. Using only 8 (Otsu).
        _, self.i_b = cv2.threshold(self.i_g, 0, 255, 8)
        self.i_f = cv2.resize(self.i_g, (128, 128))
        
        l_e = [self._f1, self._f2, self._f3, self._f4, self._f5, self._f6, 
               self._f7, self._f8, self._f9, self._f10, self._f11, self._f12]
        t_n, t_t, t_v = [], [], []
        for f in l_e:
            n, t, v = f()
            t_n.extend(n); t_t.extend(t); t_v.extend(v)
        return t_n, t_t, t_v