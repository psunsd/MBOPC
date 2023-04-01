from builtins import object
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import matplotlib.pyplot as pl
from matplotlib.path import Path
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from scipy.signal import convolve,convolve2d
from scipy.special import expit,jv
from scipy.optimize import minimize
from numpy.linalg import norm
from math import pi,sin,cos,sqrt,atan
from skimage.draw import polygon
import time
import gdspy
import cv2

__author__ = "Peng Sun"
__license__ ="BSD-3"

OTRENCHS=[15,1]

class MetaOPC(with_metaclass(ABCMeta, object)):
    def __init__(self, device=None):
        self._device = device

class MBOPC(MetaOPC):
    def __init__(self):
        super(MBOPC, self).__init__(device='psgc')

    @abstractmethod
    def Create_Prescribed(self):
        pass

    # @abstractmethod
    # def F1(self, theta):
    #     pass
    #
    # @abstractmethod
    # def F1jac(self, theta):
    #     pass

    def Forward_Model(self, m):
        maerial = convolve(m, self.Hk, method='fft')**(self.Coherent+1)
        maerial = maerial/np.max(maerial)
        maerial = maerial[int(self.Nk/2):self.mask_size[0]+int(self.Nk/2), int(self.Nk/2):self.mask_size[1]+int(self.Nk/2)]
        z = expit(self.k*(maerial-self.tr))
        return z

    def F1(self, theta):
        theta = theta.reshape(self.mask_size)
        self.m = (1 + np.cos(theta)) / 2.0
        if self.Coherent is True:
            self.maerial = convolve(self.m, self.Hk, method='fft')**2
        else:
            self.maerial = convolve(self.m, self.Hk, method='fft')

        self.maerial = self.maerial / np.max(self.maerial)
        self.maerial = self.maerial[int(self.Nk/2):self.mask_size[0]+int(self.Nk/2), int(self.Nk/2):self.mask_size[1]+int(self.Nk/2)]
        self.z = expit(self.k*(self.maerial-self.tr))
        res = norm(self.zp - self.z) ** 2
        if self.Use_Quad_Penalty is True:
            res += self.quad_coeff * np.sum(4.0 * self.m * (1.0 - self.m))
        if self.Use_Complex_Penalty is True:
            ### H/V edges of (m-zp)
            f = np.abs(self.m - self.zp)
            complex_penalty = np.sum(np.abs(np.matmul(self.Qx, f))) + np.sum(np.abs(np.matmul(f, self.Qy)))

            ### edge of m
            # medge = convolve(m,H)
            # complex_penalty = np.sum(np.abs(medge))

            ### edge of sigmoid(k*(|m|-5))
            # medge = convolve(m,H)
            # complex_penalty = np.sum(expit(40*(np.abs(medge)-5)))

            res += self.complex_coeff * complex_penalty
        return res

    def F1jac(self, theta):
        theta=theta.reshape(self.mask_size)
        if self.Coherent is True:
            mjac = convolve((self.zp-self.z)*self.z*(1-self.z), self.Hk, method='fft')
        else:
            mjac = convolve((self.zp-self.z)*self.z*(1-self.z)*self.maerial, self.Hk, method='fft')

        mjac = mjac / np.max(mjac)
        mjac = mjac[int(self.Nk/2):self.mask_size[0]+int(self.Nk/2), int(self.Nk/2):self.mask_size[1]+int(self.Nk/2)]
        jac = self.k * np.sin(theta) * mjac

        if self.Use_Quad_Penalty is True:
            jac_quad_penalty = (-8.0 * self.m + 4.0)
            jac += self.quad_coeff * jac_quad_penalty

        if self.Use_Complex_Penalty is True:
            ### h/v edges of (m-zp)
            f = np.abs(self.m - self.zp)
            Qxf = np.matmul(self.Qx,f)
            Qyf = np.matmul(f,self.Qy)
            QxTQxf=np.matmul(np.transpose(self.Qx), np.sign(Qxf+1e-15))
            QyTQyf=np.matmul(np.sign(Qyf+1e-15), np.transpose(self.Qy))
            jac_complex_penalty = (QxTQxf + QyTQyf+1e-15)*np.sign(self.m-self.zp)

            ### edges of abs(m-zp) by differential kernel
            # medge = convolve(f,H)
            # jac_complex_penalty = convolve(np.sign(medge),np.fliplr(np.flipud(H)))*np.sign(m-zp+1e-15)
            ### edge of m
            # medge = convolve(m,H)
            # jac_complex_penalty = convolve(np.sign(medge),np.fliplr(np.flipud(H)))

            ### edge of sigmoid(k*(|m|-5))
            # medge = convolve(m,H)
            # jac_complex_penalty = convolve(np.sign(medge), np.fliplr(np.flipud(H))) * 40*expit(40*(np.abs(medge)-5))*(1-expit(40*(np.abs(medge)-5)))

            jac += self.complex_coeff * jac_complex_penalty

        return jac.flatten()

    def Jinc_kernel(self, Nk, kNa):
        hx, hy = np.meshgrid(np.linspace(0,Nk-1,Nk)-(Nk-1)/2, np.linspace(0,Nk-1,Nk)-(Nk-1)/2)
        self.hr = np.sqrt(hx**2+hy**2)
        self.hr[int(Nk/2), int(Nk/2)] = 1.0
        Hk = jv(1, self.hr*kNa)/(self.hr*kNa)
        Hk[int(Nk/2), int(Nk/2)] = 0.5
        return Hk

    def Jinc_kernel_derivative(self, Nk, kNa):
        # hx, hy = np.meshgrid(np.linspace(0,Nk-1,Nk)-(Nk-1)/2, np.linspace(0,Nk-1,Nk)-(Nk-1)/2)
        # hr = np.sqrt(hx**2+hy**2)
        # hr[int(Nk/2), int(Nk/2)] = 1.0
        Hk = jv(1, self.hr*kNa)/(self.hr*kNa)
        Hk[int(Nk/2), int(Nk/2)] = 0.5
        Hkp = jv(0, self.hr*kNa) - Hk/kNa
        return Hkp

    def Gaussian_kernel(self, Nk, sigma):
        hx, hy = np.meshgrid(np.linspace(0,Nk-1,Nk)-(Nk-1)/2, np.linspace(0,Nk-1,Nk)-(Nk-1)/2)
        self.hr = np.sqrt(hx**2+hy**2)
        Hk = np.exp(-self.hr**2/2/sigma**2)
        return Hk

    def Import_Edges(self, filename, mask_size=(10,10), si_idx=9, mask_idx=14):
        edge = np.genfromtxt(filename, delimiter=',', skip_header=1)
        siedge = edge[edge[:,0]==si_idx]
        maskedge = edge[edge[:,0]==mask_idx]
        maskctr = np.average(maskedge[:,1:3], axis=0)
        maskangle = np.arctan2(maskedge[:,2]-maskctr[1], maskedge[:,1]-maskctr[0])
        maskedgesort = maskedge[maskangle.argsort()]

        self.sipattern, self.maskpattern = np.zeros(mask_size), np.zeros(mask_size)
        self.sipattern_size = mask_size
        rr, cc = polygon(siedge[:,1]+50, siedge[:,2]+60)
        self.sipattern[rr,cc] = 1
        rr, cc = polygon(maskedgesort[:,1]+50, maskedgesort[:,2]+60)
        self.maskpattern[rr,cc] = 1

    def Update_Hk(self):
        if self.Coherent is True:
            self.Hk = self.Jinc_kernel(self.Nk, self.kNa)
        else:
            self.Hk = self.Gaussian_kernel(self.Nk, self.sigma)

    def Eval_Fwd_Err(self, p=np.array([25.0, 0.26])):
        """
        Evaluate forward model
        Parameters
        ------------
        p[0]: sigma in pixel for incoherent imaging system, and kNa for coherent
        p[1]: resist threshold
        """
        if self.Coherent is True:
            self.Hk = self.Jinc_kernel(self.Nk, p[0])
        else:
            self.Hk = self.Gaussian_kernel(self.Nk, p[0])
        self.maerial = convolve(self.maskpattern, self.Hk, method='fft')**(self.Coherent+1)
        self.maerial = self.maerial/np.max(self.maerial)
        self.maerial = self.maerial[int(self.Nk/2):self.sipattern_size[0]+int(self.Nk/2), int(self.Nk/2):self.sipattern_size[1]+int(self.Nk/2)]
        self.z = expit(self.k*(self.maerial-p[1])).reshape(self.sipattern_size)
        err = norm(self.z-self.sipattern)**2
        return err

    def Import_GDS(self, filename, mask_size=(10,10)):
        img = np.zeros(mask_size).flatten()
        lib = gdspy.GdsLibrary(infile=filename)
        topcell = lib.top_level()[0]
        poly = topcell.get_polygons(by_spec=True)
        OTRENCHS_polys = poly[OTRENCHS]

        x, y = np.meshgrid(np.arange(mask_size[0]), np.arange(mask_size[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        for p in OTRENCHS_polys:
            edgex, edgey = p[:, 0] * 1000 - 24400, p[:, 1] * 1000 - 24400
            if True and edgex.max() >= 0 and edgex.min() < mask_size[0]:
                if edgey.max() >= 0 and edgey.min() < mask_size[1]:
                    if np.any(np.logical_and.reduce((edgex >= 0, edgex < mask_size[0], edgey >= 0, edgey < mask_size[1]))):
                        ### skimage.draw.polygon() is extremely slow
                        # rr,cc=polygon(edgex,edgey)
                        # if np.any(rr<mask_size[0]) or np.any(cc<mask_size[1]) or np.any(rr>=0) or np.any(cc>=0):
                        #     mask=np.logical_and(rr<mask_size[0], cc<mask_size[1])
                        #     img[rr[mask],cc[mask]]=1

                        ### matplotlib.path.Path is faster
                        pt = np.array([edgex, edgey]).T
                        path = Path(pt)
                        img = np.logical_or(img, path.contains_points(points))
        # pl.imshow(img.reshape(mask_size)); pl.show()
        return img.reshape(mask_size)

    def Export_GDS(self, img, th=63, Ngauss=5):
        scale = 60
        lib = gdspy.GdsLibrary()
        PSGCenclosure = lib.new_cell('ENCLOSURE', overwrite_duplicate=True)
        PSGCscatterall = lib.new_cell('SCATTERS')
        PSGC = lib.new_cell('PSGC')

        encx = [0, 20, 20, 0]
        ency = [0, 0, 20, 20]
        enc_vert = [(encx[ii], ency[ii]) for ii in range(len(encx))]
        enc = gdspy.Polygon(enc_vert, layer=OTRENCHS[0], datatype=OTRENCHS[1])
        PSGCenclosure.add(enc)

        mopc = img
        mopcgray = np.round(mopc*255).astype('uint8')
        mopcblur = cv2.GaussianBlur(mopcgray, (Ngauss, Ngauss), 0)
        ret, mopcth = cv2.threshold(mopcblur, th, 255, cv2.THRESH_BINARY)
        cnt, _ = cv2.findContours(mopcth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt:
            if len(c) >= 3:
                Area = 1e3
                if len(c) <= 10:
                    x, y = c.squeeze()[:,0]/scale, c.squeeze()[:,1]/scale
                    Area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                # pl.plot(c.squeeze()[:,0],c.squeeze()[:,1])
                if Area > 1e-3:
                    scatter = gdspy.Polygon(np.flipud(c.squeeze())/scale, layer=OTRENCHS[0], datatype=OTRENCHS[1])
                    PSGCscatterall.add(scatter)

        PSGCsubtracted = gdspy.boolean(PSGCenclosure, PSGCscatterall, "xor", layer=OTRENCHS[0], datatype=OTRENCHS[1])

        PSGC.add(PSGCsubtracted)
        lib.remove(PSGCenclosure)
        lib.remove(PSGCscatterall)
        lib.write_gds('PSGCOPC_Ng_' + str(Ngauss) + '_th_' + str(th) + '.gds')
        gdspy.current_library.remove('ENCLOSURE')
        gdspy.current_library.remove('SCATTERS')
        gdspy.current_library.remove('PSGC')

    def Import_CSV(self, filename, mask_size=(10,10)):
        self.mask_size = mask_size
        maskraw = np.genfromtxt(filename, delimiter=',', skip_header=1)
        mopc = maskraw[:,0].reshape(mask_size)
        if self.Use_Complex_Penalty is True:
            self.Qx = np.identity(self.mask_size[0]) - shift(np.identity(self.mask_size[0]), (1, 0))
            self.Qy = np.identity(self.mask_size[1]) - shift(np.identity(self.mask_size[1]), (0, 1))
        return mopc

    def Export_CSV(self, img, filename, mask_size=(10,10)):
        nidx=np.linspace(0,mask_size[0]*mask_size[1]-1,mask_size[0]*mask_size[1])
        nx = np.mod(nidx, mask_size[0])
        ny = np.floor(nidx/mask_size[0])
        np.savetxt(filename, np.transpose([img, nx, ny]), header='mask,x,y', comments='', delimiter=',')
