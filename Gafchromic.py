import SimpleITK as sitk 
import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt


class GafchromicFilms:
    
    # Constructor
    #  filename: gafchromic tiff file to read
    #  si nbOfImgs=1, seul le filename est utilisé
    #  autrement filename + firstNb + fileExtension
    def __init__(self, filename, firstNb=0, nbOfImgs=1, fileExtension='.tif', addMethod='median'):
        self.readImg(filename, firstNb, nbOfImgs, fileExtension, addMethod)


    # Reads the image
    #  filename: filename of the gafchromic tiff file to read
    #  si nbOfImgs=1, seul le filename est utilisé
    #  autrement filename + firstNb + fileExtension
    def readImg(self, filename, firstNb=0, nbOfImgs=1, fileExtension='.tif', addMethod='median'):
        if nbOfImgs == 1:
            img = sitk.ReadImage(filename)
            self._sizex = img.GetWidth()
            self._sizey = img.GetHeight()
            self._imgOrigin = img.GetOrigin()
            self._imgSpacing = img.GetSpacing()
            self._array = sitk.GetArrayFromImage(img)
        else:
            if addMethod == 'median':
                img = sitk.ReadImage(filename+str(firstNb)+fileExtension)
                self._sizex = img.GetWidth()
                self._sizey = img.GetHeight()
                self._imgOrigin = img.GetOrigin()
                self._imgSpacing = img.GetSpacing()
                size = (sitk.GetArrayFromImage(img).shape[0], 
                    sitk.GetArrayFromImage(img).shape[1], 
                    sitk.GetArrayFromImage(img).shape[2], 
                    nbOfImgs)
                imgs = np.zeros(size)
                for i in range(nbOfImgs):
                    img = sitk.ReadImage(filename+str(firstNb+i)+fileExtension)
                    imgs[:,:,:,i] = sitk.GetArrayFromImage(img)
                self._array = np.median(imgs, axis=3)
            elif addMethod == 'mean':
                img = sitk.ReadImage(filename+str(firstNb)+fileExtension)
                self._sizex = img.GetWidth()
                self._sizey = img.GetHeight()
                self._imgOrigin = img.GetOrigin()
                self._imgSpacing = img.GetSpacing()
                size = (sitk.GetArrayFromImage(img).shape[0], 
                    sitk.GetArrayFromImage(img).shape[1], 
                    sitk.GetArrayFromImage(img).shape[2], 
                    nbOfImgs)
                imgs = np.zeros(size)
                for i in range(nbOfImgs):
                    img = sitk.ReadImage(filename+str(firstNb+i)+fileExtension)
                    imgs[:,:,:,i] = sitk.GetArrayFromImage(img)
                self._array = np.mean(imgs, axis=3)
            else:
                self._sizex = 0
                self._sizey = 0
                self._imgOrigin = 0
                self._imgSpacing = 0
                self._array = None
                raise ValueError('La valeur de la méthode de sommation des images choisie est inconnue')


    # Methode appelée par print(instance)
    def __str__(self):
        return 'Class GafchromicFilms: \
                \n  * sizex: ' + str(self._sizex) + \
                '\n  * sizey: ' + str(self._sizey) + \
                '\n  * imgOrigin: ' + str(self._imgOrigin) + \
                '\n  * imgSpacing: ' + str(self._imgSpacing)


    # Getter de l'array:
    def getArray(self):
        return self._array

    # Getter de la taille de l'image:
    def getSize(self):
        return (self._sizex, self._sizey)

    # Converts the gafchromic image to dose using the optical density of red over blue channels and a polynomial 
    #  conversion curve (4th degree)
    #  coefs: calibration curve coefficients
    #  dosemax: maximum dose over which the dose is not calculated
    def convertToDose_polynomeLogRB(self, coefs, rbmin, rbmax, dosemax):
        # replaces every 65535 value in array with 65534 to avoid division by zero:
        self._array[self._array==65535]=65534
        
        # converts in optical density
        dor = -np.log10(self._array[:,:,0]/65535.0)
        dob = -np.log10(self._array[:,:,2]/65535.0)
    
        # red channel over blue channel:
        rsb = dor/dob
        rsb[rsb<rbmin] = rbmin
        rsb[rsb>rbmax] = rbmax
        
        # converting in dose:
        doseimg = coefs[0]*rsb**6 + coefs[1]*rsb**5 + coefs[2]*rsb**4 + coefs[3]*rsb**3 + coefs[4]*rsb**2 + coefs[5]*rsb + coefs[6]
        doseimg[doseimg>dosemax] = dosemax
        doseimg[doseimg<0] = 0
        
        return doseimg
    
    
    # Converts the gafchromic image to dose using the red over blue pixel values and a polynomial 
    #  conversion curve (3rd degree)
    #  coefs: calibration curve coefficients
    #  dosemax: maximum dose over which the dose is not calculated
    def convertToDose_polynomeGreyValueRB(self, coefs, rbmin, rbmax):
        # replaces every 65535 value in array with 65534 to avoid division by zero:
        self._array[self._array==65535]=65534
        
        # red channel over blue channel:
        rsb = self._array[:,:,0]/self._array[:,:,2]
        rsb[rsb<rbmin] = rbmin
        rsb[rsb>rbmax] = rbmax
        
        # converting in dose:
        doseimg = coefs[0]*rsb**6 + coefs[1]*rsb**5 + coefs[2]*rsb**4 + coefs[3]*rsb**3 + coefs[4]*rsb**2 + coefs[5]*rsb + coefs[6]
        
        return doseimg

    
    # Converts the gafchromic image to dose using the red over blue pixel values and a spline
    #  conversion curve
    #  coefs: calibration curve coefficients
    #  dosemax: maximum dose over which the dose is not calculated
    def convertToDose_cubicSplineFit(self, splinefile, dosemax):
        # replaces every 65535 value in array with 65534 to avoid division by zero:
        self._array[self._array==65535]=65534
 
        # reads the spline file:
        with open(splinefile) as f:
            i = 0
            dose = []
            rbvalues = []
            for line in f:
                if (line[0]).isdigit():
                    s = line.split("\t")
                    rbvalues.append(float(s[0]))
                    dose.append(float(s[1]))
        
        cs = CubicSpline(rbvalues, dose)

        # red channel over blue channel:
        rsb = self._array[:,:,0]/self._array[:,:,2]
        
        # converting in dose:
        doseimg = cs(rsb)
        doseimg[doseimg>dosemax] = dosemax
        doseimg[doseimg<0] = 0
        
        return doseimg
    
    
    # Saves the dose image to a tiff file that can be read using Verisoft
    # doseimg: img to save
    # filename: filename the dose img will be written to
    def saveToTiff(self, doseimg, filename):
        imagetif = sitk.Image([doseimg.shape[1],doseimg.shape[0]], sitk.sitkVectorUInt16, 3)
        imagetif.SetSpacing(self.img_spacing)
        imagetif.SetOrigin(self.img_origin)
        for j in range(0, doseimg.shape[0]):
            for i in range(0, doseimg.shape[1]):
                a = int(doseimg[j,i])
                imagetif.SetPixel(i,j,[a, a, a])
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filename)
        writer.Execute(imagetif)
        return True




# 
if __name__ == '__main__':
    filename = 'G:/Commun/PHYSICIENS/Erwann/EBT3/09 - etalonnage lot 10151801 19juil2019/films a 24h/scan'
    nbofimgs = 5
    firstimg = 1
    m_splineFile = 'G:/Commun/PHYSICIENS/Erwann/EBT3/09 - etalonnage lot 10151801 19juil2019/films a 24h/bSpline_data.txt'
    m_dosemax = 1000.0 #cGy
    
    try:
        g = GafchromicFilms(filename, firstimg, nbofimgs)
        doseimg = g.convertToDose_cubicSplineFit(m_splineFile, m_dosemax)
    except ValueError as err:
        print('Erreur: '+err)

    print(g)
    plt.figure(1, figsize=(10,10))
    plt.imshow(g.getArray()[:,:,0])
    plt.imshow(doseimg)
    plt.show()

