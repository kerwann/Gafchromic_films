import SimpleITK as sitk 
import numpy as np
from scipy.interpolate import CubicSpline
from skimage import filters

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
    #  if nbOfImgs=1, only the filename is used
    #  otherwise filename + firstNb + fileExtension
    def readImg(self, filename, firstNb=0, nbOfImgs=1, fileExtension='.tif', addMethod='median'):
        self._scannerCorr = False
        if nbOfImgs == 1:
            img = sitk.ReadImage(filename+str(firstNb)+fileExtension)
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


    # Sub samples the array: could be used to minimize the noise
    #  subfactor: nb of pixels to sum in x and y direction to make one new pixel
    #  ATTENTION: QUAND CETTE FONCTION EST UTILISEE, IL N'EST PLUS POSSIBLE
    #  D'UTILISER LA FONCTION D'ENREGISTREMENT DE L'IMAGE EN TIF. LE SPACING
    #  UTILISE EST CELUI DE L'IMAGE INITIALE.
    #  JE N'ARRIVE PAS A CHANGER CA. CA FONCTIONNE DANS IMAGEJ MAIS PAS DANS 
    #  VERISOFT (il ne veut meme pas ouvrir l'image...)
    def subSampleDataArray(self,subfactor):
        self._sizex = int(self._sizex/subfactor)
        self._sizey = int(self._sizey/subfactor)
        #self._imgSpacing = (self._imgSpacing[0]*subfactor, self._imgSpacing[1]*subfactor)  # le pb vient de cette ligne
        self._array = self._array[0:self._sizey*subfactor, 0:self._sizex*subfactor, :]\
                        .reshape((self._sizey, subfactor, self._sizex, subfactor, 3)).mean(3).mean(1)


    # Crops the RGB image
    #  x0, x1: first and last pixel position in x direction 
    #  y0, y1: first and last pixel position in y direction 
    def cropImg(self, x0, x1, y0, y1):
        if (0<=x0<x1<self._sizex) and (0<=y0<y1<self._sizey):
            self._sizex = x1-x0
            self._sizey = y1-y0
            self._array = self._array[y0:y1, x0:x1, :]
            return True
        else:
            print('The dimensions do not match !')
            return False


    # Correct non uniformity of scanner response. For this correction, a non irradiated film
    #  must be scanned above/below the irradiated film.
    #  y0, y1: lines to be used for the correction (uniform non irradiated film)
    #  returnResults: if True, returns the main calculation values
    def correctScannerResponse(self, y0, y1, filterValue=25, returnResults=False):
        if not self._scannerCorr :
            profileR = np.mean(self._array[y0:y1,:,0], 0)
            profileG = np.mean(self._array[y0:y1,:,1], 0)
            profileB = np.mean(self._array[y0:y1,:,2], 0)

            filteredProfileR = filters.gaussian(profileR, filterValue)
            filteredProfileG = filters.gaussian(profileG, filterValue)
            filteredProfileB = filters.gaussian(profileB, filterValue)

            mean = np.mean(filteredProfileR)
            corrR = mean / filteredProfileR
            mean = np.mean(filteredProfileG)
            corrG = mean / filteredProfileG
            mean = np.mean(filteredProfileB)
            corrB = mean / filteredProfileB

            for i in range(self._sizey):
                self._array[i,:,0] = self._array[i,:,0] * corrR
                self._array[i,:,1] = self._array[i,:,1] * corrG
                self._array[i,:,2] = self._array[i,:,2] * corrB

            self._array[self._array<1] = 1
            self._array[self._array>65534] = 65534

            self._scannerCorr = True

            if returnResults:
                return [profileR, profileG, profileB], [filteredProfileR, filteredProfileG, filteredProfileB], [corrR, corrG, corrB]
        else:
            print("Scanner response correction already applied.")


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
        self._array[self._array<1] = 1
        
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
        self._array[self._array<1]=1
 
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
        
        cs = CubicSpline(rbvalues[::-1], dose[::-1])

        # red channel over blue channel:
        rsb = self._array[:,:,0]/self._array[:,:,2]
        
        # converting in dose:
        doseimg = cs(rsb)
        doseimg[doseimg>dosemax] = dosemax
        doseimg[doseimg<0] = 0
        
        return doseimg
    

    # Converts the gafchromic image to dose using the red over blue pixel values and a spline
    #  conversion curve. In this version, the white pixels are not converted (dose = 0)
    #  coefs: calibration curve coefficients
    #  dosemax: maximum dose over which the dose is not calculated
    def convertToDose_cubicSplineFit_onlyFilm(self, splinefile, dosemax, blueVmax = 35000):
        # replaces every 65535 value in array with 65534 to avoid division by zero:
        self._array[self._array<1]=1
 
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
        
        cs = CubicSpline(rbvalues[::-1], dose[::-1])

        # red channel over blue channel:
        rsb = self._array[:,:,0]/self._array[:,:,2]
        
        # converting in dose:
        doseimg = cs(rsb)
        doseimg[self._array[:,:,2]>blueVmax] = 0
        doseimg[doseimg>dosemax] = dosemax
        doseimg[doseimg<0] = 0
        
        return doseimg

    
    # Saves the dose image to a tiff file that can be read using Verisoft
    # doseimg: img to save
    # filename: filename the dose img will be written to
    def saveToTiff(self, doseimg, filename):
        imagetif = sitk.Image([doseimg.shape[1],doseimg.shape[0]], sitk.sitkVectorUInt16, 3)

        imagetif.SetSpacing(self._imgSpacing)
        imagetif.SetOrigin(self._imgOrigin)
        
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

