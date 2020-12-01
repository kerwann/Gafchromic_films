import SimpleITK as sitk 
import numpy as np
import math

from scipy.signal.signaltools import wiener

from bokeh.plotting import figure, show
from bokeh.layouts import column








class GafchromicFilms:
    
    # Constructor
    #  filename: gafchromic tiff file to read
    #  if nbOfImgs=1, only the filename is used
    #  otherwise filename + firstNb + fileExtension
    def __init__(self, filename, firstNb=0, nbOfImgs=1, fileExtension='.tif', addMethod='median'):
        self.readImg(filename, firstNb, nbOfImgs, fileExtension, addMethod)


    # Reads the image
    #  filename: filename of the gafchromic tiff file to read
    #  if nbOfImgs=1, only the filename is used
    #  otherwise filename + firstNb + fileExtension
    def readImg(self, filename, firstNb=0, nbOfImgs=1, fileExtension='.tif', addMethod='median'):
        if nbOfImgs == 1:
            img = sitk.ReadImage(filename+str(firstNb)+fileExtension)
            self._sizex = img.GetWidth()
            self._sizey = img.GetHeight()
            self._imgOrigin = img.GetOrigin()
            self._imgSpacing = img.GetSpacing()
            self._array = sitk.GetArrayFromImage(img)
            self._multilinearCoef = [0, 0, 0]
            self._doses = []
            self._Ccalr = []
            self._Ccalg = []
            self._Ccalb = []
        else:
            self._multilinearCoef = [0, 0, 0]
            self._doses = []
            self._Ccalr = []
            self._Ccalg = []
            self._Ccalb = []
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


    # method called when print(GafchromicFilms instance)
    def __str__(self):
        return 'Class GafchromicFilms: \
                \n  * sizex: ' + str(self._sizex) + \
                '\n  * sizey: ' + str(self._sizey) + \
                '\n  * imgOrigin: ' + str(self._imgOrigin) + \
                '\n  * imgSpacing: ' + str(self._imgSpacing)


    # Returns the array of the gafchromic film image
    def getArray(self):
        return self._array


    # Returns the image size:
    def getSize(self):
        return (self._sizex, self._sizey)


    # Returns the img spacing:
    def getSpacing(self):
        return self._imgSpacing


    # Return the img origin:
    def getOrigin(self):
        return self._imgOrigin


    # Crops the array:
    #  x0, x1: first and last pixel in the x direction
    #  y0, y1: first and last pixel in the y direction
    def cropDataArray(self, x0, x1, y0, y1):
        self._array = self._array[y0:y1, x0:x1]



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



    # Filters the input data using a Wiener filter
    def applyWienerFilter(self):
        self._array[:,:,0] = wiener(self._array[:,:,0])
        self._array[:,:,1] = wiener(self._array[:,:,1])
        self._array[:,:,2] = wiener(self._array[:,:,2])


    # Does a streak correction on images
    def applyStreakCorrection(self, a):
        
        # First pixels sum and normalization:
        firstPixAvg_r = np.sum(self._array[0:10,:,0] ,axis=0)
        firstPixAvg_g = np.sum(self._array[0:10,:,1] ,axis=0)
        firstPixAvg_b = np.sum(self._array[0:10,:,2],axis=0)

        firstPixAvg_r = firstPixAvg_r / np.mean(firstPixAvg_r)
        firstPixAvg_g = firstPixAvg_g / np.mean(firstPixAvg_g)
        firstPixAvg_b = firstPixAvg_b / np.mean(firstPixAvg_b)


        # Last pixels sum and normalization:
        lastPixAvg_r = np.sum(self._array[-9:,:,0] ,axis=0)
        lastPixAvg_g = np.sum(self._array[-9:,:,1] ,axis=0)
        lastPixAvg_b = np.sum(self._array[-9:,:,2],axis=0)

        lastPixAvg_r = lastPixAvg_r / np.mean(lastPixAvg_r)
        lastPixAvg_g = lastPixAvg_g / np.mean(lastPixAvg_g)
        lastPixAvg_b = lastPixAvg_b / np.mean(lastPixAvg_b)


        # Streak correction image:
        #  This correction must be done on an uncroped image
        streakCorrImage = np.zeros(self._array.shape)
        x = [0, self._sizey-1]

        for i in range(self._sizex):
            allpix_y = range(self._sizey)
            y0 = [firstPixAvg_r[i], lastPixAvg_r[i]]
            streakCorrImage[:,i,0] = 1 / np.interp(allpix_y, x, y0)
    
            y1 = [firstPixAvg_g[i], lastPixAvg_g[i]]
            streakCorrImage[:,i,1] = 1 / np.interp(allpix_y, x, y1)

            y2 = [firstPixAvg_b[i], lastPixAvg_b[i]]
            streakCorrImage[:,i,2] = 1 / np.interp(allpix_y, x, y2)
    
    
        # Apply streak correction to the image:
        self._array = self._array * streakCorrImage


    # Calculates multilinear regression coefficients and the fingerprint at 
    #       the time of calibration.
    #  doses: array with all given doses (from 0 to 800cGy typically)
    #  rectangles: rectangles to be used to calculate mean pixel values
    #       the rectangles order must be the same as the doses order
    def multilinearRegression(self, doses, rectangles, dispResults=False):
        
        # Calculates RGB values in the 
        rvalues, gvalues, bvalues = [], [], []
        for i in range(len(rectangles)):
            rvalues.append(np.mean(self._array[rectangles[i][1]:rectangles[i][3],
                                 rectangles[i][0]:rectangles[i][2], 0]))
            gvalues.append(np.mean(self._array[rectangles[i][1]:rectangles[i][3],
                                 rectangles[i][0]:rectangles[i][2], 1]))
            bvalues.append(np.mean(self._array[rectangles[i][1]:rectangles[i][3],
                                 rectangles[i][0]:rectangles[i][2], 2]))


        # Calculates nPVrgb:
        nPVr, nPVg, nPVb = [], [], []
        regressiondoses = []
        if doses[0] == 0 :
            for i in range(1, len(rvalues)):
                ctrlIndex = 0
                nPVr.append(rvalues[ctrlIndex]/rvalues[i]-1)
                nPVg.append(gvalues[ctrlIndex]/gvalues[i]-1)
                nPVb.append(bvalues[ctrlIndex]/bvalues[i]-1)
            regressiondoses = doses[1:]
        elif doses[-1] == 0:
            for i in range(0, len(rvalues)-1):
                ctrlIndex = len(rvalues)-1
                nPVr.append(rvalues[ctrlIndex]/rvalues[i]-1)
                nPVg.append(gvalues[ctrlIndex]/gvalues[i]-1)
                nPVb.append(bvalues[ctrlIndex]/bvalues[i]-1)
            regressiondoses = doses[0:-1]

            # changement d'ordre des matrices pour avoir des doses croissantes:
            regressiondoses = regressiondoses[::-1]
            nPVr = nPVr[::-1]
            nPVg = nPVg[::-1]
            nPVb = nPVb[::-1]
        else:
            print("eRROR: missing ctrl film at first or last place...")
            return


        # Calculates the fingerprint:
        self._doses = doses
        self._Ccalr, self._Ccalg, self._Ccalb = [], [], []
        for i in range(0, len(rvalues)):
            self._Ccalr.append(rvalues[i]/(rvalues[i]+gvalues[i]+bvalues[i]))
            self._Ccalg.append(gvalues[i]/(rvalues[i]+gvalues[i]+bvalues[i]))
            self._Ccalb.append(bvalues[i]/(rvalues[i]+gvalues[i]+bvalues[i]))
        if doses[-1] == 0:
            self._doses = self._doses[-1]
            self._Ccalr = self._Ccalr[-1]
            self._Ccalg = self._Ccalg[-1]
            self._Ccalb = self._Ccalb[-1]


        self._multilinearCoef = np.linalg.lstsq(np.array([[nPVr[i], nPVg[i], nPVb[i]] for i in range(len(nPVr))]), regressiondoses, rcond=None)[0]


        if dispResults :
            print('Calculated coefficients:')
            print(self._multilinearCoef)

            p1 = figure(plot_width=700, plot_height=400, title='RGB values', toolbar_location="above")
            p1.xaxis.axis_label = "Dose"
            p1.yaxis.axis_label = "nPV"
            p1.line(regressiondoses, nPVr, line_width=2, line_color='firebrick')
            p1.line(regressiondoses, nPVg, line_width=2, line_color='green')
            p1.line(regressiondoses, nPVb, line_width=2, line_color='blue')

            r_nPVr = [i * self._multilinearCoef[0] for i in nPVr]     # multiply every elt of a list with a value
            r_nPVg = [i * self._multilinearCoef[1] for i in nPVg]
            r_nPVb = [i * self._multilinearCoef[2] for i in nPVb]
            nPVrgb = [i+j+k for i,j,k in zip(r_nPVr,r_nPVg,r_nPVb)]
            p2 = figure(plot_width=700, plot_height=400, title='nPVrgb vs dose', toolbar_location="above")
            p2.xaxis.axis_label = "Dose"
            p2.yaxis.axis_label = "r*nPV or nPVrgb"
            p2.line(regressiondoses, regressiondoses, line_width=1, line_dash="4 4",line_color='black')
            p2.line(regressiondoses, nPVrgb, line_width=2, line_color='black')

            p3 = figure(plot_width=700, plot_height=400, title='Fingerprints', toolbar_location="above")
            p3.xaxis.axis_label = "Dose"
            p3.yaxis.axis_label = "Fingerprints Ccal"
            p3.line(regressiondoses, self._Ccalr, line_width=2, line_color='firebrick', legend='Ccal_r')
            p3.line(regressiondoses, self._Ccalg, line_width=2, line_color='green', legend='Ccal_g')
            p3.line(regressiondoses, self._Ccalb, line_width=2, line_color='darkblue', legend='Ccal_b')

            show(column(p1,p2,p3))




    # Saves the multilinear regression coefficients in a text file
    #  filename: filename to read
    #  batchNb: batch number
    #  dispStatus: display results
    def saveMultilinearRegressionFile(self, filename, batchNb="WTF007", dispStatus=False):

        with open(filename, "w") as f:
            f.write("Multilinear regression coefficients for batch nb "+batchNb+" \n\n")
            f.write(str(self._multilinearCoef[0])+'\t'+str(self._multilinearCoef[1])+'\t'+str(self._multilinearCoef[2]))
            f.write("\n\n\n\n\nInitial Fingerprints:\n")
            f.write("Nb of points: "+str(len(self._doses))+"\n\n")
            f.write("Doses:\n")
            for i in range(len(self._doses)): f.write(str(self._doses[i])+"\t")
            f.write("\n\nCcal R:\n")
            for i in range(len(self._doses)): f.write(str(self._Ccalr[i])+"\t")
            f.write("\n\nCcal G:\n")
            for i in range(len(self._doses)): f.write(str(self._Ccalg[i])+"\t")
            f.write("\n\nCcal B:\n")
            for i in range(len(self._doses)): f.write(str(self._Ccalb[i])+"\t")
            

        if dispStatus:
            print('Multilinear coefficients: ', self._multilinearCoef)
            print('Multilinear coefficients saved in file:', filename)


    # Reads the multilinear regression coefficients in a text file
    #  filename: filename to read
    #  dispStatus: display results
    def readMultilinearRegressionFile(self, filename, dispStatus=False):
        
        nbofpoints = 0
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i == 2:
                    s = line.split("\t")
                    self._multilinearCoef[0] = float(s[0])
                    self._multilinearCoef[1] = float(s[1])
                    self._multilinearCoef[2] = float(s[2])
                if 'Nb of points' in line:
                    s = line.split(":")
                    nbofpoints = int(s[1])
                if i == 11: # Doses
                    s = line.split("\t")
                    for i in range(len(s)-1): self._doses.append(float(s[i]))
                if i == 14: # CcalR
                    s = line.split("\t")
                    for i in range(len(s)-1): self._Ccalr.append(float(s[i]))
                if i == 17: # CcalR
                    s = line.split("\t")
                    for i in range(len(s)-1): self._Ccalg.append(float(s[i]))
                if i == 20: # CcalR
                    s = line.split("\t")
                    for i in range(len(s)-1): self._Ccalb.append(float(s[i]))

        if dispStatus:
            print("Multilinear Regression File Read !")
            print("\nCoefs:", self._multilinearCoef)
            print("\nDoses:", self._doses)
            print("\nCcal R:", self._Ccalr)
            print("\nCcal G:", self._Ccalg)
            print("\nCcal B:", self._Ccalb)




    # Converts the gafchromic image to dose using multilinear regression coefs
    #  doserect: rectangle of the image to convert to dose
    #  ctrlrect: rectangle to use for the ctrl pixel values (non irradiated film)
    #  dosemin: minimum dose (values below dosemin will be set to dosemin)
    #  dosemax: maximum dose (values above dosemax will be set to dosemax)
    def convertToDose(self, doserect, ctrlrect, dosemin=0, dosemax=850):

        if self._multilinearCoef[0] == 0: 
            print('You must set the multilinear regression coefficients first !')
            return None

        # Finds mean values of R, G and B in the ctrl film
        ctrl_r = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 0])
        ctrl_g = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 1])
        ctrl_b = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 2])


        # Computes the dose image using the coefs determined above:
        doseImg = self._multilinearCoef[0]*(ctrl_r / self._array[doserect[1]:doserect[3], doserect[0]:doserect[2],0] - 1) + \
                    self._multilinearCoef[1]*(ctrl_g / self._array[doserect[1]:doserect[3], doserect[0]:doserect[2],1] - 1) + \
                    self._multilinearCoef[2]*(ctrl_b / self._array[doserect[1]:doserect[3], doserect[0]:doserect[2],2] - 1)

        doseImg[doseImg>dosemax] = dosemax
        doseImg[doseImg<dosemin] = dosemin
        
        return doseImg
 


    # Use the dose image and the fingerprints at calibration to iterate a new dose image
    # array: array to convert to dose
    # doseimg: dose image to use as a begin image
    # ctrl_values: mean values of R, G and B in the ctrl film
    # cmeasr, cmeasg, cmeasb: measured values of Cmeas_r, Cmeas_g and Cmeas_b 
    #     for each pixel (stored in an array)
    def fingerprintIteration(self, array, doseimg, ctrl_values, cmeasr, cmeasg, cmeasb):
    
        # Calculation of the interpolated Ccal for each pixel
        ccalr = np.interp(doseimg, self._doses, self._Ccalr)
        ccalg = np.interp(doseimg, self._doses, self._Ccalg)
        ccalb = np.interp(doseimg, self._doses, self._Ccalb)

        meanCcal = [np.mean(ccalr), np.mean(ccalg), np.mean(ccalb)]

        # Computes correction Factors:
        cr = cmeasr / ccalr
        cg = cmeasg / ccalg
        cb = cmeasb / ccalb

        # New dose image:
        imgCorr = cr*self._multilinearCoef[0]*(ctrl_values[0] / array[:,:,0] - 1) + \
                    cg*self._multilinearCoef[1]*(ctrl_values[1] / array[:,:,1] - 1) + \
                    cb*self._multilinearCoef[2]*(ctrl_values[2] / array[:,:,2] - 1)

        return imgCorr, meanCcal



    
    # Converts the gafchromic image to dose using multilinear regression coefs and 
    #          fingerprint correction.
    #  doserect: rectangle of the image to convert to dose
    #  ctrlrect: rectangle to use for the ctrl pixel values (non irradiated film)
    #  dosemin: minimum dose (values below dosemin will be set to dosemin)
    #  dosemax: maximum dose (values above dosemax will be set to dosemax)
    #  dispStatus: set to True to print information during the conversion process
    #  convThreashold: threashold value after which dose is considered to be converged
    def convertToDoseWithFingerPrint(self, doserect, ctrlrect, dosemin=0, dosemax=850, dispStatus=False, convThreashold=0.005):

        # checks multilinear coefficients first: 
        if self._multilinearCoef[0] == 0: 
            print('You must set the multilinear regression coefficients first !')
            return None

        # checks for the fingerprints at the time of calibration:
        if self._Ccalr == [] or self._Ccalb == [] or self._Ccalg == [] :
            print('You must set the fingerprints first !')
            return None


        if dispStatus: print("Conversion to dose with fingerprints initiated :")

        # Finds mean values of R, G and B in the ctrl film
        ctrl_r = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 0])
        ctrl_g = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 1])
        ctrl_b = np.mean(self._array[ctrlrect[1]:ctrlrect[3],
                                 ctrlrect[0]:ctrlrect[2], 2])
        ctrl_values = [ctrl_r, ctrl_g, ctrl_b]


        # Crops the input image
        toDose_array = self._array[doserect[1]:doserect[3], doserect[0]:doserect[2],:]


        # Computes the dose image using the coefs determined above:
        doseImg = self._multilinearCoef[0]*(ctrl_r / toDose_array[:,:,0] - 1) + \
                    self._multilinearCoef[1]*(ctrl_g / toDose_array[:,:,1] - 1) + \
                    self._multilinearCoef[2]*(ctrl_b / toDose_array[:,:,2] - 1)
        if dispStatus: print("  - 1st dose image computed")


        # Computes the fingerprints at measurements time:
        Cmeas_r = toDose_array[:,:, 0] / (toDose_array[:,:, 0] \
                    + toDose_array[:,:, 1] + toDose_array[:,:, 2])
        Cmeas_g = toDose_array[:,:, 1] / (toDose_array[:,:, 0] \
                    + toDose_array[:,:, 1] + toDose_array[:,:, 2])
        Cmeas_b = toDose_array[:,:, 2] / (toDose_array[:,:, 0] \
                    + toDose_array[:,:, 1] + toDose_array[:,:, 2])
        if dispStatus: print("  - Fingerprints images computed")


        # Calculate Ccal and new image dose
        doseImg, meanCcal0 = self.fingerprintIteration(toDose_array, doseImg, ctrl_values, \
                                              Cmeas_r, Cmeas_g, Cmeas_b)
        
        doseImg[doseImg>dosemax] = dosemax
        doseImg[doseImg<dosemin] = dosemin

        if dispStatus: print("  - First fingerprint iteration done")

        err = 1
        i = 0
        while (err > convThreashold) and (i<100):
            doseImg, meanCcalIter = self.fingerprintIteration(toDose_array, doseImg, ctrl_values, \
                                                     Cmeas_r, Cmeas_g, Cmeas_b)
            doseImg[doseImg>dosemax] = dosemax
            doseImg[doseImg<dosemin] = dosemin

            err = max([(meanCcalIter[0] - meanCcal0[0])/meanCcal0[0], \
                            (meanCcalIter[1] - meanCcal0[1])/meanCcal0[1], \
                            (meanCcalIter[2] - meanCcal0[2])/meanCcal0[2] ])

            i += 1
            if dispStatus:
                print("  - Iter no:", i)
                print("      meanCcal iteration:", meanCcalIter)
                print("      meanCcal 0:", meanCcal0)
                print("      convergence R:", (meanCcalIter[0]-meanCcal0[0])/meanCcalIter[0]*100, "%")
                print("      convergence R:", (meanCcalIter[1]-meanCcal0[1])/meanCcalIter[1]*100, "%")
                print("      convergence R:", (meanCcalIter[2]-meanCcal0[2])/meanCcalIter[2]*100, "%")
                print("      max error:", err)
                print("\n")

            meanCcal0 = meanCcalIter


        doseImg[doseImg>dosemax] = dosemax
        doseImg[doseImg<dosemin] = dosemin
        
        if dispStatus: print("\ncONVERSION dONE !")

        return doseImg



     
    # Saves the dose image to a tiff file that can be read using Verisoft
    # doseimg: img to save
    # filename: filename the dose img will be written to
    def saveToTiff(self, doseimg, filename):

        imagetif = sitk.Image([doseimg.shape[1],doseimg.shape[0]], sitk.sitkVectorUInt16, 3)

        imagetif.SetSpacing(self._imgSpacing)
        #imagetif.SetOrigin(self._imgOrigin)  # probablement le probleme!

        for j in range(0, doseimg.shape[0]):
            for i in range(0, doseimg.shape[1]):
                a = int(doseimg[j,i])
                imagetif.SetPixel(i,j,[a, a, a])

        writer = sitk.ImageFileWriter()
        writer.SetFileName(filename)
        writer.Execute(imagetif)
        return True




# Main file
if __name__ == '__main__':
    filename = 'G:/Commun/PHYSICIENS/Erwann/EBT3/09 - etalonnage lot 10151801 19juil2019/films a 24h/scan'
    nbofimgs = 5
    firstimg = 1
    m_splineFile = 'G:/Commun/PHYSICIENS/Erwann/EBT3/09 - etalonnage lot 10151801 19juil2019/films a 24h/bSpline_data.txt'
    m_dosemax = 1000.0 #cGy
    
    try:
        g = GafchromicFilms(filename, firstimg, nbofimgs)
    except ValueError as err:
        print('Erreur: '+err)

    print(g)
    plt.figure(1, figsize=(10,10))
    plt.imshow(g.getArray()[:,:,0])
    plt.imshow(doseimg)
    plt.show()

