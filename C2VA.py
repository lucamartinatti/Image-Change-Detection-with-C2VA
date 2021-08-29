'''
Apply Change Vector Analysis (CVA) to 2 RGB images with no reference.
'''
import os
import numpy as np
from scipy.stats import gaussian_kde
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def loadOneraDataset(basepath, image_with_groundtruth= False, listOfPlaces = False):  
    if image_with_groundtruth == True:
        f = open(Bpath+'train.txt','r') 
        content = f.read() 
        nameList = content.split(',') 
    
    for i in range(2):
        ls=[] 
        lEntry=[] 
        for entry in os.listdir(basepath):
            if os.path.isdir(os.path.join(basepath, entry)):
                if image_with_groundtruth == True:
                    if any(entry in s for s in nameList):
                        if i == 0:
                            im = '/imgs_1_rect/' 
                        elif i == 1:
                            im = '/imgs_2_rect/' 
                        path = basepath + entry + im 
                        fil_list = os.listdir(path) 
                        x = np.array([np.array(Image.open(path+fname)) for fname in fil_list]) 
                        lEntry.append(entry) 
                        ls.append(x) 
                elif image_with_groundtruth == False:
                    if i == 0:
                        im = '/imgs_1_rect/' 
                    elif i == 1:
                        im = '/imgs_2_rect/' 
                    path = basepath + entry + im 
                    fil_list = os.listdir(path) 
                    x = np.array([np.array(Image.open(path+fname)) for fname in fil_list]) 
                    lEntry.append(entry) 
                    ls.append(x) 
        if i == 0:
            ls1 = ls 
        elif i == 1: 
            ls2 = ls 
    if listOfPlaces == True:
        return ls1, ls2, lEntry 
    else:
        return ls1, ls2 

def loadGroundTruth(basepath, listOfPlaces = False):
    ls=[] 
    lEntry=[] 
    for entry in os.listdir(basepath):
        if entry != 'README.txt':
            if os.path.isdir(os.path.join(basepath, entry)):
                path = basepath + entry + '/cm/' 
                x = np.array(Image.open(path+entry+'-cm.tif')) 
                lEntry.append(entry) 
            ls.append(x-1)  # original is ls.append(x)   the -1 is to convert ground truth from [2,1] range into [1,0] range 
    if listOfPlaces == True:
        return ls, lEntry 
    else:
        return ls 

def c2va(iBefore, iAfter):
    # Magnitude
    dif = np.zeros((iBefore.shape[0], iBefore.shape[1], iBefore.shape[2])) 
    for i in range(iBefore.shape[0]):
        #dif[i] = cv2.absdiff(iBefore[i], iAfter[i]) 
        dif[i] = iAfter[i] - iBefore[i] 
        if i == 0:
            sumX = dif[i]**2 
        else:
            sumX += dif[i]**2 
    dMag = np.sqrt(sumX) 

    # Direction
    xRef = np.ones(iBefore.shape[0])*(np.divide(np.sqrt(iBefore.shape[0]),iBefore.shape[0])) 
    arg = np.zeros((iBefore.shape[1],iBefore.shape[2])) 
    #for i in range(arg.shape[0]):
    #    for j in range(arg.shape[1]):
    Num = np.zeros(dMag.shape) 
    Den = np.zeros(dMag.shape) 
    tmp = np.zeros(dMag.shape) 
    for b in range(dif.shape[0]):
        Num += dif[b]*xRef[b] 
        tmp += dif[b]**2 
    Den = np.sqrt(tmp*(sum(xRef**2))) 
    
    out = np.ones((iBefore.shape[1], iBefore.shape[2]))*0.0000001  # Zero value are redefined as small number 0.0000001 (see below)
    dDir = np.arccos(np.divide(Num, Den, out= out, where= Den != 0))  # To avoid nan problem due to an dominator equal to zero
    return dMag, dDir     

def uint16_to_uint8(img16):
    tmp = [] 
    for i in range(len(img16)):
        #image = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) 
        #img8_before.append(cv2.convertScaleAbs(img_before[i], None))  
        tmp.append(np.uint8(cv2.normalize(img16[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))) 
    return tmp    

def imageSplitting(mag, dir, image=None):
    '''
    imageSplitting(mag, dir, image=None)
        
        PARAMETERS:    
                mag = 2-D array like magnitude image (image raw, image column)

                dir = 2-D array like direction image (image raw, image column)

                image = 3-D array like multispectral images (bands, image raw, image column)

        RETURNS: 
                structure:  [out_Mag, out_Dir, out_Image]
                
                ls = list of 2-D array which contain:
                        |
                        |----N x M split (not all are equal) which contain:  
                            |    
                            |---- I x J pixel of one of the input  (not all are equal)              
    '''
    # TODO implement also the ouput for image
    a, b = mag.shape[:] 

    rSize = 100 
    cSize = 100 
    ls = [] 

    splitList = [mag, dir] 

    rSizeDimensioning = True 
    cSizeDimensioning = True 
    while rSizeDimensioning:
        if (a%rSize) == 0:
            rFinal = (a%rSize) 
            rSizeDimensioning = False 
        elif (a%rSize) <= (0.35*rSize):
            rSize += 1 
        elif (a%rSize) >= (0.75*rSize):
            rFinal = (a%rSize) 
            rSizeDimensioning = False 
        else:
            rSize -=1 
    while cSizeDimensioning:
        if (b%cSize) == 0:
            cFinal = (b%cSize) 
            cSizeDimensioning = False 
        elif (b%cSize) <= (0.35*cSize):
            cSize += 1 
        elif (b%cSize) >= (0.75*cSize):
            cFinal = (b%cSize) 
            cSizeDimensioning = False 
        else:
            cSize -=1 
    
    for sl in splitList:
        window = [] 
        for r in range(0,a - rSize, rSize):
            tmp= [] 
            for c in range(0,b - cSize, cSize):
                tmp.append(sl[r:r+rSize,c:c+cSize]) 
                if (c + cSize + cFinal) == b:
                    tmp.append(sl[r:r+rSize,c+cSize:c+cSize+cFinal]) 
            if tmp != []:
                window.append(tmp) 
            tmp=[] 
            if (r + rSize + rFinal) == a:
                for c in range(0,b - cSize, cSize):
                    tmp.append(sl[r+rSize:r+rSize+rFinal,c:c+cSize]) 
                    if (c + cSize + cFinal) == b:
                        tmp.append(sl[r+rSize:r+rSize+rFinal,c+cSize:c+cSize+cFinal]) 
            if tmp != []:
                window.append(tmp) 
        win = np.array(window) 
        ls.append(win) 
    return ls 

def splitSelection(split, selection_value=0, num_of_split = 6):
    '''
    splitSelection(split, selection_value=0, num_of_split = 6)

        PARAMETERS:
                split = list of splitted images produced by "imageSplitting" function

                selection_value = int, optional
                                  Decide which element of split use to make the seletion. 
                                  Default value is '0' indicate that use magnitude for selection

                num_of_split = int, optional
                               Indicate the number of split selected (default value = 6)

        RETURNS:
                structure:  [out_Mag, out_Dir, out_Image]

                ls = list of 'num_of_split' selected split.
    '''
    ls=[] 
    sp = split[selection_value] 
    standDev = np.zeros((sp.shape[0],sp.shape[1])) 
    for r in range(sp.shape[0]):
        for c in range(sp.shape[1]):
            standDev[r,c] = sp[r,c].std() 
    
    mag=[] 
    dir=[] 
    img=[] 
    for i in range(num_of_split):
        ind = np.unravel_index(np.argmax(standDev, axis=None), standDev.shape) 
        mag.append(split[0][ind]) 
        dir.append(split[1][ind]) 
        if len(split) == 3:
            img.append(split[2][ind]) 
        standDev[ind] = 0.0 
    ls=[mag, dir, img] 
    return ls 

def CDmap(image, threshold= None):
    # Normalize image between 0 and max
    im = image - image.min() 
    if threshold != None:
        T = threshold 
    else:
        rang = im.max() 
        if abs(rang) < 1.0:
            T = round((rang/4), ndigits=1) 
        else:
            # Shift threshold from 1/2 to be more sensitive
            T = round(rang/4) 
    t, it = cv2.threshold(im, T, 1, type= cv2.THRESH_BINARY) 
    w = np.count_nonzero(it) 
    b = np.count_nonzero(it==0) 
    if w > b:
        ti = abs(it-1) 
        return np.array(ti, dtype=int) 
    else:
        return np.array(it, dtype=int) 

def accuracy(prediction,real):
    '''
    accuracy(prediction, real)
        
        PARAMETERS: 
                prediction = 1-D or 2-D array like
                             Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
                             and 1 (or 255) indicate the presence of changing. 
                
                real = 1-D or 2-D array like
                       Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
                       and 1 (or 255) indicate the presence of changing. 
        
        RETURNS:
                acc = accuracy of prediction w.r.t. the real case.  Evaluated as (TP + TN)/ (TN + FN + FP + TP)
    '''
    if prediction.shape == real.shape:
        error = abs(prediction - real) 
        summa = np.count_nonzero(error) 
        acc = ((prediction.size-summa)/prediction.size) * 100 
        return acc 
    else: 
        print('Error sizes, accuracy function has different size input') 

def confusionMatrix(prediction,real):
    '''
    confusionMatrix(prediction, real)
        
        PARAMETERS: 
                prediction = 1-D or 2-D array like
                             Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
                             and 1 (or 255) indicate the presence of changing. 
                
                real = 1-D or 2-D array like
                       Binary ([0,1] or [0,255]) array or matrix where 0 indicate absence of changing
                       and 1 (or 255) indicate the presence of changing. 
        
        RETURNS:
                structure:  [acc, P_err, P_fa, P_ma]

                acc = accuracy of prediction w.r.t. the real case.  Evaluated as (TP + TN)/ (TN + FN + FP + TP). 
                      It is express in percentage (%).

                f1 = F1 score.

                P_fa = probability of false allarm. Evaluated as FP / (TP + FP). It is express in percentage (%).

                P_ma = probability of miss allarm. Evaluated as FN / (TN + FN). It is express in percentage (%).
                
    '''
    if prediction.shape == real.shape:
        error = prediction - real 
        FP = np.count_nonzero(error > 0) 
        FN = np.count_nonzero(error < 0) 
        true = np.count_nonzero(error == 0)  # TP + TN
        pos = np.count_nonzero(prediction)  # TP + FP
        neg = np.count_nonzero(prediction ==0)  # TN + FN
        TN = neg - FN 
        TP = pos - FP 

        P_fa = (FP/ pos)*100 
        P_ma = (FN/ neg)*100 
        rec = TP / (TP + FN) 
        pre = TP / (TP + FP) 

        f1 = (2 * rec * pre)/(rec + pre) 
        
        summa = np.count_nonzero(error) 
        acc = ((prediction.size-summa)/prediction.size)*100 
        ls = [acc, f1, P_fa, P_ma] 
        return ls 
    else: 
        print('Error sizes, accuracy function has different size input') 

def parzenWindow(mag):
    mag = mag.ravel() 
    #x_grid = np.linspace(mag.min(), mag.max(), round(mag.max())) 
    sort = np.round(np.sort(mag))  #np.round(np.sort(mag), decimals=1) 
    n, bins, patches = plt.hist(sort, bins= 'auto') 
    print('')

def removeBands(before, after, bandsList):
    bL = (np.array(bandsList))-1 
    if bandsList != []:
        i_before = [] 
        i_after = [] 
        for im in range(len(before)):
            tmpA = np.zeros((before[im].shape[0]-len(bandsList),before[im].shape[1],before[im].shape[2])) 
            tmpB = np.zeros((before[im].shape[0]-len(bandsList),before[im].shape[1],before[im].shape[2])) 
            z = 0 
            for b in range(before[im].shape[0]):
                if b in bL:
                    pass
                else:    
                    tmpA[z] = after[im][b] 
                    tmpB[z] = before[im][b] 
                    z +=1 
            i_after.append(tmpA) 
            i_before.append(tmpB) 
        return i_before, i_after 
    else:
        return before, after 

def graphs():

    # Plot the Magnitude histogram graph
    his = plt.figure(figsize=(8,8)) 
    n, bins, patches = plt.hist(dMag.ravel(), bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85) 
    plt.grid(axis='y', alpha=0.75) 
    plt.xlabel(r'X$\rho$') 
    plt.ylabel(r'h(X$\rho)$') 
    plt.title('Magnitude', fontsize=20) 
    plt.text(23, 45, r'$\mu=15, b=3$') 
    plt.show()

    # Plot the magnitude and phase images
    fig, ax = plt.subplots(1,2)
    fig.suptitle('Magnitude and Phase images', fontsize=20)
    ax[0].imshow(dMag)
    ax[0].set_title(r'Magnitude value $\rho$') 
    ax[1].imshow(dDir)
    ax[1].set_title(r'Phase value $\theta$') 
    plt.show()

    # Plot the comparison between the true and the producted Change Detection maps
    fig, ax = plt.subplots(1,3)
    fig.suptitle('Change Detection maps', fontsize=20)
    cdM = [gT, CDmap(dMag), CDmap(dMag) ] 
    titles = ['Ground truth','Magnitude', 'Direction'] 
    ax[0].imshow(cdM[0], cmap='gray', interpolation='nearest') 
    ax[0].set_title(titles[0]) 
    ax[1].imshow(cdM[1], cmap='gray', interpolation='nearest') 
    confM = confusionMatrix(cdM[1], gT) 
    ax[1].set_title(titles[1] + ' acc= %.2f' %confM[0] +'%' + ' f1= %.2f' %confM[1]) 
    ax[2].imshow(cdM[2], cmap='gray', interpolation='nearest') 
    confM = confusionMatrix(cdM[2], gT) 
    ax[2].set_title(titles[2] + ' acc= %.2f' %confM[0] +'%' + ' f1= %.2f' %confM[1]) 
    plt.show() 


if __name__ == "__main__":

    ### LOAD DATA AND PRE-PROCESSING ###

    # Onera Dataset folders paths
    Bpath = 'Onera Satellite Change Detection dataset - Images/' 
    GtPath = 'Onera Satellite Change Detection dataset - Train Labels/' 

    # Load dataset and split it in before/after change images 
    img_before, img_after = loadOneraDataset(Bpath,image_with_groundtruth=True) 

    # Choose the list of images to analyze (Alphabetically ordered)
    imageList =[0] 

    # Remove unuseful bands for change detection
    unuseful_bands = [10] 
    i_before, i_after = removeBands(img_before, img_after, unuseful_bands) 

    # Load ground truth
    groundTr = loadGroundTruth(GtPath) 


    ### IMAGE ANALYSIS ###

    for im in imageList:

        # Apply C2VA Amalysis
        dMag, dDir = c2va(i_before[im], i_after[im]) 

        # Split image
        splittedImage = imageSplitting(dMag, dDir) 

        gT = groundTr[im] 

        ### PLOT GRAPHS ###
        graphs() 
        
    

