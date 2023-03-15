# import
import cv2
from PIL import Image
from scipy import signal,ndimage
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import openpyxl 
import pickle
import os as os
import tiffile as tif
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy.matlib as matlib


# create definitions
def crop_box(im,xvals,yvals):
    # Redraw the box to confirm it what they want
    # Determine the Rectangle lower left point
    ll_y = math.ceil(min(yvals))
    ll_x = math.ceil(min(xvals))
    # Determine rectangle width
    width = math.ceil(max(xvals)) - ll_x
    # Determine rectangle height
    height = math.ceil(max(yvals)) - ll_y
    fig, ax = plt.subplots(1, figsize=(1248 / 200, 1024 / 200))
    ax.imshow(im)
    # Add rectangle
    ax.add_patch(Rectangle((ll_x, ll_y), width, height, edgecolor='red', fill=False))
    plt.title(f"Cropped Area Sample")
    plt.show()
    return ll_x, ll_y, width, height

def RRC(img_path, rotate_crop_params,package):
    '''
    Rotates and crops the given image.

    Inputs:
    img                  := image path
    rotate_crop_params   := dictionary of values: {theta, x1, x2, y1, y2}, where
        theta            := angle of counter clockwise rotation
        x1               := start pixel of x-axis crop
        x2               := end pixel of x-axis crop
        y1               := start pixel of y-axis crop
        y2               := end pixel of y-axis crop

    Ouputs:
    img                  := rotated and cropped image
    '''
    if package == 'cv2':
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # read images 
    elif package == 'pil':
        img = Image.open(img_path, 'r')
    rotated = ndimage.rotate(img, rotate_crop_params['theta'])  # reads image and rotates
    img = rotated[rotate_crop_params['y1']:rotate_crop_params['y2'],
          rotate_crop_params['x1']:rotate_crop_params['x2']]  # crops image
    return img

def segment_on_dt(a, img, threshold):
    '''
    Implements watershed segmentation.

    Inputs:
    a         := the raw image input
    img       := threshold binned image
    threshold := RGB threshold value

    Outputs:
    lbl       := Borders of segmented droplets
    wat       := Segmented droplets via watershed
    lab       := Indexes of each segmented droplet
    '''
    # estimate the borders of droplets based on known and unknown background + foreground (computed using dilated and erode)
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, None)
    # segment droplets via distance mapping and thresholding
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, threshold, 255, cv2.THRESH_BINARY)
    # obtain the map of segmented droplets with corresponding indices
    lbl, ncc = ndimage.label(dt)
    lbl = lbl * (255 / (ncc + 1))
    lab = lbl
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(np.int32)
    a = cv2.cvtColor(a,
                     cv2.COLOR_GRAY2BGR)  # we must convert grayscale to BGR because watershed only accepts 3-channel inputs
    wat = cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl, wat, lab  # return lab, the segmented and indexed droplets

def water(image, small_elements_pixels, large_elements_pixels):
    '''
    Applies watershed image segmentation to separate droplet pixels from background pixels.

    Inputs:
    image                   := input droplet image to segment
    large_elements_pixels   := Cleans large elements that contain more than specified number of pixels.


    Outputs:
    droplet_count           := Image of droplet interiors indexed by droplet number
    binarized               := Binary image indicating total droplet area vs. empty tube space
    ''' 
    RGB_threshold = 0
    pixel_threshold = 0
    # Added these Lines 
    # kernel = np.ones((12,12), np.uint8)
   # image= cv2.erode(image, kernel)
    # added these lines 
    img = image.copy()
    img = 255 - img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 3)
    _, img_bin = cv2.threshold(img, 0, 255,
                               # threshold image using Otsu's binarization # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
                               cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                               np.ones((4, 4), dtype=int))
    # first fold of watershed to remove white centers
    result, water, labs = segment_on_dt(a=img, img=img_bin,
                                        threshold=RGB_threshold)  # segment droplets from background and return indexed droplets


    # remove small/large elements
    uniq_full, uniq_counts = np.unique(water,
                                       return_counts=True)  # get all unique watershed indices with pixel counts
    large_elements = uniq_full[uniq_counts > large_elements_pixels]  # mask large elements based on number of pixels
    small_elements = uniq_full[uniq_counts < small_elements_pixels] # mask small elements based on number of pixels
    for n in range(len(large_elements)):
        water[water == large_elements[n]] = 0  # remove all large elements
    for n in range(len(small_elements)):
        water[water == small_elements[n]] = 0  # remove all small elements

    result[result == 255] = 0
    droplet_count = result.copy()
    return water

# Define sample Class
class Sample:
    def __init__(self, crop_params, crop_img_pil, sample_ID,drop_IDs,drops, Number_of_drops, water, img_erode, crop_image_erode, PIL_crop_image_erode, img_name, Notes, PTL):
        # drops is a 3D array with the eroded droplet image in each layer
        # SAMPLE_ID is a string with the asmple ID/name assigned in the spreadsheet 
        # drops_IDs is an array with each oth the droplet numbers in it 
        # drops is a 3D array where each layer is zero everywhere and except the location of a single droplet (comes from cv2)
        # Water is the watershed result
        # img_erode is a 2D array with the eroded watershed result 
        # crop_image_erode and PIL_crop_image_erode are images with the watershed droplets super imposed on the cropped image 
        # crop_image_erode is estracted by CV2 format and PIL_crop_image_erode is extracted by PIL
        # img_name is the name of the image used for watershed 
        # Notes are any experimental Notes 
        # PTL is the pixel to lenght conversion factor 
        
        self.ID = sample_ID
        self.drops = drops
        self.drop_IDs = drop_IDs
        self.Number_of_drops = Number_of_drops
        self.img_erode = img_erode
        self.cv2_eroded = crop_image_erode
        self.PIL_eroded = PIL_crop_image_erode
        self.water = water 
        self.img_name = img_name
        self.Notes = Notes
        self.PTL = PTL
        self.crop_img_pil = crop_img_pil
        self.crop_params = crop_params 
    
    def save(self):
        # save all the sample information in a pickle format 
        pickle.dump(self.ID, open('./sample-data/sample_ID','wb'))
        pickle.dump(self.drops, open('./sample-data/drops','wb'))
        pickle.dump(self.drop_IDs, open('./sample-data/drop_IDs','wb'))
        pickle.dump(self.Number_of_drops, open('./sample-data/Number_of_drops','wb'))
        pickle.dump(self.img_erode, open('./sample-data/img_erode','wb'))
        pickle.dump(self.cv2_eroded, open('./sample-data/crop_image_erode','wb'))
        pickle.dump(self.PIL_eroded, open('./sample-data/PIL_crop_image_erode','wb'))
        pickle.dump(self.water, open('./sample-data/water','wb'))
        pickle.dump(self.img_name, open('./sample-data/img_name','wb'))
        pickle.dump(self.Notes, open('./sample-data/Notes','wb'))
        pickle.dump(self.PTL, open('./sample-data/PTL','wb'))
        print('Saved sample data')
        
   # create a function that makes a sample class and saves the sample droplet information in a pickle format 
def output_for_labeling(crop_params, crop_img_pil, imgn_erode, sample_ID, water,crop_image_erode, PIL_crop_image_erode, img_name, Notes, PTL):
    # Create a Sample class
    # Create Drop_IDs List 
    num_drops = np.unique(imgn_erode)
    if np.size(np.where(num_drops==0)) != 0:
        num_drops = np.extract(num_drops!=0, num_drops)
    drop_IDs = num_drops 
    Number_of_drops = np.size(drop_IDs)
    # Create the 3D array drops where each layer is zero everywhere and except the location of a single droplet
    drops_rows = np.size(imgn_erode, 0)
    drops_col = np.size(imgn_erode, 1)
    drops = np.zeros((Number_of_drops,drops_rows,drops_col))
    for i in range(Number_of_drops):
        n = drop_IDs[i]
        drop_loc = np.copy(imgn_erode)
        drop_loc[drop_loc!=n] = 0
        drops[i,:,:] = drop_loc
    sample = Sample(crop_params, crop_img_pil,sample_ID, drop_IDs, drops, Number_of_drops, water, imgn_erode, crop_image_erode, PIL_crop_image_erode, img_name, Notes, PTL)
    # Save all information in a pickle format
    sample.save()
    return sample   

def cc_crop(x_vals,y_vals, x_space,y_space,img,spacex,spacey,offsetx,offsety, type_card):
    # x_vals is a 1x2 array with min and max x value of the whole color card crop box 
    # y_vals is the same 
    # x_space is spacing in the x -direction between boxes 
    # y_vals is spacing in the y direction between boxes 
    if (type_card == 'cc'):
        r = 4
        c = 6
    elif (type_card == 'xrite'):
        r = 4
        c = 6
    xvals = np.zeros((r,c))
    yvals = xvals
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    width = x_space
    height = y_space
    ll_x_a = []
    ll_y_a = []
    for i in range(c):
        for j in range(r):
            ll_y = math.ceil(min(y_vals)+ (j)*(y_space + spacey)+ offsety*i)
            ll_x = math.ceil(min(x_vals)+(i)*(x_space + spacex)+offsetx*j)
            ax.add_patch(Rectangle((ll_x, ll_y), width, height, edgecolor='red', fill=False))
            ll_x_a.append(ll_x)
            ll_y_a.append(ll_y)
    plt.title(f'Croped Colors of {type_card} Color Card')
    plt.show()
    return ll_x_a, ll_y_a
     
# Function to extract Color Card RGB data from 1 image  
def CC_RGB(cc_ll_x, cc_ll_y, width,height,img_path):
    # set Crop indices 
    R_drop_cc = []
    G_drop_cc=[]
    B_drop_cc = []
    R_hi_drop_cc = []
    G_hi_drop_cc = []
    B_hi_drop_cc = []
    R_lo_drop_cc = []
    G_lo_drop_cc = []
    B_lo_drop_cc = []
    for j in range(24):
        crop_params = {
        'theta': 0,
        'x1': cc_ll_x[j],
        'x2': cc_ll_x[j]+width,
        'y1': cc_ll_y[j], 
        'y2': cc_ll_y[j]+height
        }
        crop_image = RRC(img_path, crop_params,'pil')
        R_channel = np.array(crop_image, dtype=np.uint8)[:,:,0]
        G_channel = np.array(crop_image, dtype=np.uint8)[:,:,1]
        B_channel = np.array(crop_image, dtype=np.uint8)[:,:,2]
        R_drop_cc.append([round(np.mean(R_channel))])
        G_drop_cc.append([round(np.mean(G_channel))])
        B_drop_cc.append([round(np.mean(B_channel))])
        R_hi_drop_cc.append([np.percentile(np.sort(R_channel, axis = None), 95)])
        G_hi_drop_cc.append([np.percentile(np.sort(G_channel, axis = None), 95)])
        B_hi_drop_cc.append([np.percentile(np.sort(B_channel, axis = None), 95)])
        R_lo_drop_cc.append([np.percentile(np.sort(R_channel, axis = None), 5)])
        G_lo_drop_cc.append([np.percentile(np.sort(G_channel, axis = None), 5)])
        B_lo_drop_cc.append([np.percentile(np.sort(B_channel, axis = None), 5)])
        # For each square create RGB series over time, the matrices returned contain nested array where R_drop[0] for 
        # example has all values over time of Red channel of the first Color Card color 
    
    return R_drop_cc,G_drop_cc,B_drop_cc, R_hi_drop_cc,G_hi_drop_cc,B_hi_drop_cc, R_lo_drop_cc,G_lo_drop_cc,B_lo_drop_cc
   
# taken also from web https://stackoverflow.com/questions/52767317/how-to-convert-rgb-image-pixels-to-lab 
def convert_LAB_RGB(data, to_space, from_space):
    # Input:
    # - data: a np array with dimensions (n_samples, {optional
    #   dimension: n_times}, n_color_coordinates=3) (e.g., a direct output of
    #   'rgb_extractor()' or 'rgb_extractor_Xrite_CC()')
    # - from_space: choose either 'RGB' or 'Lab'
    # - to_space: choose either 'RGB' or 'Lab'
    # Output:
    # - converted: a np array with the same dimensions than in the input

    n_d = data.ndim
    if n_d == 2:
        data = np.expand_dims(data, 1)
    elif n_d != 3:
        raise Exception('Faulty number of dimensions in the input!')
    if (from_space == 'RGB') and (to_space == 'LAB'):
        # Values from rgb_extractor() are [0,255] so let's normalize.
        data = data/255
        # Transform to color objects (either sRGBColor or LabColor).
        data_objects = np.vectorize(lambda x,y,z: sRGBColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        # Target color space
        color_space = matlib.repmat(LabColor, *data_objects.shape)
        # Transform from original space to new space 
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        # We got a matrix of color objects. Let's transform to a 3D matrix of floats.
        converted = np.transpose(np.vectorize(lambda x: (x.lab_l, x.lab_a, x.lab_b))(
            converted_objects), (1,2,0))

    elif (from_space == 'LAB') and (to_space == 'RGB'):
        data_objects = np.vectorize(lambda x,y,z: LabColor(x,y,z))(
            data[:,:,0], data[:,:,1], data[:,:,2])
        color_space = matlib.repmat(sRGBColor, *data_objects.shape)
        converted_objects = np.vectorize(lambda x,y: convert_color(x,y))(
            data_objects, color_space)
        converted = np.transpose(np.vectorize(lambda x: (x.rgb_r, x.rgb_g, x.rgb_b))(
            converted_objects), (1,2,0))
        # Colormath library interprets rgb in [0,1] and we want [0,255] so let's
        # normalize to [0,255].
        converted = converted*255
    else:
        raise Exception('The given input space conversions have not been implemented.')
    if n_d == 2:
        converted = np.squeeze(converted)
    return converted

# Function to color calibrate the results 
# Taken from previous color calibration code  in 2023
def color_calibration(drop,ll_x, ll_y, width,heigth, file, type_card,XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo,reference_CC_lab):
    # print('In Col Cal')

    # Let's extract the rgb colors from our color passport picture.
    # Convert the color card into LAB values 
    img_array = np.array(np.stack((XR,XG,XB), axis= 2),dtype = np.uint8)
    RGB_reshaped = img_array.reshape(24,3)
    CC_lab = convert_LAB_RGB(RGB_reshaped, 'LAB', 'RGB')
    
    # print('This is X_lab')
    # print(CC_lab)
    # Convert droplet RGBs into LAB values 
    sample_lab = convert_LAB_RGB(drop, 'LAB', 'RGB')
    # Reorganize the reference to match the colors orders in the color card
    
    if (type_card == 'cc'):
        order = [23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
        ref_copy = np.copy(reference_CC_lab)
        for h in range(24):
            ref_copy[h] = reference_CC_lab[order[h],:]
        reference_CC_lab = np.copy(ref_copy)
    
    # X_hi_lab = X_hi_lab[order]
    # X_lo_lab =  X_lo_lab[order]
    # sample_lab = sample_lab[order]
    
    ###########################
    # Color calibration starts.
    
    # Number of color patches in the color chart.
    N_patches = 24
    
    # Let's create the weight matrix for color calibration using 3D thin plate
    # spline.

    # Data points of our color chart in the original space.
    P = np.concatenate((np.ones((N_patches,1)), CC_lab), axis=1)
    # Data points of our color chart in the transformed space.
    V = reference_CC_lab
    # Shape distortion matrix, K
    K = np.zeros((N_patches,N_patches))
    for i in range(N_patches):
        for j in range(N_patches):
            if i != j:
                r_ij = np.sqrt((P[j,0+1]-P[i,0+1])**2 +
                               (P[j,1+1]-P[i,1+1])**2 +
                               (P[j,2+1]-P[i,2+1])**2)
                U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                K[i,j] = U_ij
    # Linear and non-linear weights WA:
    numerator = np.concatenate((V, np.zeros((4,3))), axis=0)
    denominator = np.concatenate((K,P), axis=1)
    denominator = np.concatenate((denominator,
                                  np.concatenate((np.transpose(P),
                                                  np.zeros((4,4))),axis=1)), axis=0)
    WA = np.matmul(np.linalg.pinv(denominator), numerator)

    # Checking if went ok. We should get the same result than in V (except for
    # the 4 bottom rows)
    CC_lab_double_transformation = np.matmul(denominator,WA)
    #print('Color chart patches in reference Lab:', reference_CC_lab,
    #      'Color chart patches transformed to color calibrated space and back - this should be the same than above apart from the last 4 rows',
    #      CC_lab_double_transformation, 'subtracted: ', reference_CC_lab-CC_lab_double_transformation[0:-4,:])
    # print('Checking if color transformation is successful - all values here should be near zero:/n', reference_CC_lab-CC_lab_double_transformation[0:-4,:])
    
    # Let's perform color calibration for the sample points!
    N_samples = sample_lab.shape[0]
    N_times = sample_lab.shape[1]
    sample_lab_cal = np.zeros((N_samples,N_times+4,3))
    # We are recalculating P and K for each sample, but using the WA calculated above.
    for s in range(N_samples):
        # Data points of color chart in the original space.
        P_new = np.concatenate((np.ones((N_times,1)), sample_lab[s,:,:]), axis=1)
        K_new = np.zeros((N_times,N_patches))
        # For each time point (i.e., picture):
        for i in range(N_times):
            # For each color patch in Xrite color chart:
            for j in range(N_patches):
                #if i != j:
                r_ij = np.sqrt((P_new[i,0+1]-P[j,0+1])**2 + (P_new[i,1+1]-P[j,1+1])**2 + (P_new[i,2+1]-P[j,2+1])**2)
                U_ij = 2* (r_ij**2)* np.log(r_ij + 10**(-20))
                K_new[i,j] = U_ij
        dennom = np.concatenate((K_new,P_new),axis=1)
        denden = np.concatenate((np.transpose(P), np.zeros((4,4))), axis=1)
        sample_lab_cal[s,:,:] = np.matmul(np.concatenate((dennom, denden), axis=0), WA)
    # Remove zeros, i.e., the last four rows from the third dimension.
    sample_lab_cal = sample_lab_cal[:,0:-4,:]
    ################################
    # Color calibration is done now.
    
    # Let's transform back to rgb.
    sample_rgb_cal = convert_LAB_RGB(sample_lab_cal, 'RGB', 'LAB')
    #sample_rgb_cal = convert_LAB_RGB(sample_lab_cal, 'RGB', 'N')
    
    # X_hi_lab = convert_LAB_RGB([XR_hi,XG_hi,XB_hi], 'LAB')
    # X_lo_lab = convert_LAB_RGB([XR_lo, XG_lo, XB_lo],'LAB')
    
    
    # Let's return both lab and rgb calibrated values.
    return sample_rgb_cal, sample_lab_cal

def Results(type_card, cc_ll_x, cc_ll_y, xc_ll_x, xc_ll_y, int_ll_x, int_ll_y, wid, hei, wid_x,hei_x, wid_int, hei_int, crop_params,Time_Steps,start_img,cut):
    # This function runs color calibration on the droplets over all files based on given calibration scheme and outputs the results 
    # The calibration scheme is 'xrite' calibrate relative to the xrite picture 
    # Code structure: For each image extract all calibrated droplet colors  
    # Inputs:
    # type_card ('xrite') = The type of color checker card used 
    # a_ll_x, a_ll_y = The crop box lower left and lower right indices for
    #                  - printout of the color chart (a=cc)
    #                  - Xrite Colour Checker (a=xc)
    #                  - intermediary color chart int he same image as the xrite color chart (a = int)
    # wid_b, hei_b = The width and height of the crop boxes for 
    #                  - printout of the color chart (b= ' ')
    #                  - Xrite Colour Checker (b=x)
    #                  - intermediary color chart int he same image as the xrite color chart (b = int)
    # crop params = Crop parameters of the reference image  
    # Time_steps = Time step between each image defined in the sample_names spreadsheet
    # start_img = The index of the starting image in the image time series (removes the first couple of frames of the timeseries data)
    # Outputs:
    # drops_rgb_cal_x, drops_lab_cal_x = RGB and LAB colour images of the color clibrated images
    #                                    where drops_rgb_cal_x[x][0] = the color clibrated image at time step x from the start image 
    # cut = The number of the jump between for your color sample points. Tf cut = 1 you extract the color data of every image (not recommended for processing speeds) 
    
    # Extract times, image names, and number of droplets in the sample 
    num_pics = pickle.load(open('./sample-data/num_pics','rb'))
    names = pickle.load(open('./sample-data/name','rb'))
    time = pickle.load(open('./sample-data/time','rb'))
    
    drops_rgb_cal_x = []
    drops_lab_cal_x = []
    reference_CC_lab = np.array([[37.54,14.37,14.92],[62.73,35.83,56.5],[28.37,15.42,-49.8],
                                    [95.19,-1.03,2.93],[64.66,19.27,17.5],[39.43,10.75,-45.17],
                                    [54.38,-39.72,32.27],[81.29,-0.57,0.44],[49.32,-3.82,-22.54],
                                    [50.57,48.64,16.67],[42.43,51.05,28.62],[66.89,-0.75,-0.06],
                                    [43.46,-12.74,22.72],[30.1,22.54,-20.87],[81.8,2.67,80.41],
                                    [50.76,-0.13,0.14],[54.94,9.61,-24.79],[71.77,-24.13,58.19],
                                    [50.63,51.28,-14.12],[35.63,-0.46,-0.48],[70.48,-32.26,-0.37],
                                    [71.51,18.24,67.37],[49.57,-29.71,-28.32],[20.64,0.07,-0.46]])
    if (type_card == 'xrite'):
        file = './sample-data/images/xrite.jpg'
        XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo = CC_RGB(xc_ll_x, xc_ll_y, wid,hei,file)
        # Reference data is in different order (from upper left to lower left, upper
        # 2nd left to lower 2nd left...). 
        width = wid_x
        height = hei_x
        c_ll_x = xc_ll_x
        c_ll_y = xc_ll_y
    elif (type_card == 'cc'):
        # Calibrate the printed out color chart and use the results as the reference_CC_lab 
        file = './sample-data/images/xrite.jpg'
        image_pil = Image.open(file, 'r')
        image_cc = np.array(image_pil, dtype=np.uint8)
        XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo = CC_RGB(xc_ll_x, xc_ll_y, wid_x,hei_x,file)
        drop_rgb_cal, drop_lab_cal = color_calibration(image_cc, xc_ll_x, xc_ll_y, wid_x,hei_x, file, 'xrite',XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo, reference_CC_lab)
        print(drop_rgb_cal)
        drop_rgb_cal = Image.fromarray(drop_rgb_cal, 'RGB')
        drop_rgb_cal = drop_rgb_cal.save('int_drop.jpg')
        XRi,XGi,XBi, XR_hii,XG_hii,XB_hii, XR_loi,XG_loi,XB_loi = CC_RGB(int_ll_x, int_ll_y, wid_int, hei_int, './int_drop.jpg') 
        img_arrayi = np.array(np.stack((XRi,XGi,XBi), axis= 2),dtype = np.uint8)
        reference_CC_lab = img_arrayi.reshape(24,3)
        width = wid
        height = hei
        c_ll_x = cc_ll_x
        c_ll_y = cc_ll_y
    for k in range(start_img-1, num_pics):
        # get the cropped PIL image 
        # Crop the image 
        file_k = f"./sample-data/images/{names[k]}"
        crop_k = RRC(file_k, crop_params, 'pil')
        drop = np.copy(crop_k)
        # replace all pixels where the droplets are not present with zero 
        # use the eroded image with indexed droplets (img-erode ) as the index for locating droplets 
        # drop[np.where(sample.img_erode == 0)] = 0
        # Run color calibration on the image
        # Set the calibration file based on the scheme 
            # the xrite file must alwasy be called this 
        if(type_card == 'cc'):
            file = f'./sample-data/images/{names[k]}'
            XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo = CC_RGB(cc_ll_x, cc_ll_y, wid,hei,file)
            # drop is an image array with just the droplets and not background 
        drop_rgb_cal, drop_lab_cal = color_calibration(drop, c_ll_x, c_ll_y, width,height, file, type_card,XR,XG,XB, XR_hi,XG_hi,XB_hi, XR_lo,XG_lo,XB_lo, reference_CC_lab)
        print(f'Image {k-1}/{int(num_pics-start_img)} Stability Computation Complete . . .')
        # For each image you are adding calibrated RGB values for the droplets. 
        drops_rgb_cal_x.append([drop_rgb_cal])
        drops_lab_cal_x.append([drop_lab_cal])

    print('DONE Results')
    return drops_rgb_cal_x, drops_lab_cal_x