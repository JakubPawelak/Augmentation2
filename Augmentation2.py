import cv2
import os
import shutil
import random
import numpy as np

#for i in range (51):
#    os.mkdir(str(i))


#if text.startswith("1"):

##SORT PICTURE##
# listtxt = os.listdir("RamkiAll")
#
# for item in listtxt:
#     with open("RamkiAll/"+item, "r") as file:
#         data = file.read()
#     temp = data.split(" ",1)
#     foldernumber = temp[0]
#     str1="Zdjecia/"
#     jpg_item = item[:len(item)-3]
#     jpg_item = jpg_item+"jpg"
#     current=(str1+jpg_item)
#     newpath=(foldernumber+"/"+jpg_item)
#     try:
#         os.replace(current,newpath)
#     except OSError as error:
#         print(item)

##CHECK NUMBER OF ELEMENTS##
# temp=0
# for i in range(52):
#     listdr=os.listdir(str(i))
#     print(str(i)+": "+str(len(listdr)))
#     if temp<len(listdr):
#         temp=len(listdr)
# print("max: "+str(temp))
#
##IMAGE MULTIPLICATION##
# for i in range(52):
#     listdr=os.listdir(str(i))
#     if len(listdr) < 1:
#         continue
#     else:
#         mitem=temp-len(listdr)
#         for m in range(mitem):
#             r=random.randrange(len(listdr))
#             src=(str(i)+"/"+(listdr[r]))
#             dsc=src[:len(src)-4]
#             dsc=str(dsc)+str(m)+".jpg"
#             shutil.copy(src,dsc)

##AUGMENTATION##
def brightness(img):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(img)
        img=cv2.imread(path)
        low=0.8
        high=1.4
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path,img)

def rotation(img):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(img)
        img=cv2.imread(path)
        angle=4
        angle = int(random.uniform(-angle, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(path,img)


def add_light(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        gamma=random.uniform(0.7,3.0)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        image=cv2.LUT(image, table)
        if gamma>=1:
            cv2.imwrite(path,image)
        else:
            cv2.imwrite(path,image)

def saturation_image(image):
    probability = random.randrange(0, 100)
    if probability <= 60:
        path=str(image)
        image=cv2.imread(path)
        saturation=random.uniform(1,50)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = image[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image[:, :, 2] = v

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path,image)

def gausian_blur(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        blur=random.uniform(1,5)
        image = cv2.GaussianBlur(image,(5,5),blur)
        cv2.imwrite(path,image)

def erosion_image(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        shift=random.randrange(1,3)
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.erode(image,kernel,iterations = 1)
        cv2.imwrite(path,image)

def dilation_image(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        shift=random.randrange(1,3)
        kernel = np.ones((shift, shift), np.uint8)
        image = cv2.dilate(image,kernel,iterations = 1)
        cv2.imwrite(path,image)

def opening_image(image):
    probability = random.randrange(0, 100)
    if probability >= 30:
        path=str(image)
        image=cv2.imread(path)
        shift=random.randrange(1,3)
        kernel = np.ones((shift, shift), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(path,image)

def closing_image(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        shift=random.randrange(1,3)
        kernel = np.ones((shift, shift), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(path,image)

def sharpen_image(image):
    probability = random.randrange(0, 100)
    if probability <= 30:
        path=str(image)
        image=cv2.imread(path)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(path,image)



def sp_noise(image):
    probability = random.randrange(0, 100)
    if probability <= 60:
        path=str(image)
        image=cv2.imread(path)
        prob=random.uniform(0.01,0.02)
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        cv2.imwrite(path,output)

for i in range(52):
    listdr=os.listdir(str(i))
    for u in range(len(listdr)):
        if len(listdr) >1:
            brightness(str(i)+"/"+listdr[u])
            rotation(str(i)+"/"+listdr[u])
            add_light(str(i)+"/"+listdr[u])
            saturation_image(str(i)+"/"+listdr[u])
            gausian_blur(str(i)+"/"+listdr[u])
            erosion_image(str(i)+"/"+listdr[u])
            dilation_image(str(i)+"/"+listdr[u])
            opening_image(str(i)+"/"+listdr[u])
            closing_image(str(i)+"/"+listdr[u])
            sharpen_image(str(i)+"/"+listdr[u])
            sp_noise(str(i)+"/"+listdr[u])

##MAKE TXT FILES##
# for i in range(52):
#     listdr=os.listdir(str(i))
#     for u in range(len(listdr)):
#         name=listdr[u]
#         name = name[:len(name)-3]
#         name = name+"txt"
#         with open("txt/"+name, 'w') as f:
#             f.write(str(i)+" "+ "0.500000 0.500000 1.000000 1.000000")
