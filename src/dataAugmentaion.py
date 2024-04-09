import os
import cv2 as cv
import numpy as np


def rotate_image(filePNG):
    img = cv.imread(filePNG)
    rotations = [90, 180, 270]

    for angle in rotations:
        rows, cols = img.shape[:2]
        M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv.warpAffine(img, M, (cols, rows))
        #cv.imwrite(filePNG[:-4] + "_rotated" + str(angle) + ".png", rotated)
        cv.imshow("Rotated", rotated)
        cv.waitKey(0)


def flip_image(filePNG):
    img = cv.imread(filePNG)
    flipped = cv.flip(img, 1)
    #cv.imwrite(filePNG[:-4] + "_flipped.png", flipped)
    cv.imshow("Flipped", flipped)
    cv.waitKey(0)


def translate_image(filePNG):
    img = cv.imread(filePNG)
    rows, cols = img.shape[:2]
    translations = [(10, 10), (-10, -10), (10, -10), (-10, 10)]

    for translation in translations:
        M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated = cv.warpAffine(img, M, (cols, rows))
        #cv.imwrite(filePNG[:-4] + "_translated" + str(translation[0]) + str(translation[1]) + ".png", translated)
        cv.imshow("Translated", translated)
        cv.waitKey(0)

def shear_image(filePNG):
    img = cv.imread(filePNG)
    rows, cols = img.shape[:2]
    shearsX = [0.9, 1.1]
    shearsY = [0.1, -0.1]

    for shearX in shearsY:
        for shearY in shearsX:
            M = np.float32([[shearY, shearX, 0], [0, 1, 0]])
            sheared = cv.warpAffine(img, M, (cols, rows))
            #cv.imwrite(filePNG[:-4] + "_sheared" + str(shear) + ".png", sheared)
            cv.imshow("Sheared", sheared)
            cv.waitKey(0)


def geometric_transformations(filePNG):
    rotate_image(filePNG)
    flip_image(filePNG)
    translate_image(filePNG)
    shear_image(filePNG)


def grey_scale(filePNG):
    img = cv.imread(filePNG)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imwrite(filePNG[:-4] + "_grey.png", grey)
    cv.imshow("Grey", grey)
    cv.waitKey(0)

def color_jittering(filePNG):
    img = cv.imread(filePNG)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h += 10
    s += 10
    v += 10
    hsv = cv.merge((h, s, v))
    jittered = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #cv.imwrite(filePNG[:-4] + "_jittered.png", jittered)
    cv.imshow("Jittered", jittered)
    cv.waitKey(0)

def noise_addition(filePNG):
    img = cv.imread(filePNG)
    noise = np.random.normal(0, 0.5, img.shape).astype(np.uint8)
    noisy = cv.add(img, noise)
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)
    #cv.imwrite(filePNG[:-4] + "_noisy.png", noisy)
    cv.imshow("Noisy", noisy)
    cv.waitKey(0)

def lighting_conditions(filePNG):
    img = cv.imread(filePNG)
    alpha = 1.1
    beta = 40
    lighting = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    #cv.imwrite(filePNG[:-4] + "_lighting.png", lighting)
    cv.imshow("Lighting", lighting)
    cv.waitKey(0)

def vignetting(filePNG):
    img = cv.imread(filePNG)
    rows, cols = img.shape[:2]
    scale = 0.5
    center = (cols/2, rows/2)
    for i in range(rows):
        for j in range(cols):
            img[i, j] = img[i, j] * (1 - scale * np.sqrt((i - center[1])**2 + (j - center[0])**2)/(np.sqrt(rows**2 + cols**2)/2))
    #cv.imwrite(filePNG[:-4] + "_vignette.png", img)
    cv.imshow("Vignette", img)
    cv.waitKey(0)

def blurring(filePNG):
    img = cv.imread(filePNG)
    kernel = np.ones((2, 2), np.float32)/4
    blurred = cv.filter2D(img, -1, kernel)
    #cv.imwrite(filePNG[:-4] + "_blurred.png", blurred)
    cv.imshow("Blurred", blurred)
    cv.waitKey(0)

def photometric_transformations(filePNG):
    grey_scale(filePNG)
    color_jittering(filePNG)
    noise_addition(filePNG)
    lighting_conditions(filePNG)
    vignetting(filePNG)
    blurring(filePNG)


if __name__ == "__main__":

    # 1. Read all images from the folder
    # 2. Apply the data augmentation techniques
    # 3. Save the augmented images 

    currPath = os.getcwd() + "/data"
    augmentedPath = os.getcwd() + "/augmented_data"
    dirs = [item for item in os.listdir(currPath) if os.path.isdir(os.path.join(currPath, item))]
    filesPNG = []
    filesWAW = []

    for dir in dirs:
        for file in os.listdir(os.path.join(currPath, dir)):
            if file.endswith(".png"):
                filesPNG.append(os.path.join(currPath, dir, file))
            elif file.endswith(".waw"):
                filesWAW.append(os.path.join(currPath, dir, file))

    geometric_transformations(filesPNG[0])
    photometric_transformations(filesPNG[0])

    
