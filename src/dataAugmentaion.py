import os
import numpy as np
from tqdm import tqdm

# Images
import cv2 as cv

# Audio
import librosa
import soundfile as sf


# -- Image augmentation ---------------------------------------------------------------------

def rotate_image(img, angle):
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv.warpAffine(img, M, (cols, rows))
    return rotated

def flip_image(img):
    return cv.flip(img, 1)

def translate_image(img, translation):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    return cv.warpAffine(img, M, (cols, rows))

def shear_image(img, shearX=0.0, shearY=0.0):
    rows, cols = img.shape[:2]
    M = np.float32([[1, shearX, 0], [shearY, 1, 0]])
    return cv.warpAffine(img, M, (cols, rows))

def grey_scale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def color_jittering(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = cv.add(h, 10)
    s = cv.add(s, 10)
    v = cv.add(v, 10)
    hsv = cv.merge((h, s, v))
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def noise_addition(img):
    noise = np.random.normal(0, 0.5, img.shape)
    noisy = cv.add(img, noise.astype(np.uint8))
    return np.clip(noisy, 0, 255)

def lighting_conditions(img, alpha=1.1, beta=40):
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

def vignetting(img):
    rows, cols = img.shape[:2]
    scale = 0.5
    center = (cols/2, rows/2)
    mask = np.zeros((rows, cols, 3), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
            mask[i, j] = np.clip((1 - scale * dist / (np.sqrt(rows**2 + cols**2) / 2)), 0, 1)
    
    return cv.multiply(img.astype(np.float32), mask).astype(np.uint8)

def blurring(img):
    kernel = np.ones((2, 2), np.float32) / 4
    return cv.filter2D(img, -1, kernel)

def apply_geometric_transformations(img, outputPath):
    # Apply geometric transformations to the image and save the augmented images.
    
    # Rotate
    for angle in [90, 180, 270]:
        augmented_img = rotate_image(img, angle)
        save_augmented_image(augmented_img, outputPath, f"rotated{angle}")
    
    # Flip
    augmented_img = flip_image(img)
    save_augmented_image(augmented_img, outputPath, "flipped")
    
    # Translate
    for translation in [(10, 10), (-10, -10), (10, -10), (-10, 10)]:
        augmented_img = translate_image(img, translation)
        save_augmented_image(augmented_img, outputPath, f"translated_{translation[0]}_{translation[1]}")
    
    # Shear
    shear_factors = [(0.2, 0), (0, 0.2)]
    for i, (shear_x, shear_y) in enumerate(shear_factors):
        sheared_img = shear_image(img, shear_x, shear_y)
        save_augmented_image(sheared_img, outputPath, f"sheared_{i+1}")

def apply_photometric_transformations(img, outputPath):
    
    # Greyscale
    grey_img = grey_scale(img)
    save_augmented_image(grey_img, outputPath, "grey", is_gray=True)

    # Color Jittering
    jittered_img = color_jittering(img)
    save_augmented_image(jittered_img, outputPath, "jittered")
    
    # Noise Addition
    noisy_img = noise_addition(img)
    save_augmented_image(noisy_img, outputPath, "noisy")
    
    # Lighting Conditions
    light_img = lighting_conditions(img)
    save_augmented_image(light_img, outputPath, "light")
    
    # Vignetting
    vignette_img = vignetting(img)
    save_augmented_image(vignette_img, outputPath, "vignette")
    
    # Blurring
    blurred_img = blurring(img)
    save_augmented_image(blurred_img, outputPath, "blurred")
        
def augment_image(filePNG):
    img = cv.imread(filePNG)
    # Geometric Transformations
    apply_geometric_transformations(img, filePNG)
    # Photometric Transformations
    apply_photometric_transformations(img, filePNG)
 
def save_augmented_image(img, outputPath, augmentation_type, is_gray=False):
    #Saves the augmented image file with an informative filename.
    outputPath = outputPath[:-4]
    filepath = f"{outputPath}_{augmentation_type}.png"
    if is_gray:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert back to BGR for saving
    cv.imwrite(filepath, img)

# -- Audio augmentation ---------------------------------------------------------------------

def trim_audio(audio, sr, trim_sec):
    trim_samples = int(sr * trim_sec)
    return audio[trim_samples:]

def add_noise(audio, sr, noise_level=0.005):
    noise_amp = noise_level * np.random.uniform() * np.amax(audio)
    return audio + noise_amp*np.random.normal(size=audio.shape[0])

def time_shift(audio, sr, shift_max=100): # [ms]
    shift_amount = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift_amount)

def change_speed_pitch(audio, sr, speed_range=(0.9, 1.1)):
    speed_factor = np.random.uniform(*speed_range)
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def adjust_volume(audio, sr, volume_range=(0.5, 1.5)):
    dyn_change = np.random.uniform(*volume_range)
    return audio * dyn_change

def save_augmented_audio(audio, sr, outputPath, augmentation_type):
    """Saves the augmented audio file with an informative filename."""
    outputPath = outputPath[:-4]
    filepath = f"{outputPath}_{augmentation_type}.wav"
    sf.write(filepath, audio, sr)
    return filepath

def augment_audio(fileWAV, sr=16000):
    # Load the audio file
    audio, sr = librosa.load(fileWAV, sr=sr)

    # Remove first 2s (weird noise, mr.Burget's tip)
    audio_trimmed = trim_audio(audio, sr, 2)
    save_augmented_audio(audio_trimmed, sr, fileWAV, 'trim')

    # Audio augmentation (each augmentation is saved as a separate file)
    audio_noise = add_noise(audio_trimmed, sr)
    save_augmented_audio(audio_noise, sr, fileWAV, 'noise')

    audio_shifted = time_shift(audio_trimmed, sr)
    save_augmented_audio(audio_shifted, sr, fileWAV, 'shift')
    
    audio_speed_pitch = change_speed_pitch(audio_trimmed, sr)
    save_augmented_audio(audio_speed_pitch, sr, fileWAV, 'speed_pitch')
    
    audio_volume = adjust_volume(audio_trimmed, sr)
    save_augmented_audio(audio_volume, sr, fileWAV, 'vol')
    
    # low voice volume
    # audio_volume_low = adjust_volume(audio_trimmed, sr, volume_range=(0.1, 0.5))
    # save_augmented_audio(audio_volume_low, sr, outputPath, base_filename, 'vol_low')


if __name__ == "__main__":

    # 1. Read all images from the folder
    # 2. Apply the data augmentation techniques (image, audio)
    # 3. Save the augmented images 

    currPath = os.getcwd() + "/data/dev" 
    dirs = [item for item in os.listdir(currPath) if os.path.isdir(os.path.join(currPath, item))]
    
    currPath2 = os.getcwd() + "/data/train"
    dirs2 = [item for item in os.listdir(currPath2) if os.path.isdir(os.path.join(currPath2, item))]

    filesPNG = []
    filesWav = []

    # Read all files (images, audio)
    for dir in dirs:
        for file in os.listdir(os.path.join(currPath, dir)):
            if file.endswith("0.png"):  # file ending with 0.png is original image
                filesPNG.append(os.path.join(currPath, dir, file))
            elif file.endswith("0.wav"):
                filesWav.append(os.path.join(currPath, dir, file))

        # Read all files (images, audio)
    for dir in dirs2:
        for file in os.listdir(os.path.join(currPath2, dir)):
            if file.endswith("0.png"):  # file ending with 0.png is original image
                filesPNG.append(os.path.join(currPath2, dir, file))
            elif file.endswith("0.wav"):
                filesWav.append(os.path.join(currPath2, dir, file))
    
    # Image augmentation
    for filePNG in tqdm(filesPNG, desc="Augmenting images"):
        augment_image(filePNG)
        

    # Audio augmentation
    for fileWAV in tqdm(filesWav, desc="Augmenting audio"):
        augment_audio(fileWAV)
        