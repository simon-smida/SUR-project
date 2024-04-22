import os
import numpy as np
from tqdm import tqdm
import shutil

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
        
def augment_image(file_path, output_dir):
    img = cv.imread(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    target_path = os.path.join(output_dir, base_filename)
    apply_geometric_transformations(img, target_path)
    apply_photometric_transformations(img, target_path)
    
def save_augmented_image(img, output_base_path, augmentation_type, is_gray=False):
    """Saves the augmented image file with an informative filename."""
    new_filename = f"{output_base_path}_{augmentation_type}.png"
    if is_gray:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert back to BGR for saving
    cv.imwrite(new_filename, img)


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

def add_noise2(audio, sr, noise_level=0.01):
    noise_amp = noise_level * np.random.uniform() * np.amax(audio)
    return audio + noise_amp * np.random.normal(size=audio.shape[0])

def add_noise3(audio, sr, noise_level=0.02):
    noise_amp = noise_level * np.random.uniform() * np.amax(audio)
    return audio + noise_amp * np.random.normal(size=audio.shape[0])

def augment_audio(file_path, output_dir, sr=16000):
    # Load the original audio file
    audio, sr = librosa.load(file_path, sr=sr)
    
    # Remove first 2s (to eliminate any unwanted noise at the start of the audio)
    audio_trimmed = trim_audio(audio, sr, 2)

    # Base filename for saving the augmented files
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Ensure the correct output path is constructed for the augmented files
    output_base_path = os.path.join(output_dir, base_filename)

    # Generate and save augmented audio versions
    audio_noise = add_noise(audio_trimmed, sr)
    save_augmented_audio(audio_noise, sr, output_base_path, 'noise')

    audio_shifted = time_shift(audio_trimmed, sr)
    save_augmented_audio(audio_shifted, sr, output_base_path, 'shift')
    
    audio_speed_pitch = change_speed_pitch(audio_trimmed, sr)
    save_augmented_audio(audio_speed_pitch, sr, output_base_path, 'speed_pitch')
    
    audio_volume = adjust_volume(audio_trimmed, sr)
    save_augmented_audio(audio_volume, sr, output_base_path, 'vol')
    
    # ---
    audio_noise = add_noise2(audio_trimmed, sr)
    save_augmented_audio(audio_noise, sr, output_base_path, 'noise2')

    audio_noise = add_noise3(audio_trimmed, sr)
    save_augmented_audio(audio_noise, sr, output_base_path, 'noise3')

def save_augmented_audio(audio, sr, output_base_path, augmentation_type):
    """Saves the augmented audio file with an informative filename."""
    new_filename = f"{output_base_path}_{augmentation_type}.wav"
    sf.write(new_filename, audio, sr) 

def process_original_audio(input_dir, output_dir, sr=16000):
    """ Processes each audio file to remove the first 2 seconds and saves it to the output directory. """
    for item in os.listdir(input_dir):
        src = os.path.join(input_dir, item)
        dest = os.path.join(output_dir, item)
        if os.path.isdir(src):
            os.makedirs(dest, exist_ok=True)
            process_original_audio(src, dest)  # Recursive call for subdirectories
        elif src.endswith('.wav'):
            audio, sr = librosa.load(src, sr=sr)
            trimmed_audio = trim_audio(audio, sr, 2)
            sf.write(dest, trimmed_audio, sr)

def copy_original_images(input_dir, output_dir):
    """ Copies all image files from input_dir to output_dir. """
    for item in os.listdir(input_dir):
        src = os.path.join(input_dir, item)
        dest = os.path.join(output_dir, item)
        if os.path.isdir(src):
            os.makedirs(dest, exist_ok=True)
            copy_original_images(src, dest)  # Recursive call for subdirectories
        elif src.endswith('.png'):
            shutil.copy2(src, dest)
                  
def augment_and_save_files(input_dir, output_dir, file_extension, augment_function):
    """ Handles the augmentation and saving of augmented files with progress indication using tqdm. """
    files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
    for file in tqdm(files, desc=f"Augmenting {file_extension} files in {os.path.basename(input_dir)}"):
        file_path = os.path.join(input_dir, file)
        augment_function(file_path, output_dir)


if __name__ == "__main__":
    
    # Define the base directories and data categories 
    input_base_dir = "data"
    output_base_dir = "augmented_data"
    categories = ["train", "dev"]
    types = ["non_target", "target"]

    # Augment and save files for each category and type
    for category in categories:
        for _type in types:
            input_dir = os.path.join(input_base_dir, category, f"{_type}_{category}")
            output_dir = os.path.join(output_base_dir, category, f"{_type}_{category}")
            os.makedirs(output_dir, exist_ok=True)
            # Process and save trimmed original audio files
            process_original_audio(input_dir, output_dir)
            # Copy original image files
            copy_original_images(input_dir, output_dir)
            # Augment and save image and audio files
            augment_and_save_files(input_dir, output_dir, ".png", augment_image)
            augment_and_save_files(input_dir, output_dir, ".wav", augment_audio)