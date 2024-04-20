import os
import random
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

def translate_image(img, trans_x, trans_y):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return cv.warpAffine(img, M, (cols, rows))


def shear_image(img, shear_x, shear_y):
    rows, cols = img.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv.warpAffine(img, M, (cols, rows))


def grey_scale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def color_jittering(img, change):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = cv.add(h, int(change))
    s = cv.add(s, int(change))
    v = cv.add(v, int(change))
    hsv = cv.merge((h, s, v))
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def noise_addition(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = cv.add(img.astype(np.float32), noise.astype(np.float32))
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def lighting_conditions(img, alpha, beta):
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

def vignetting(img, scale):
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    mask = np.zeros((rows, cols, 3), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
            mask[i, j] = np.clip((1 - scale * dist / (np.sqrt(rows ** 2 + cols ** 2) / 2)), 0, 1)
    return cv.multiply(img.astype(np.float32), mask).astype(np.uint8)

def blurring(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    blurred_img = cv.filter2D(img, -1, kernel)
    return blurred_img

def sharpening(img, kernel):
    sharpened_img = cv.filter2D(img, -1, kernel)
    return sharpened_img


def apply_geometric_transformations(img, outputPath, augmentation_factor):
    # Adjust the step ranges based on augmentation factor
    angles = np.linspace(-45, 45, augmentation_factor)
    translations = np.linspace(-10, 10, augmentation_factor)
    shears = np.linspace(-0.2, 0.2, augmentation_factor)
    
    for i in range(augmentation_factor):
        # Rotate
        rotated_img = rotate_image(img, angles[i % len(angles)])
        save_augmented_image(rotated_img, outputPath, f"rotated_{i+1}")
        # Translate
        trans_img = translate_image(img, translations[i % len(translations)], translations[(i * 2) % len(translations)])
        save_augmented_image(trans_img, outputPath, f"translated_{i+1}")
        # Shear
        shear_img = shear_image(img, shears[i % len(shears)], shears[(i * 3) % len(shears)])
        save_augmented_image(shear_img, outputPath, f"sheared_{i+1}")
        # Flip (only once since it's binary)
        if i == 0:
            flipped_img = flip_image(img)
            save_augmented_image(flipped_img, outputPath, "flipped")  
        
        
def apply_photometric_transformations(img, outputPath, augmentation_factor):
    kernel_sizes = [3, 5, 7, 9]  # Define possible kernel sizes for blurring
    sharpening_kernels = [  # Define different levels of sharpening using different kernels
        np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),  # Basic sharpening
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Light sharpening
        np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0,  # More complex
        np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])  # High-frequency emphasis
    ]

    # Process other transformations as usual
    grey_img = grey_scale(img)
    save_augmented_image(grey_img, outputPath, "grey", is_gray=True)

    # Dynamic ranges for other transformations
    change_ranges = np.linspace(-15, 15, augmentation_factor)
    sigma_ranges  = np.linspace(0, 10, augmentation_factor)
    alpha_ranges  = np.linspace(0.9, 1.2, augmentation_factor)
    beta_ranges   = np.linspace(-50, 50, augmentation_factor)
    scale_ranges  = np.linspace(0.3, 0.7, augmentation_factor)

    for i in range(augmentation_factor):
        # Apply transformations
        jittered_img  = color_jittering(img, change_ranges[i % len(change_ranges)])
        noisy_img     = noise_addition(img, sigma_ranges[i % len(sigma_ranges)])
        lighted_img   = lighting_conditions(img, alpha_ranges[i % len(alpha_ranges)], beta_ranges[i % len(beta_ranges)])
        vignetted_img = vignetting(img, scale_ranges[i % len(scale_ranges)])
        if i < 4: # only 4 kernel sizes
            blurred_img   = blurring(img, kernel_sizes[i % len(kernel_sizes)])
            sharpened_img = sharpening(img, sharpening_kernels[i % len(sharpening_kernels)])
            save_augmented_image(blurred_img, outputPath, f"blurred_{i+1}")
            save_augmented_image(sharpened_img, outputPath, f"sharpened_{i+1}")

        # Save the augmented images
        save_augmented_image(jittered_img, outputPath, f"jittered_{i+1}")
        save_augmented_image(noisy_img, outputPath, f"noisy_{i+1}")
        save_augmented_image(lighted_img, outputPath, f"lighted_{i+1}")
        save_augmented_image(vignetted_img, outputPath, f"vignetted_{i+1}")

def augment_image(file_path, output_dir, augmentation_factor=1):
    img = cv.imread(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    target_path = os.path.join(output_dir, base_filename)
    apply_geometric_transformations(img, target_path, augmentation_factor)
    apply_photometric_transformations(img, target_path, augmentation_factor)

    
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

# --
def add_noise2(audio, sr, noise_level=0.01):
    noise_amp = noise_level * np.random.uniform() * np.amax(audio)
    return audio + noise_amp * np.random.normal(size=audio.shape[0])

def add_noise3(audio, sr, noise_level=0.02):
    noise_amp = noise_level * np.random.uniform() * np.amax(audio)
    return audio + noise_amp * np.random.normal(size=audio.shape[0])

def augment_audio(file_path, output_dir, sr=16000):
    
    # Load the original audio file
    audio, sr = librosa.load(file_path, sr=sr)
    
    if len(audio) < 2048:
        print(f"Short audio file: {file_path}")
        return
    
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
                  
def augment_and_save_files(input_dir, output_dir, file_extension, augment_function, augmentation_factor=1):
    files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
    for file in tqdm(files, desc=f"Augmenting {file_extension} files in {os.path.basename(input_dir)}"):
        file_path = os.path.join(input_dir, file)
        if file_extension == ".wav":
            augment_function(file_path, output_dir)  # Call without augmentation_factor for audio
        else:
            augment_function(file_path, output_dir, augmentation_factor) # Call with augmentation_factor for images



if __name__ == "__main__":
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
            augmentation_factor = 18 if _type == "target" else 2  # Increase for target class
            process_original_audio(input_dir, output_dir)
            copy_original_images(input_dir, output_dir)
            augment_and_save_files(input_dir, output_dir, ".png", augment_image, augmentation_factor)
            augment_and_save_files(input_dir, output_dir, ".wav", augment_audio, augmentation_factor)
    print("Data augmentation completed.")