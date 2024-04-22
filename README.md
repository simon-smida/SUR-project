# SUR-project
The project assignment is [here](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2023-2024/SUR_projekt2023-2024.txt).

## Data
Project zip archive download: [SUR_projekt2023-2024.zip](https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2023-2024/SUR_projekt2023-2024.zip)

To work with this project, make sure you have the correct directory structure and relevant files (same as in this repository), as well as the project zip archive containing the data (above).

## Dependencies
1. Create virtual environment
```bash
python3 -m venv venv
```
2. Activate the environment
```bash
source ./venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Augmentation
If you don't already have the data and it is not the required directory structure format, download it from [Data section](#data) and do:
```bash
# Creates correct directory structure
bash organizeData.sh
```

From the main directory do:
```bash
# Creates augment_data folder containing augmented images and audio
python3 ./SRC/dataAugmentation.py
```

## Usage
### CNN model

**Training the [CNN model](./SRC/imageModel.py)** on the full dataset (train+dev):
```bash
python3 ./SRC/visualDetection.py
```

**Cross-validation**
```bash
python3 ./SRC/visualDetection.py --cross_valid
```

**Model evaluation** (requires evaluation data in `/data/eval` folder):
```bash
python3 ./SRC/visualDetection.py --eval <path_to_cnn_model>
```
