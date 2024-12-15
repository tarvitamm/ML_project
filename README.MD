# **ML Project: U-Net Image Segmentation**

## **Project Description**
This project implements a U-Net architecture for image segmentation tasks. It allows the user to slice images into tiles, train a U-Net model, and run inference to evaluate model predictions.

---

## **Prerequisites**

1. **Python 3.10**  
   Python 3.10 is required because:
   - TensorFlow 2.9.2 and other libraries are tested and compatible with Python 3.10.
   - Some dependencies may not work correctly with newer or older Python versions.

   **Download Python 3.10** from the [official Python website](https://www.python.org/downloads/release/python-3100/) or use a Python version manager like `pyenv`.

2. **Dependencies**  
   Install the required libraries from the `requirements.txt` file.

---

## **Setup Instructions**

### Step 1: Create a Virtual Environment with Python 3.10

**On Windows**: 
```bash
python3.10 -m venv myenv
myenv\Scripts\activate 
```
**On Linux/Mac**:  
```bash
python3.10 -m venv myenv
source myenv/bin/activate
```
---

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Trained model from Google Drive
- Copy a trained model to models/<trained_model> path
- Google Drive link: https://drive.google.com/drive/folders/1m8hvyMoGypXltUQIVVkFta0CRu_O-9yX?usp=sharing
---

## **Scripts Overview**

### **Possible Tile Sizes**
The supported tile sizes are **[256, 512]**.  
When running the scripts, you can choose a tile size from this list. 
### 1. **slice_images.py**
This script slices input images into tiles of a specified size and performs an 80/20 train-validation split.

**Usage**:  
```bash
python slice_images.py <tile_size>  
```
**Example**:
```bash  
python slice_images.py 512  
```
**Input**:  
- Training images from `images/X_train/`  
- Ground truth masks from `labels/y_train/`  

**Output**:  
- Sliced tiles stored as images and `.npy` files:  
   - `X_train_<tile_size>.npy`  
   - `y_train_<tile_size>.npy`  
   - `X_val_<tile_size>.npy`  
   - `y_val_<tile_size>.npy`  

---

### 2. **train_model.py**
This script trains a U-Net model using the sliced tiles.

**Usage**:  
```bash  
python train_model.py <image_size> 
```
**Example**:
```bash    
python train_model.py 512
```

**Input**:  
- Train data: `X_train_<image_size>.npy`, `y_train_<image_size>.npy`  
- Validation data: `X_val_<image_size>.npy`, `y_val_<image_size>.npy`  

**Output**:  
- Trained U-Net model saved as `models/unet_<image_size>.h5`  

---

### 3. **do_inference.py**
This script runs inference on the test images using a trained U-Net model.
For the tile size only a size can be chosen with what a model has been trained.

**Usage**: 
```bash    
python do_inference.py --tile_size <tile_size> 
```

**Example**:  
```bash    
python do_inference.py --tile_size 512
```

**Input**:  
- Test images from `images/X_test/`  
- Ground truth masks from `labels/y_test/`  
- Trained model: `models/unet_<tile_size>.h5`  

**Output**:  
- Metrics (Accuracy, Precision, Recall) printed to the console.  
- Predictions visualized and saved as `predictions.png`.  

---

## **How to Run the Project**

1. **Slice the Images**:  
   ```bash 
    python slice_images.py 512 
    ```
2. **Train the Model**:  
    ```bash 
   python train_model.py 512
    ```
3. **Run Inference**:
    ```bash 
   python do_inference.py --tile_size 512 
    ```
4. **View Results**:  
   - Metrics (Accuracy, Precision, Recall) will be printed.  
   - Predictions will be saved as `predictions.png`.  

---