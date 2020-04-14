# COVID

## Classification
Below are the instructions for running the classification script
# Requirements


## Detection
Below are instructions for running the detection script
### Requirements:
- Pytorch 0.4.1, Torchvision 0.2.2
- Pydicom 1.4.2
- sklearn 0.20.3

### To Run:
- Run `. create_env.sh` in the detection folder
- Unzip the model in models/* (Uploaded via LFS)
- Create a results directory (where output csv is written to)
- Navigate to predict_images.py
- Change lines #164-166# as necessary

### Input and Output formats:
- Input: CSV containing the dicom image names (without the .dcm extension in the string name)
- Image path: Relative path to the folder containing all of the images 
- Output: CSV file that writes the image name (same format as from input), the x, y, width, height, and score
  - x: X position of the rectangle (int)
  - y: Y position of the rectangle (int)
  - width: width of the rectangle (int)
  - height: height of the rectangle (int)
  - score: How confident the model is in its prediction. 
