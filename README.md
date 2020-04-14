# COVID

## Classification
Below are the instructions for running the classification script
### Requirements
Please see the requirements.txt file in classification folder

### To Run
- The network can be both trained and validated using the run.sh script provided by changing --evaluate argument
- Tweak the other arguments as per convenience. A description of each argument is given in next section. 
- CSV format: Two columns separated by ','
  - path : full path pf the image file
  - target : 1 for presence of anomaly and 0 for absence 
  
### Arguments
--data : path to dataset
--model-dir : path to model directory
--image-size', '-i' : image size (default: 320)
--workers : number of data loading workers (default: 4)
--epochs : number of total epochs to run
--start-epoch : manual epoch number (useful on restarts)
--batch_size : mini-batch size (default: 16)
--lr : initial learning rate
--lrp : learning rate for pre-trained layers
--momentum : momentum
--weight-decay : weight decay (default: 1e-4)
--print_freq : print frequency (default: 0)
--resume : path to latest checkpoint (default: none)
--evaluate : evaluate model on validation set
--k : number of regions (default: 0)
--alpha : weight for the min regions (default: 0)
--maps : number of maps per class (default: 0)
--adam : Use Adam
--arch : Use Baseline/Wildcat/Weldon (default:0 (Baseline))
--variant : Use Densenet/Resnet/VGG (default:0 (Densenet))
--train_csv : Give path to train csv
--val_csv : Give validation csv
--balanced : Specify if you need balanced sampling or not (Default:1 (Balanced Sampling))
--loss : Specify which criterion you need (default:0 (BCELoss))

Note - The model works with default arguments, you really don't need to set them, they are just there incase you need greater control over what the model does, the basic arguments that you should ideally set are all written up in run.sh, just replace the values ther.

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
