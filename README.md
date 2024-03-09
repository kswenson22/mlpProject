Code for MLP Project 2023/24 at the University of Edinburgh

The project involves adjusting the loss function of a CycleGAN model to include an optimal transport loss term, the sinkhorn loss.
The CycleGAN is then trained on the classic style translation task of Horses to Zebras.

Code was run on Python 3.11 on Macbook Air M2, 8-core GPU.

In order to run the code on your device, make sure you have the relevant packages installed: pytorch, torch,
torchvision, matplotlib, scikit-image, pillow, pandas, numpy.

For PyTorch, it is recommended to consult the PyTorch documentation (https://pytorch.org/) to determine the correct version for your machine.
It is also advisable to create a separate virtual environment for PyTorch with the relevant packages and versioning. This can be created
using pip with the following commands:

python -m venv pytorch_env
source env/bin/activate

The relevant packages can then be installed as follows:
pip install pytorch torch torchvision matplotlib scikit-image pillow pandas numpy

To run on your device, download the zebraHorse dataset, unzip it, and place it in your working directory. Go inside the trainA
and trainB folders, create a new folder inside of each of them (horses for trainA and zebras for trainB) and move all of the images
inside of that folder. Then change lines 195-196 to reflect the location of these folders. To save metrics correctly, go up and change
line 32 to the directory you want to save metrics to. Optionally, change the checkpoint frequency in line 37 to save the model state 
less frequently. Each checkpoint will take ~65 MB of data, so you may want to consider this. Finally, if you are running on a MAC and 
have configured the use of the Metal Performance Shaders, leave the device set to 'mps' in lines 29 and 391. If not, change to CPU or cuda, 
for GPU optimization on Windows software.

After this, the code should be set to run! Happy training!
