## RootPainter3D

This software is not approved for clinical use. Also please See [LICENSE](https://github.com/Abe404/RootPainter3D/blob/master/LICENSE) file.

Described in the paper "RootPainter3D: Interactive-machine-learning enables rapid and accurate contouring for radiotherapy".

Preprint is available at: https://arxiv.org/pdf/2106.11942.pdf

Published version is available at: http://doi.org/10.1002/mp.15353 

#### Server setup 

For the next steps I assume you have a dedicated GPU with CUDA and cuDNN installed. A recent version of python 3 is required.

1. Clone the RootPainter3D code from the repository and then cd into the trainer directory (the server component).
```
git clone https://github.com/Abe404/RootPainter3D.git
cd RootPainter3D/trainer
```

2. To avoid alterating global packages, I suggest using a virtual environment:
```
python -m venv env --clear
```

Note: Make sure to use python3. You may need to write `python3` instead of `python` to do this.

3. Then activate the virtual environment:

On Linux:
```
source ./env/bin/activate
```

On Windows:
```
env\Scripts\activate.bat
```

4. Install PyTorch by following the instructions at the [pytorch website](https://pytorch.org/get-started/locally/)

5. Install the other dependencies in the virtual environment:
```
pip install -r requirements.txt
```

6. Then, simply run RootPainter by:
```
python main.py
```

This will first create the sync directory. 

You will be prompted to input a location for the sync directory. This is the folder where files, including instructions, are shared between the client and server. I will use ~/root_painter_sync

The RootPainter3D server will then create folders inside ~/root_painter_sync and start watching for instructions from the client.

You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/root_painter_sync on your local machine.

#### Client setup

1. Clone the RootPainter3D code from the repository and then cd into the painter directory (the client component).
```
git clone https://github.com/Abe404/RootPainter3D.git
cd RootPainter3D/painter
```

2. To avoid alterating global packages. I suggest using a virtual environment. Create a virtual environment 
```
python -m venv env
```

And then activate it.

On linux/mac:
```
source ./env/bin/activate
```

On windows:
```
env\Scripts\activate.bat
```

3. Install the other dependencies in the virtual environment
```
pip install -r requirements.txt
```

4. Run the client.
```
python src/main.py
```

To interactively train a model and annotate images, you will need to add a set of compressed NIfTI images (.nii.gz) to a folder in the datasets folder and then create a project using the client that references this dataset.

For more details, see [the published article](http://doi.org/10.1002/mp.15353).
