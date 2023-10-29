# MNIST-Digit-Recognition Basics 

This small project was made to learn the very basics of PyTorch training and testing of the very popular digit MNIST set. This is basically the Hello World for machine learning.

## Installation

If you're interested in running this, I suggest you install [Anaconda](https://www.anaconda.com/installation-success?source=installer) and run the given command from [here](https://pytorch.org/get-started/locally/) to setup your local environment with the dependencies needed. In my case, this is what I ran:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Note: I am using Windows 10 and do have an NVIDIA GPU.

## Usage
To create your first model, you will need to run the main file and wait for a model to be trained and saved. This is done by running
```
python3 main.py
```
By default the model runs through 10 generations, or epochs. If you wish to change this number, simply go to the bottom of the file and change the `epochs` variable to the number of generations you'd like to run through. Once the model is created, trained, and saved, you can start testing with your own data. 

To do this, locally add 28x28 files with white backgrounds and black writing to the local directory. The naming of the files here doesn't matter. Then, go to the `test.py` file and change the `image_paths` variable to either an array of paths like in the example, or change the logic to loop over the paths in a folder for example. 

Finally, to run through the images with the model, run 
```
python3 test.py
```
and enjoy watching your model fail miserably, or somehow guess right.

## Reflection

Since this is my first time doing anything with machine learning, there's obviously much I don't understand, and most of the code is from the [PyTorch Quickstart Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). The testing of custom images is very far off despite the model claiming to be 90% accurate after about 25 generations. I assume the issue here is how the custom images are loaded and processed, or maybe some other factor. Either way, this was, like stated earlier, basically the Hello World of Machine Learning, so I don't plan on spending much more time on it, if any more at all.