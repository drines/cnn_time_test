# CNN Time Test
This project is part of the Udacity AI programming course and tests the different timing using a pre-trained image classifier.

## PURPOSE
Classifies images using a pretrained CNN model, compares these
classifications to the true identity of object in the images, and
summarizes how well the CNN performed on the image classification
task.

Note that the true identity of the object in the image is
must be indicated by the filename of the image. The program
extracts the pet image label from the filename before
classifying the images using the pretrained CNN model. This
program was developed to compare the performance of 3 different CNN
model architectures to determine which provides the 'best'
classification.

## Instructions
Use argparse Expected Call with <> indicating expected user input:

 `python check_images.py --dir <directory with images> --arch <model> --dogfile <file that contains dognames>`

Example call:

 `python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt`
