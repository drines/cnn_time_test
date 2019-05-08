#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER:       Daniel Rines
# DATE CREATED:     2019.04.23
# REVISED DATE:     2019.04.24
# PURPOSE: Classifies images using a pretrained CNN model, compares these
#          classifications to the true identity of object in the images, and
#          summarizes how well the CNN performed on the image classification
#          task.
#          Note that the true identity of the or object in the image is
#          must be indicated by the filename of the image. The program
#          extracts the pet image label from the filename before
#          classifying the images using the pretrained CNN model. This
#          program was developed to compare the performance of 3 different CNN
#          model architectures to determine which provides the 'best'
#          classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results


# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()

    # Retrieve the 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. Returns
    # the collection of these command line arguments from the function
    # call as the variable in_arg
    in_arg = get_input_args()

    # Checks command line arguments using in_arg
    check_command_line_arguments(in_arg)

    # Create the results dictionary that contains the results, this dictionary
    # is returned from the function call
    results = get_pet_labels(in_arg.dir)

    # Check Pet Images in the results Dictionary using results
    check_creating_pet_image_labels(results)

    # Create Classifier Labels with classifier function, Compares Labels,
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Check Results Dictionary using results.
    check_classifying_images(results)    

    # Adjust the results dictionary to determine if classifier correctly
    # classified images as 'a dog' or 'not a dog'. This demonstrates if
    # model can correctly classify images as real animal (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Check the Results Dictionary for is-a-dog adjustment using results
    check_classifying_labels_as_dogs(results)

    # Create the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages).    
    # Calculates results of run and add statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Check the Results Statistics Dictionary using results_stats
    check_calculating_results(results, results_stats)

    # Print summary results, incorrect classifications of animals (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)

    # Measure total program runtime by collecting end time
    end_time = time()

    # Compute overall runtime in seconds & prints it in hh:mm:ss format
    # calculate difference between end time and start time
    tot_time = end_time - start_time    
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()
