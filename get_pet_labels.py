#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER:   Daniel Rines
# DATE CREATED: 2019.04.23
# REVISED DATE: 2019.04.23
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Grab the list of files from the provided directory
    file_names = listdir(image_dir)
    
    # Instantiate empty dict for the file_name/file_label pairs
    results_dict = {}
    
    # Loop through the file_names list. Note: there is an 
    # underscore character in the file_names that will need
    # to be stripped for pet label designation.
    for file_name in file_names:
        
        # skip any files that start with a '.' (e.g. dirs)
        if file_name[0] != '.':
            
            # Don't add any duplicate file_name keys
            if file_name not in results_dict:
                # create a temporary pet_label variable by striping
                # the file extension and removing the '_' characters.
                # Populate results_dict with the full file_name but 
                # trim any trailing numbers or spaces, and make everything 
                # lowercase for consistency with labels in dognames.txt
                pet_label = file_name.rstrip('.jpg').replace('_',' ')
                results_dict[file_name] = [pet_label.rstrip(' 0123456789').lower()]
            else:
                print("** Warning: Duplicate files exist in directory:", file_name)
    
    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dict
