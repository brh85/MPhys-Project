# MPhys-Project

14-2-22
Just uploaded the main file to the repository. The code operates similar to what we used to produce our semester 1 results with a few changes: 
- Updated (hopefully correct) method of calculating the Dice score
- Option to remove scan slices which are all background in both the automatic and corrected segmentation, and save these in new folders
- Also the graphs folder will have 'unfinished' on the end until the code is finished running. This should make it easier to remove the unwanted results folders.

24-2-22
Updated the code which converts the files to numpy arrays so that they are more consistently named and should be a bit more readable. This code relies on the .mha files being placed in a folder called 'Data_old' but this can be easily editted - and also good to check that the filenames all match up. I've also called all the actual numpy files 'init.npy' to differentiate them from 
