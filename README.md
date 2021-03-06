# MPhys-Project

14-2-22
Just uploaded the main file to the repository. The code operates similar to what we used to produce our semester 1 results with a few changes: 
- Updated (hopefully correct) method of calculating the Dice score
- Option to remove scan slices which are all background in both the automatic and corrected segmentation, and save these in new folders
- Also the graphs folder will have 'unfinished' on the end until the code is finished running. This should make it easier to remove the unwanted results folders.

24-2-22
Updated the code which converts the files to numpy arrays so that they are more consistently named and should be a bit more readable. This code relies on the .mha files being placed in a folder called 'Data_old' but this can be easily editted - and also good to check that the filenames all match up. I've also called all the actual numpy files 'init.npy' to differentiate them from 

25-2-22
The code to extract feature vectors is updated to handle the new format of the numpy array data and now saves them in the data folder with them.

8-3-22
Uploaded the code which can now augment the data. Initially this is just rotation (90, 180, 270) and flipping (horizontal and vertical), but there is room for more to be added

28-4-22
Updated the 'Classification and Regression.py' file so it now has the ability to calculate the continuous DSC. Running this gives a very good AUC (around 0.97) so not sure if this is legit so may be worth looking at.

2-5-22
Added 'collate probablility data into correct folder' which converts the prob data from steve into the method we are storing the rest of the data.
Added 'feature vectors using probablilty data' which calculates the feat. vector for the prob data in R and G and the MRI data in B. Can then be used in main code under name 'prob_feat'
