# MAT 167 Final Project

# Intro 

Within this folder, I have sectioned my code into four main modular progrmas. processing.py accesses the Free Music Archive and samples 7 GB of publically available music. This is equivalent to audio clips of 30 seconds each with 8000 snippets taken per second to extrapolate the data. This is stored in 79 matlab files containing nearly 100 clips each


# Methods

After Processing these, I load the data into the extraction folder. This samples the audio data for each instance and pulls features from each file. This will be useful later when we dimensionality reduction and clustering


## important note - 
I used MFCC method of feature extraction in lieu of sliding discrete fourier transformation. I have used the implementation separately with this code, but it provided large outliers which messed up the dimensionality reduction. 

