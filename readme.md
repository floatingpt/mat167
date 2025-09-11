# MAT 167 Final Project

# Intro 

Within this folder, I have sectioned my code into four main modular progrmas. processing.py accesses the Free Music Archive and samples 7 GB of publically available music. This is equivalent to audio clips of 30 seconds each with 8000 snippets taken per second to extrapolate the data. This is stored in 79 matlab files containing nearly 100 clips each


# Methods

After Processing these, I load the data into the extraction folder. This samples the audio data for each instance and pulls features from each file. This will be useful later when we dimensionality reduction and clustering


## important note - 
I used MFCC method of feature extraction in lieu of sliding discrete fourier transformation. I have used the implementation separately with this code, but it provided large outliers which messed up the dimensionality reduction. 

Next, I computed the laplacian matrix using cosine similarity on the audio data BEFORE MFCC. I did K-means and PCA on this, and the outcome showed to be a poor indicator for accurately clustering.

In the feature-extracted data, I found 6 clusters within the approx 8000 clips using PCA. I need to map these to values such as , but due to time contraints, I could not finish this tonight.

A link to the [project github](https://github.com/floatingpt/mat167)
