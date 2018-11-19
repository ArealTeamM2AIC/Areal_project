This is a starting kit for the Inria Aerial Image Labeling challenge. 
We use the INRIA satelite dataset from [link](https://project.inria.fr/aerialimagelabeling/). The data set contains 180 of high resolution satelite images of 5 towns, each having 36 images. For every image, there is a reference image. In a reference image, the tiles are single-channel images with values 255 for the building class and 0 for the not building class.

Example pictures of Chicago
![picture](https://project.inria.fr/aerialimagelabeling/files/2011/12/chi1.jpg)
![picture](https://project.inria.fr/aerialimagelabeling/files/2011/12/chi2.jpg)
References and credits: 
Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat, Pierre Alliez. Can Semantic Labeling
Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark. IEEE International
Symposium on Geoscience and Remote Sensing (IGARSS), Jul 2017, Fort Worth, United States.
<hal-01468452>

Prerequisites:
Install Anaconda Python 3.6.6 

Usage:

(1) If you are a challenge participant:

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Inria Aerial Image Labeling challenge. 
At the prompt type:
jupyter-notebook README.ipynb

- modify sample_code_submission to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly:

    `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_data sample_result_submission scoring_output`

- the metric identified by metric.txt in the utilities directory is the metric used both to compute performances in README.ipynb and for the challenge.
