# Philip_Portfolio
Data Science Portfolio


# Project 1: NBC

In this section I include two main parts.

1. Cross Validation, through the python script cross_val_training.py I pick the hyper-papermeters for the Naive Bayes model used in main.py.

   1. This scripts imports the classes from the files:
      1. pre_processing.py: this file pre-processes the film reviews depending on the parameters passed.
      2. training.py: this file trains the pre-processed files, the training set is determined by the parameter passed, and also takes in an alpha value for laplace smoothing.
      3. evaluate.py: this evaluates the test set using the trained model created by training.py, the test is also determined by the paramter passed.
2. Main, this script is used to train and evaluate the Naive Bayes model with either the base hyper-papermeters or the tuned parameters found the cross validation script.

   1. This takes the same steps as cross_val_training.py but only runs the model once.

## Cross Validation

In this section I will further discuss how this part works.

This file is used to tune the hyper-parameters, I use 10-fold CV on the 1800 training reviews (900 positive, 900 negative). The parameters can be split into two types:

1. Pre-processing paramters:

   1. Keep Unique: This can be either True or False, if True I only keep the unqiue words and their count as 1 in a review else I save the words with their counts.
   2. Remove Punctuation: This can be either True or False, if True I remove all punctuation in the reviews else I don't.
   3. Expand Negation: This can be either True or False, an example sentence of how expand Negation works is input("I don't know how I passed. Well how did you do?") -> output("I do not_know not_how not_I not_passed. Well how did you do?"). If false the file doesn't change anything.
2. Training parameters:

   1. Alpha: This is the value used for laplace smoothing, as seen below. Laplace smoothing is used to account for words that appear in one class but not the other. The range of alpha used in the python script are [0.01,0.1,0.25,0.5,0.75,1.0].

   $$
   \hat{P}(w_i|c) = \frac{count(w_i, c)+\alpha}{(\sum_{w \in V}count(w,c))+\alpha|V|}
   $$

The cross_val_training file takes in the five flags:

1. "-ku": ["True", "False"], this is the value for Keep Unique, used while you tune for alpha
2. "-rp":["True", "False"], this is the value for Remove Punctuation, used while you tune for alpha
3. "-exn":["True", "False"], this is the value for Expand Negation, used while you tune for alpha
4. "-a":[0.01-1], this is the value for alpha if you want to tune for on the paramters 1- 3
5. "-test_alpha":["True", "False"], if "True" the model tunes -a else it tunes -ku, -rp and -exn.

The code below runs the cross validation model with a constant alpha of 1, while it searches for the best hyper-parameters for -ku, -rp and -exn.

`"python .\cross_val_train.py -test_alpha False"`

![Command Line Output](imgs/cv_01.PNG "Running The Python Script")

The scripts starts by pre-processing the film reviews with the current -ku, -rp and -exn paramteres. The processed files are saved as json files with the key, value pairs word and its count. After the model is done pre-processing, training.py takes the pre-processed reviews and trains using 10-fold cross validation, the accuracy and confusin matrix below are calculated by the evaluation.py script on the 180 validation reviews.

![Command Line Output](imgs/cv_02.PNG "Running The Python Script")

Once the cross validation model is done the results are saved to "cross_validation_results/constant_alpha.csv". 

If you want to tune alpha you can run the line below. This tunes alpha while keeping -ku, -rp and -exn constant.

`"python .\cross_val_train.py -ku True -rp True -exn False"`

![Command Line Output](imgs/cv_03.PNG "Running The Python Script")

Once the cross validation model is done the results are saved to "cross_validation_results/different_alpha.csv". I will discuss this further in the overview.pdf and the ANALYSIS.md file.

Once I have picked the values for -ku, -rp,-exn and -a. I will train the model using all the training data and evaluate on the unseen test data using the python script main.py.
