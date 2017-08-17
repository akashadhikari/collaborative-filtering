import zipfile
from surprise import Reader, Dataset, SVD, evaluate

'''

The Collaborative Filtering Algorithm:

CF predicts items based on the history of ratings that a user gave 
and the history of rating given to an item.
==================================================================

*MOVIELENS DATA FORMAT*

userID | itemID | rating | timestamp
196	      242	  3	       881250949
186	      302	  3	       891717742
22	      377	  1	       878887116
244	      51	  2	       880606923
166	      346	  1	       886397596
...       ...     ...      ...

'''

# Unzip ml-100k.zip
zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
zipfile.extractall()
zipfile.close()

# Read data into an array of strings
with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

# Data preparation with separation '\t' to be used in Surprise
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load the data from the file using the reader format
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# For cross validation, split the dataset into 5 folds
data.split(n_folds=5)

# Choose the algorithm-- Singular Value Decomposition (SVD)
algo = SVD()

'''
The difference between the actual and the predicted rating 
is measured using classical error measurements such as 
Root mean squared error (RMSE) and Mean absolute error (MAE).

'''
# Train and test reporting the RMSE and MAE scores
evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)

# Predict a certain item
userid = str(196)
itemid = str(302)
actual_rating = 4
print(algo.predict(userid, itemid, actual_rating))
