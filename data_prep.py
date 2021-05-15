import pandas as pd
import numpy as np

# the data set needs cleaning to maximize efficiency
# it will be exported to a new csv with this program then used in the main project file

# this section of the data will also include some observations about the data to make note of later

# import data
# the reason score and score max have to be strings is because the original data has empty cells

df = pd.read_csv('data.csv', dtype={'score': 'str', 'score_max': 'str'})

# the data includes many different languages, for the purpose of analysis it will use exclusively English, though
# so first it will be a good idea to remove the other language entries

df.drop(df[df['lang'] != 'en'].index, inplace=True)

print(len(df))

# this confirms that while the original data set held 1415138 reviews, the new one is down to 554746
# that is approx. 39%, which is the same amount as what the Kaggle dataset says (at time of accessing)
# this means that we have successfully dropped the non-English rows

# check for null values in the score and max columns

print(df['score'].isnull().sum())
print(df['score_max'].isnull().sum())

# assuming that all the empty score values are the same (we have no reason not to), there are only 5107 empty values
# this is an extremely small amount of empty data, so it can be dropped

# fill in those values with 0 and convert to numeric
# this will also allow us to try and fill in later with the ML model

df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0, downcast='infer')
df['score_max'] = pd.to_numeric(df['score_max'], errors='coerce').fillna(0, downcast='infer')

# check for the unique values in score_max
# this will give us a better look at exactly what is needed for the sentiment analysis

def get_unique(num):

    list = []
    unique = set(num)

    for num in unique:
        list.append(num)

    return list

print(get_unique(df['score_max']))

# we can see that all of the maximum scores are out of 10
# however, it is possible that some of the scores given might be 0, so keeping the score max column for now
# is useful since it will allow us to check for the originally empty cells more easily in the future

# the first csv export will not include a positive/negative boolean, but will be saved

df.to_csv('saved_score.csv')

# separate scores into positive/negative boolean
# scores in the middle could go either way, so we can remove them and even could use them for later
# though there are only 1869 scores of the 554746 English reviews that are = 5
# also so they do not get added into the testing set, we are dropping all of the rows where the scores are not known
# rows with unknown scores are where score_max is not 10

df = df[df['score']!= 5]
df = df[df['score_max'] == 10]
df['positive'] = np.where(df['score']>5, 1, 0)

# csv export

df.to_csv('new_data.csv')