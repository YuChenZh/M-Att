import glob
import pandas as pd
from sklearn.metrics import jaccard_similarity_score


## ----------------- merge files -----------------------######
# interesting_files = glob.glob("data_csv/train/test/*.csv")
#
# header_saved = False
# with open('data_csv/train/train1.csv','wb') as fout:
#     for filename in interesting_files:
#         with open(filename) as fin:
#             header = next(fin)
#             if not header_saved:
#                 fout.write(header)
#                 header_saved = True
#             for line in fin:
#                 fout.write(line)



### ------------------- similarities among persons(tasks) --------------------#####

n_labels = 51

# load the dataset
dataframe1 = pd.read_csv('data_csv/train/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv')
dataframe1 = dataframe1.drop('label_source', axis=1)  # drop the last column
dataframe1 = dataframe1.fillna(0)

labels1 = dataframe1[dataframe1.columns[-n_labels:]]  # labels are the last 6 columns


dataframe2 = pd.read_csv('data_csv/train/0BFC35E2-4817-4865-BFA7-764742302A2D.features_labels.csv')
dataframe2 = dataframe2.drop('label_source', axis=1)  # drop the last column
dataframe2 = dataframe2.fillna(0)

labels2 = dataframe2[dataframe2.columns[-n_labels:]]  # labels are the last 6 columns

# print (labels1[0:1])

dataframe3 = pd.read_csv('data_csv/train/11B5EC4D-4133-4289-B475-4E737182A406.features_labels.csv')
dataframe3 = dataframe3.drop('label_source', axis=1)  # drop the last column
dataframe3 = dataframe3.fillna(0)

labels3 = dataframe3[dataframe3.columns[-n_labels:]]  # labels are the last 6 columns


similarity1 = jaccard_similarity_score(labels1[0:3000],labels2[0:3000])
similarity2 = jaccard_similarity_score(labels1[0:3000],labels3[0:3000])
similarity3 = jaccard_similarity_score(labels2[0:3000],labels3[0:3000])

print(similarity1)
print(similarity2)
print(similarity3)