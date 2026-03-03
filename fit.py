from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.tsv', sep='\t')
X_train, X_test, y_train, y_test = train_test_split(data.drop(['category_id', 'department_id'], axis=1),
                                                    data[['category_id', 'department_id']],
                                                    test_size=0.2, random_state=42)

departments = data['department_id'].value_counts()
categories = data['category_id'].value_counts()
print(departments)
print(categories)