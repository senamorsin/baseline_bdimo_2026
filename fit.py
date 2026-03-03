from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = pd.read_csv('train.tsv', sep='\t')
data = data.fillna('')
X_train, X_test, y_train, y_test = train_test_split(data.drop(['category_id', 'department_id'], axis=1),
                                                    data['department_id'],
                                                    test_size=0.2, random_state=42)

catboost_clf = CatBoostClassifier(iterations=2500, learning_rate=0.01, depth=6, verbose=True, task_type="GPU")
catboost_clf.fit(X_train,
                 y_train,
                 text_features=['title', 'description'],
                 cat_features=['vendor_name', 'vendor_code', 'shop_category_name'],
                 use_best_model=True,
                 eval_set=[(X_test, y_test)])

predictions = catboost_clf.predict(X_test)
print(f1_score(y_test, predictions, average='macro'))