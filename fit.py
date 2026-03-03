from catboost import CatBoostClassifier
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')

#ЗАГРУЗКА ДАТАСЕТА
data = pd.read_csv('train.tsv', sep='\t')
data = data.fillna('')

#Чистка
stop_words = {'<br>', '<br/>', '<p>', '</p>', '<ul>', '</ul>', '<li>', '</li>', '<a>', '</a>', '<b>', '</b>',
    '&quot;', '&nbsp;', '\\r', '\\n', '\\t', '\\xa0', '\n', '\t', ' ', '  ', '   ',}



def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words_lmtk = set(stopwords.words('russian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    filtered_words = [word for word in filtered_tokens  if word not in stop_words]
    return f" ".join(filtered_words)


data['description'] = data['description'].apply(clean_text)

data['vendor_name'] = data['vendor_name'].replace([",Без бренда", "Нет бренда"], None)

vectorizer = TfidfVectorizer()
vectorizer.fit(data['description'])
feature_names = vectorizer.get_feature_names_out()
idf_scores = vectorizer.idf_
words_idf = pd.DataFrame({'word': feature_names, 'idf': idf_scores})
words_idf = words_idf.sort_values(by='idf', ascending=True)
corpus_stopwords = words_idf.head(100)['word'].tolist()
print(corpus_stopwords[:20]) #top 20 stopwords
with open('custom_stopwords.txt', 'w', encoding='utf-8') as f:
    for word in corpus_stopwords:
        f.write(f"{word}\n")


#SPLIT
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

catboost_clf.save_model("model.json", format="json")



