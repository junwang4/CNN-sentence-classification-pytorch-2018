import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import data_helpers

def CV_evaluate(X, y, C=1):
    clf = LinearSVC(C = C)
    K = 10
    cv = KFold(K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=5)
    print(scores)
    print("Accuracy: {:.3f}".format(np.mean(scores)))


def do_svm_test_with_extracted_cnn_sentence_vectors():
    fpath_sentence_vectors = 'models/sentence_vectors.npy'
    fpath_sentence_label = 'models/sentence_vectors_y.npy'
    if os.path.exists(fpath_sentence_label) and os.path.exists(fpath_sentence_vectors):
        data = np.load('models/sentence_vectors.npy')
        labels = np.load('models/sentence_vectors_y.npy')
        CV_evaluate(data, labels, C=0.01)
    else:
        print('\nTo run this, you need to first run "python cnn_pytorch.py" to generate sentence vectors.')


def do_bag_of_words():
    vectorizer = CountVectorizer
    vectorizer = TfidfVectorizer
    vec = vectorizer(ngram_range=(1, 2), min_df=2, token_pattern='[^ ]+')

    X_text, y = data_helpers.load_sentences_and_labels()
    X_sparse = vec.fit_transform(X_text).astype(float)
    CV_evaluate(X_sparse, y, C=.5)


def main():
    do_bag_of_words()
    #do_svm_test_with_extracted_cnn_sentence_vectors()

if __name__ == "__main__":
    main()
