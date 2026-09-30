"""Shared featurizer: word 1-2 gram TF-IDF unioned with char 2-5 gram TF-IDF.

Lives in its own module so both train_model.py and app.py import it by the
same path — that's what makes joblib.load() able to reconstruct the object.
"""

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class PairFeaturizer:
    """Word 1-2 gram TF-IDF unioned with char 2-5 gram TF-IDF."""

    def __init__(self):
        self.word_vec = TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            strip_accents="unicode",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=150_000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
        )

    def fit_transform(self, texts):
        Xw = self.word_vec.fit_transform(texts)
        Xc = self.char_vec.fit_transform(texts)
        return hstack([Xw, Xc]).tocsr()

    def transform(self, texts):
        Xw = self.word_vec.transform(texts)
        Xc = self.char_vec.transform(texts)
        return hstack([Xw, Xc]).tocsr()
