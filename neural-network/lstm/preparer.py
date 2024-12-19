import nltk
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from nltk import *
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.utils import shuffle


nltk.download('stopwords')


def read_data():
    # see in notebook
    pass


def _start_pre_processing(doc):
    doc = re.sub(r'http[s]?://\S+|www\.\S+', '', doc)
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    wpt = WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    custom_stopwords = set(stopwords.words('english')) - {
    'not', 'very', 'never', 'no', 'nothing', 'more', 'less', 'good', 'great', 'happy',
    'excellent', 'amazing', 'bad', 'horrible', 'sad', 'angry', 'worse', 'could', 'should',
    'would', 'might', 'may', 'absolutely', 'completely', 'totally', 'think', 'opinion'
    }
    filtered_tokens = [token for token in tokens if token not in custom_stopwords]
    doc = ' '.join(filtered_tokens)
    return doc


def _str_pre_processing(_str):
    sentences = _str.split('.')
    prepared_corpus = [_start_pre_processing(sentence) for sentence in sentences]
    prepared_corpus = ' '.join(list(filter(None, prepared_corpus)))
    return prepared_corpus


def do_pre_processing(doc):
    if isinstance(doc, str):
        return _str_pre_processing(doc)
    else:
        print("ERROR:   TextPreProcessor can't prepare this type of data.")
        return None


def prepare_reviews(reviews, max_text_len, pre_processing):
    if pre_processing:
        reviews = [do_pre_processing(review) for review in reviews]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    reviews = tokenizer.texts_to_sequences(reviews)
    print(f'rewiev example: {reviews[0]}')

    reviews = pad_sequences(reviews, maxlen=max_text_len)

    return reviews, len(tokenizer.word_index)


def prepare_stars(stars):
    stars = np.array(stars)
    stars = stars.reshape(-1, 1)
    encoder = OneHotEncoder()
    stars = encoder.fit_transform(stars).toarray()
    stars = np.array(stars).astype(int)
    print(f'star example: {stars[0]}')
    return stars


def downsampling(reviews, stars):
    reviews = np.array(reviews)
    stars = np.array(stars).astype(int)

    class_counts = np.bincount(stars)[1:]
    min_count = np.min(class_counts)
    print(f'class counts before downsampling: {class_counts}')

    balanced_reviews = []
    balanced_stars = []

    for star in np.unique(stars):
        class_reviews = reviews[stars == star]
        class_stars = stars[stars == star]

        if len(class_reviews) > min_count:
            class_reviews_resampled, class_stars_resampled = resample(class_reviews,
                                                                      class_stars,
                                                                      replace=False,
                                                                      n_samples=min_count,
                                                                      random_state=42)
            balanced_reviews.extend(class_reviews_resampled)
            balanced_stars.extend(class_stars_resampled)
        else:
            balanced_reviews.extend(class_reviews)
            balanced_stars.extend(class_stars)

    balanced_reviews = np.array(balanced_reviews)
    balanced_stars = np.array(balanced_stars)

    class_counts = np.bincount(balanced_stars)[1:]
    print(f'class counts after downsampling: {class_counts}')

    balanced_reviews, balanced_stars = shuffle(balanced_reviews, balanced_stars, random_state=42)

    return balanced_reviews, balanced_stars


def prepare_data(d, k, max_text_len, pre_processing=True):
    reviews, stars = read_data(d)

    reviews, stars = downsampling(reviews, stars)

    reviews, word_count = prepare_reviews(reviews, max_text_len, pre_processing)
    print(f'reviews count: {len(reviews)}')

    stars = prepare_stars(stars)

    index = int(k * len(reviews))
    train_data = reviews[:index]
    train_answers = stars[:index]
    test_data = reviews[index:]
    test_answers = stars[index:]

    return train_data, train_answers, test_data, test_answers, word_count