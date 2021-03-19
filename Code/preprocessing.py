from pygoose import *
import nltk
import nltk.data
from nltk.corpus import stopwords as st
from config import project
from helpers import get_rec_sys_dir

nltk.data.path.append(get_rec_sys_dir() + '/libs/nltk_data')

spelling_corrections = kg.io.load_json(project.aux_dir + 'spelling_corrections.json')
stopwords = set(st.words('english'))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def translate(text, translation):
    for token, replacement in translation.items():
        text = text.replace(token, ' ' + replacement + ' ')
    text = text.replace('  ', ' ')
    return text


def spell_digits(text):
    translation = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }
    return translate(text, translation)


def expand_negations(text):
    translation = {
        "can't": 'can not',
        "won't": 'would not',
        "shan't": 'shall not',
    }
    text = translate(text, translation)
    return text.replace("n't", " not")


def correct_spelling(text):
    return ' '.join(
        spelling_corrections.get(token, token)
        for token in tokenizer.tokenize(text)
    )


def get_question_tokens(question, lowercase=True, spellcheck=True, remove_stopwords=True):
    if lowercase:
        question = question.lower()

    if spellcheck:
        question = correct_spelling(question)

    question = spell_digits(question)
    question = expand_negations(question)

    tokens = [token for token in tokenizer.tokenize(question.lower() if lowercase else question)]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    # tokens.append('.')
    return tokens


def get_tokens_lowercase_spellcheck_remove_stopwords(pair):
    return get_question_tokens(pair[0], lowercase=True, spellcheck=True, remove_stopwords=True)


def get_tockens(df, field, len_train):
    tokens_spellcheck = kg.jobs.map_batch_parallel(
        df.as_matrix(columns=[field]),
        item_mapper=get_tokens_lowercase_spellcheck_remove_stopwords,
        batch_size=1000,
    )

    kg.io.save(
        tokens_spellcheck[:len_train],
        project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords_%s_train.pickle' % field
    )

    kg.io.save(
        tokens_spellcheck[len_train:],
        project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords_%s_test.pickle' % field
    )
    return tokens_spellcheck


def extract_vocabulary(tokens_spellcheck):
    vocab = set()
    for question in progressbar(np.array(tokens_spellcheck).ravel()):
        for token in question:
            vocab.add(token)
    return vocab


def get_features_for_train():
    df_train = pd.read_csv(project.data_dir + 'train.csv', nrows=100).fillna(0)
    df_test = pd.read_csv(project.data_dir + 'test.csv', nrows=100).fillna(0)
    df_all = pd.concat([df_train, df_test])
    print(df_all.head())
    tokens_spellcheck_descriptions = get_tockens(df_all, 'descriptions', len(df_train))
    tokens_spellcheck_titles = get_tockens(df_all, 'titles', len(df_train))
    vocab = extract_vocabulary(tokens_spellcheck_descriptions + tokens_spellcheck_titles)
    kg.io.save(df_train['category_id'].values, project.features_dir + 'y_train.pickle')
    kg.io.save(df_test['category_id'].values, project.features_dir + 'y_test.pickle')
    kg.io.save_lines(
        sorted(list(vocab)),
        project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords.vocab'
    )





