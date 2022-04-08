from functools import lru_cache
import re
import string

from capfalcnlp.helpers import mute


@lru_cache()
def get_spacy_model(language='fr', size='md', **kwargs):
    # Inline lazy import because importing spacy is slow
    import spacy

    if language == 'it' and size == 'md':
        print('Model it_core_news_md is not available for italian, falling back to it_core_news_sm')
        size = 'sm'
    model_name = {
        'en': f'en_core_web_{size}',
        'fr': f'fr_core_news_{size}',
        'es': f'es_core_news_{size}',
        'it': f'it_core_news_{size}',
        'de': f'de_core_news_{size}',
    }[language]
    spacy_model = spacy.load(model_name)  # python -m spacy download en_core_web_sm
    if language == 'fr':
        spacy_model = add_leff(spacy_model)
    return spacy_model


def add_leff(spacy_model):

    with mute():
        # TODO: Still does not mute everything
        from spacy_lefff import LefffLemmatizer, POSTagger

        spacy_model.add_pipe(POSTagger(), name='pos', after='parser')
        spacy_model.add_pipe(LefffLemmatizer(after_melt=True), name='lefff', after='pos')
    return spacy_model


@lru_cache(maxsize=10 ** 6)
def spacy_process(text, **kwargs):
    with mute():
        return get_spacy_model(**kwargs)(str(text))


@lru_cache()
def get_nltk_sentence_tokenizer(language='fr'):
    # Inline lazy import because importing nltk is slow
    import nltk
    nltk.download('punkt')
    language = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish',
        'it': 'italian',
        'de': 'german',
    }[language]
    return nltk.data.load(f'tokenizers/punkt/{language}.pickle')


def _split_in_sentences_spacy(text, **kwargs):
    '''Split into sentences using Spacy backend. Not as good as NLTK.'''
    return [str(sentence) for sentence in spacy_process(text, **kwargs).sents]


def _split_in_sentences_nltk(text, **kwargs):
    '''Split into sentences using NLTK backend. Better than Spacy for French.'''
    text = ' '.join(text.split('\n'))  # Remove newlines
    return get_nltk_sentence_tokenizer(**kwargs).tokenize(text)


def split_in_sentences(text, backend='nltk', **kwargs):
    if backend == 'nltk':
        return _split_in_sentences_nltk(text, **kwargs)
    elif backend == 'spacy':
        return _split_in_sentences_spacy(text, **kwargs)
    else:
        raise NotImplementedError(f'Backend {backend} does not exist.')


@lru_cache()
def get_spacy_tokenizer(**kwargs):
    spacy_model = get_spacy_model(**kwargs)
    return spacy_model.Defaults.create_tokenizer(spacy_model)


def is_spacy_content_token(token):
    return not token.is_stop and not token.is_punct and token.ent_type_ == ''  # Not named entity


def get_spacy_content_tokens(text, **kwargs):
    return [token for token in get_spacy_tokenizer(**kwargs)(text) if is_spacy_content_token(token)]


def count_content_tokens(text, **kwargs):
    return len(get_spacy_content_tokens(text, **kwargs))


def remove_multiple_whitespaces(text):
    return re.sub(r'  +', ' ', text)


def count_words(text, **kwargs):
    return len([word for word in spacy_process(text, **kwargs) if not word.is_punct])


def remove_punctuation_characters(word):
    punctuation = string.punctuation + '’'
    return ''.join([char for char in word if char not in punctuation])
