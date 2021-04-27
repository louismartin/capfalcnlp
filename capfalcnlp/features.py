from functools import lru_cache
import math
import shutil

from capfalcnlp.processing import remove_multiple_whitespaces, get_spacy_content_tokens
from capfalcnlp.paths import FASTTEXT_EMBEDDINGS_DIR
from capfalcnlp.helpers import yield_lines, download_and_extract


def download_fasttext_embeddings_if_needed(language="en"):
    fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f"cc.{language}.300.vec"
    if not fasttext_embeddings_path.exists():
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz"
        fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    return fasttext_embeddings_path


@lru_cache()
def get_word2rank(vocab_size=10 ** 5, language="fr"):
    word2rank = {}
    line_generator = yield_lines(download_fasttext_embeddings_if_needed(language))
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(" ")[0]
        word2rank[word] = i
    return word2rank


def get_rank(word, **kwargs):
    word2rank = get_word2rank(**kwargs)
    return word2rank.get(word, len(word2rank))


def get_log_rank(word, **kwargs):
    return math.log(1 + get_rank(word, **kwargs))


def get_complex_words_and_ranks(text, log_rank_threshold=9, **kwargs):
    text = text.replace("\n", " ")
    text = remove_multiple_whitespaces(text)
    complex_words_and_ranks = set()
    for token in get_spacy_content_tokens(text, **kwargs):
        log_rank = get_log_rank(token.lemma_, **kwargs)
        if log_rank > log_rank_threshold:
            complex_words_and_ranks.add((token.text, round(log_rank, 2)))
    return complex_words_and_ranks


def get_complex_words_ranks_and_indexes(text, **kwargs):
    """Finds complex words and returns a list of tuples: (complex_word, start_index, log_rank)"""

    def get_substring_start_indexes(substring, text):
        start_indexes = []
        offset = 0
        remaining_text = text
        while substring in remaining_text:
            start_index = remaining_text.index(substring)
            start_indexes.append(offset + start_index)
            offset += start_index + len(substring)
            remaining_text = text[offset:]
        return start_indexes

    words_ranks_and_indexes = []
    for complex_word, log_rank in get_complex_words_and_ranks(text, **kwargs):
        # TODO: We might include some parts of words, e.g. if complex_word = "mange", "manger" will also be included if it is present in the text
        start_indexes = get_substring_start_indexes(complex_word, text)
        words_ranks_and_indexes.extend((complex_word, start_index, log_rank) for start_index in start_indexes)
    return sorted(words_ranks_and_indexes, key=lambda items: items[1])  # Sort according to start index

    def get_long_sentences(text, content_tokens_threshold=10):
        sentences = split_in_sentences(text)
        return [sentence for sentence in sentences if count_content_tokens(sentence) > content_tokens_threshold]
