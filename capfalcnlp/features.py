from functools import lru_cache
import math
import shutil
import re

import spacy

from capfalcnlp.processing import (
    remove_multiple_whitespaces,
    get_spacy_content_tokens,
    split_in_sentences,
    count_content_tokens,
)
from capfalcnlp.paths import FASTTEXT_EMBEDDINGS_DIR
from capfalcnlp.helpers import yield_lines, download_and_extract, download


def download_fasttext_embeddings_if_needed(language="fr", training_corpus="common_crawl"):
    # TODO: Repeated code
    if training_corpus == "common_crawl":
        fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f"cc.{language}.300.vec"
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz"
        if not fasttext_embeddings_path.exists():
            fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    elif training_corpus == "wikipedia":
        fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f"wiki.{language}.vec"
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{language}.vec"
        if not fasttext_embeddings_path.exists():
            fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(download(url), fasttext_embeddings_path)
    else:
        raise NotImplementedError(f'Training corpus {training_corpus} not recognized')
    return fasttext_embeddings_path


@lru_cache()
def get_word2rank(vocab_size=10 ** 5, language="fr", training_corpus="common_crawl"):
    word2rank = {}
    line_generator = yield_lines(download_fasttext_embeddings_if_needed(language, training_corpus))
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(" ")[0]
        word2rank[word] = i
    return word2rank


def get_rank(word, **kwargs):
    word2rank = get_word2rank(**kwargs)
    return word2rank.get(word, float('inf'))


def get_log_rank(word, **kwargs):
    return math.log(1 + get_rank(word, **kwargs))


def is_number(word):
    word = str(word)
    float_regex = '^[+-]?(?:[0-9]*[.,])*[0-9]+$'
    return re.match(float_regex, word) is not None


def clean_text(text):
    text = text.replace("\n", " ")
    return remove_multiple_whitespaces(text)


def is_written_in_capitals(word):
    word = str(word)
    return word.upper() == word and word.lower() != word  # The second test excludes strings like '2002' or '.'


def is_frequent_word_written_in_capitals(word, rank_threshold=50000, language='fr', **kwargs):
    word = str(word)
    return is_written_in_capitals(word) and get_rank(word.lower(), language='fr', **kwargs) < rank_threshold


def is_english_word(word):
    # Example usage:
    # for word in ['patient', 'exclusive', 'vice', 'instance', 'cause', 'cigarette', 'hall', 'former', 'notification', 'agent', 'fax', 'satisfaction', 'condition', 'algorithm', 'feature', 'meeting', 'call', 'afterwork', 'jetlag', 'smartphone']:  # noqa
    #     print(word, is_english_word(word), get_rank(word, language='fr'), get_rank(word, language='en'))
    word = cast_to_lemma_if_possible(word)
    return (
        get_rank(word, language='fr') > 5 * get_rank(word, language='en')
        or get_rank(word, language='fr') > 15000
        and get_rank(word, language='en') < 5000
    )


def cast_to_lemma_if_possible(word):
    if isinstance(word, spacy.tokens.token.Token):
        word = word.lemma_
    return word


def is_rare_word(word, rank_threshold=3500):
    word = cast_to_lemma_if_possible(word)
    return get_rank(word.lower()) > rank_threshold and not is_number(word)


def is_accronym(word):
    word = str(word)
    return re.match(r'^(?:[A-Z]\.?)+$', word) is not None and not is_frequent_word_written_in_capitals(word)


def is_abbreviation(word):
    word = str(word)
    # From 'https://fr.wiktionary.org/wiki/Annexe:Abr%C3%A9viations_en_fran%C3%A7ais'
    # fmt: off
    abbreviations = ['1re', '2d', '2de', '1o', '2o', '3o', '7bre', '8bre', '9bre', '10bre', 'A.C.N.', 'ad lib.', 'A.M.', 'ann.', 'app.', 'appt', 'apr.', 'arrt', 'a/s', '℁', 'auj.', 'avc', 'ac', 'BAC', 'B.À.T.', 'b.à.t.', 'bât.', 'bt', 'bcp', 'bd', 'boul.', 'B.P.', 'BTP', 'ca.', 'circa', 'c.-à-d.', 'c.a.d.', 'Cal', 'c. à s.', 'c/s', 'c.c.', 'c/c', 'cc', 'C.C.', 'Cdt', 'cdmt', 'cedex', 'Cel', 'ch.', 'ch.-l.', 'chauff', 'chap.', 'Chr', 'Cie', 'C.N.', 'Cne', 'C.N.S.', 'c/o', 'a/s', '℁', 'contr.', 'C.P.I.', 'cpdt', 'C.Q.F.D', 'crs.', 'C.S.', 'Cte', 'Ctesse', 'dc', 'Dir.', 'dº', 'dito', 'doct.', 'dpdv', 'ds', 'dsl', 'dvlpt', 'dvt', 'ê', 'e. g.', 'e.g.', 'env.', 'et al.', 'cie', 'Éts', 'Ets', 'E.V.', 'exp', 'fact.', 'fasc.', 'féd.', 'fém.', 'févr.', 'FF', 'ff.', 'fg', 'fig.', 'fin.', 'fl.', 'fo', 'fº', 'fol.', 'fos', 'fr.', 'FRF', 'ft', 'fts', 'Gal', 'gd', 'gon', 'hab.', 'HS', 'ibid.', 'id.', 'i. e.', 'id est', 'c.-à-d.', 'inf.', 'Ing.', 'Ir', 'JF', 'JH', 'jms', 'J.O.', 'JO', 'K7', 'l. c.', 'L.D.', 'LL.AA.', 'LL.AA.II.', 'LL.AA.RR.', 'LL.AA.SS.', 'LL.EE.', 'LL.MM.', 'LL.MM.II.RR.', 'LOC', 'loc. cit.', 'Lt', 'méga', 'm̂', 'M.A.', 'Moyen Âge', 'màj', 'm-à-j', 'Mal', 'masc.', 'Md', 'Md', 'm.g.', 'Mgr', 'Mlle', 'Mlles', 'MM.', 'Mmes', 'ms.', 'mtn', 'NA', 'N/A', 'n/a', 'N.B.', 'N.D.', 'N.D.A.', 'N.D.L.R.', 'N.D.T.', 'N.D.W.', 'N.N.', 'NN.SS.', 'Nº', 'nº', 'N.P.A.I.', 'n/réf.', 'ns', 'N.S.', 'op. cit.', 'opt', 'pb', 'p.c.c.', 'pcq', 'p.c.q.', 'P.-D. G.', 'pdt', 'p.ex.', 'p.j.', 'P.J.', 'pk', 'pq', 'pl.', 'plrs', 'pls', 'p/o', 'p.p.', 'Pr', 'P.-S.', 'P.S.', 'pt', 'Pt', 'pt', 'pts', 'pte', 'PTI', 'PV', 'qcq', 'qd', 'QED', 'qq', 'qqch', 'qqf', 'qqf.', 'qqn', 'qqpt', 'qsp', 'qté', 'qté', 'qdm', 'R.T.F.M', 'RTFM', 'R.A.S.', 'R.À.S.', 'rd', 'R.-V.', 'RV', 'rdv', 'REP', 'R.I.P.', 'ro', 'rº', 'RLMI', 'R.P.', 'R. P.', 'RSVP', 's/', 'S.A.', 'S.A.I.', 'S.A.R.', 'S.A.S.', 's/c', 'Sce', 'Sce', 's.d.', 'SDF', 'S.E.', 'Ste', 'Ste', 'sec.', 'sect.', 'SGDG', 'sing.', 's.l.', 'slt', 'S.M.', 'S.M.I.R.', 's/o', 'spec', 'sq.', 'sqq.', 'ss', 's/s', 'SS.', 'Sts', 'Sts', 'ssi', 'Ste', 'Ste', 'Stes', 'Stes', 'S.S.', 'sté', 'Sté', 'stp', 'STP', 'suiv.', 'sup.', 'suppl.', 'svp', 'SVP', 't.', 'tél.', 'téléc.', 'TGV', 'tgv', 'tjrs', 'tjs', 'tlm', 'TPS', 'tq', 'ts', 'TSA', 'T.S.V.P.', 'tt', 'vb.', 'Ve', 'vo', 'vº', 'v/réf', 'vs.', 'V.S.O.P.', 'v .t.', 'Vte', 'Vtesse', 'Vve', 'XL', 'XXL', 'XXXL', 'TTTG', 'X.O.', 'XS', 'TP', 'y c.', 'Z.A.', 'Z.A.C.', 'Z.I.', 'ZUP', 'zup', 'aa', 'ab', 'ac', 'ae', 'af', 'ag', 'aj', 'ak', 'ao', 'ap', 'aq', 'ar', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'bf', 'bg', 'bh']  # noqa
    # fmt: on
    return word in abbreviations


def is_slang(word):
    word = str(word)
    # From https://fr.wiktionary.org/wiki/Annexe:Liste_de_termes_d%E2%80%99argot_Internet
    # fmt: off
    slang_words = ['@tt', '@tte', 'a tte', 'a12c4', 'a2m1', '2m1', 'abs', 'ac', 'ak', 'aek', 'avc', 'av', 'ajd', 'oj', 'auj', 'ajdh', 'apl', 'a+', '++', '@+', '++++', 'à+', 'ama', 'amha', 'arf', 'erf', 'asv', 'att', 'atta', 'b1', 'bb', 'bp', 'bcp', 'bg', 'bj', 'bjr', 'bsr', 'biz', 'bsx', 'bx', 'bn', 'b8', 'btg', 'bvo', 'c', 'cad', 'càd', 'cbr', 'cc', 'couc', 'chu', 'chuis', 'cki', 'cmr' 'cmer', 'cb', 'cmb', 'ctb', 'ctup', 'cpg', 'pas rave', 'crari', 'ctc', 'dac', 'dacc', 'dak', 'dc', 'dcdr', 'déc-redéc', 'déco-reco', 'dmc', 'dtc', 'dsc', 'dr', 'dsl', 'dtf', 'ect', 'ec', 'ets', 'fd’a', 'fdg', 'fdp', 'ff', 'fpc', 'fr', 'fra', 'ftg', 'gb', 'geta', 'gnih', 'gné', 'gp', 'hg', 'hs', 'hihi', 'htkc', 'jam', 'jdc', 'jdr', 'jdçdr', 'jmef', 'jpp', 'jre', 'je re', 'jrb', 'jta', 'jtad', 'jtdr', 'jtd', 'jtb(f)', 'jtkc', 'jtl(g)', 'jtm', 'j’tm', 'j’t’<3', 'j’tmz', 'jts', 'k.', 'kk', 'kay', 'K7', 'KC', 'kestuf', 'kdo', 'ki', 'kiss', 'kikoo', 'kikou', 'kikoolol', 'klr', 'clr', 'koi', 'kwa', 'koi29', 'koid9', 'lgtmp(s)', 'lgt', 'loule', 'loul', 'lal(e)', 'lold', 'laule', 'lolilol', 'lu', 'lut', 'merki', 'mici', 'mci', 'ci', 'miki', 'mdb', 'mdp', 'mdr', 'mi', 'mm', 'mouaha', 'mouhaha', 'mp', 'mpm', 'ms', 'msg', 'mtnt', 'mnt', 'mwa', 'moa', 'moua', 'ndc', 'nn', 'nan', 'na', 'nop(e)', 'noraj', 'nrv', 'ns', 'nspc', 'ntm', 'nptk', 'oki', 'ololz', 'osef', 'osefdtl', 'osefdts', 'osefdtv', 'oseb', 'ouer', 'ué', 'uè', 'vi', 'mui', 'moui', 'wé', 'woué', 'yep', 'ouep', 'ouè', 'oué', 'oé', 'oè', 'ui', 'wé', 'uep', 'vui', 'voui', 'yup', 'ouai', 'ouais', 'pb', 'pd', 'pde', 'pdt', 'plop', 'pouet', 'poy', 'yop', 'plv', 'ptdr', 'pk', 'pq', 'prq', 'prk', 'pkoi', 'pr', 'psk', 'pck', 'pq', 'pke', 'pcq', 'p-t', 'p-e', 'pê', 'ppda', 'pqtam', 'ptafqm', 'put1', 'pt1', 'tain', "p't", 'ptn', 'pv', 'qq1', 'kk1', 'qqn', 'qqun', 'rab', 'raf', 'ras', 'rav', 'raz', 'razer', 'rb', 'rep', 're reuh', 're-salut', 'rgd', 'roh', 'rtva', 'rtl', 'slt', 'snn', 'spd', 'spj', 'ss', 'j’ss', 'jss', 'st', 'staive', 'srx', 'stoo', 'stp', 'svp', 'talc', 'tdf', 'teuf', 'tfk', 'tftc', 'tg', 'taggle', 'th', 'tgv', 'tjrs', 'tjs', 'tjr', 'tof', 'tkl', 'tkt', 'tlm', 'tllmnt', 'tllmt', 'tmlt', 'tmtc', 'tmts', 'tpm', 'tps', 'tsé', 'tt', 'tte', 'ts', 'ttt', 'twa', 'toa', 'toua', 'toé', 'vg', 'voggle', 'vmk', 'vmvc', 'vmvs', 'vnr', 'vo', 'vost', 'vostfr', 'vtfe', 'vtff', 'vtv', 'vouai', 'mouai', 'vs', 'wi', 'wè', 'wai', 'wé', 'weta', 'xl', 'xlt', 'xdr', 'xpdr', 'xpldr', 'yop', 'zik', '+1', '-1', '*', 'afaik', 'afk', 'aka', 'anw', 'aright', 'asap', 'asc', 'asd', 'asl', 'atfg', 'b', 'bb', 'bbl', 'bbq', 'bbs', 'b/f', 'bf', 'brb', 'bs', 'btw', 'cu ', 'cu2', 'cul', 'cus', 'coz', 'cr', 'cya', 'ded', 'dl', 'ddl', 'del', 'dnftt', 'elf', 'err', 'fb', 'fbc', 'ffs', 'FPS', 'ftl', 'ftf', 'ftw', 'fyi', 'fu', 'fy', 'fwiw', 'g2g', 'geek', 'gf', 'gfy', 'g/f', 'gg', 'giyf', 'gj', 'gl', 'gtg', 'gn8', 'gr8', 'gtfo', 'hf', 'hp', 'hs', 'hth', 'hbtl', 'ianal', 'idk', 'ig', 'iirc', 'igam', 'ily', 'imho', 'imo', 'imy', 'irl', 'iwhbtl', 'iykwim', 'kos', 'lol', 'lo', 'lmao', 'lmfao', 'lol', 'luv', 'l2p', 'mol', 'm2c', 'm8', 'n', 'n1', 'nbd', 'ne1', 'neways', 'nh', 'nk', 'noob', 'naab', 'np', 'nop', 'ns', 'nt', 'nm', 'nm', 'nsfw', 'nvm', 'nw', 'ofc', 'oic', 'omc', 'omg', 'amagad', 'omj', 'omo', 'omfg', 'omw', 'o rly', 'otf', 'otw', 'owj', 'owned ', 'pwnd ', 'etc.', 'p911', 'pix', 'pk', 'plz', 'pol', 'pos', 'pgm', 'pw(d)', 'pwned', 'r', 'rly', 'o rly', 'rdy', 'rfyl', 'rofl', 'rofl', 'rotfl', 'r0fl', 'roflmao', 'rp', "r'pZ", 'rtfm', 'littéralement', 'rtk', 'sec brb', 'sfw', 'Shawol', 'so', 'srsly', 'sry', 'su', 'Ssup', 'STFU', 'STFW', 'sx', 'thx/ty', 'tk', 'tl;dr', 'toycn', 'troll', 'ttyl', 'u', 'up', 'usa', 'ur', 'Wassup', 'w8', 'wb', 'woot', 'wp', 'wtf', 'wtfg', 'wtg', 'wth', 'wtj', 'wuup2', 'y', 'ya rly', 'yafa', 'ygam', 'ymmv', 'yw', 'acias', 'ara', 'dsps', 'iwal', 'jaja', 'jeje', 'jiji', 'jojo', 'jd', 'kerer', 'mñn', 'muxo', 'pq', 'pk', "qva!", 'ta', 'tb', 'tp', 'tqp', 'weno', 'güeno', 'x', 'xuxa', 'a+', 'asv', 'bcp', 'dsl', 'lol', 'mdr', 'ptdr', 'rotfl', 'slt', 'tlm', '+1', 'plop', 're']  # noqa
    # fmt: on
    return word in slang_words


def get_detectors():
    return {
        # Some detectors need to take the lemma as input, other need to take the word exactly as it is written.
        'Rare': is_rare_word,
        'Majuscules': is_frequent_word_written_in_capitals,
        'Emprunt Anglais': is_english_word,
        'Accronyme': is_accronym,
        'Abbréviation': lambda word: is_abbreviation(word) or is_slang(word),
        'Nombre': is_number,
    }


def run_detectors(text):
    text = clean_text(text)
    detector_results = {}
    for token in get_spacy_content_tokens(text, language='fr'):
        if str(token) in detector_results:  # No need to run the detectors again on a word that was already seen.
            continue
        detector_results = [name for name, detector in get_detectors().items() if detector(token.lemma_)]
        if len(detector_results) > 0:
            detector_results[str(token)] = detector_results
    return detector_results


def get_substring_start_indexes(substring, text):
    # TODO: We might include some parts of words, e.g. if substring = "mange", "manger" will also be included if it is present in the text
    start_indexes = []
    offset = 0
    remaining_text = text
    while substring in remaining_text:
        start_index = remaining_text.index(substring)
        start_indexes.append(offset + start_index)
        offset += start_index + len(substring)
        remaining_text = text[offset:]
    return start_indexes


def get_long_sentences(text, threshold=11, count_method=count_content_tokens):
    sentences = split_in_sentences(text)
    return [sentence for sentence in sentences if count_method(sentence) > threshold]
