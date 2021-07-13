import argparse
import json

from capfalcnlp.helpers import read_file
from capfalcnlp.processing import spacy_process
from capfalcnlp.features import get_word_detectors, skip_word_detection, get_long_sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    args = parser.parse_args()
    text = read_file(args.input_file)

    detections = []
    for token in spacy_process(text, language='fr'): 
        detector_results = []
        if not skip_word_detection(token):
            detected_types = [name for name, detector in get_word_detectors().items() if detector(token)]
            for detected_type in detected_types:
                detections.append({'text': str(token), 'char_offset': token.idx, 'detected_type': detected_type})

    for sentence in get_long_sentences(text):
        detections.append({'text': sentence, 'char_offset': text.index(sentence), 'detected_type': 'Phrase Longue'})

    print(json.dumps(detections))
