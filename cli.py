import argparse
from pprint import pprint

from capfalcnlp.helpers import read_file
from capfalcnlp.processing import spacy_process
from capfalcnlp.features import get_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    args = parser.parse_args()
    text = read_file(args.input_file)
    detections = get_detections(text)
    pprint(detections)
