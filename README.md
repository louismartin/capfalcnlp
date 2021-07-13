# capfalcnlp

## Requirements
Python 3.7

## Install
```bash
pip install -r requirements.txt
pip install -e .
python -m spacy download fr_core_news_md
```


## Usage
```bash
$ python cli.py --input-file example_text.txt
[{'char_offset': 2, 'detected_type': 'Rare', 'text': 'intelligence'},
 {'char_offset': 15, 'detected_type': 'Rare', 'text': 'artificielle'},
 {'char_offset': 29, 'detected_type': 'Rare', 'text': 'IA'},
 {'char_offset': 29, 'detected_type': 'Accronyme', 'text': 'IA'},
 {'char_offset': 140, 'detected_type': 'Rare', 'text': 'simuler'},
 {'char_offset': 150, 'detected_type': 'Rare', 'text': 'intelligence'},
 {'char_offset': 180, 'detected_type': 'Rare', 'text': 'correspond'},
 {'char_offset': 255, 'detected_type': 'Rare', 'text': 'discipline'},
 {'char_offset': 266, 'detected_type': 'Rare', 'text': 'autonome'},
 {'char_offset': 275, 'detected_type': 'Rare', 'text': 'constituée2'},
 {'char_offset': 298, 'detected_type': 'Rare', 'text': 'instances'},
 {'char_offset': 322, 'detected_type': 'Rare', 'text': 'CNIL'},
 {'char_offset': 322, 'detected_type': 'Accronyme', 'text': 'CNIL'},
 {'char_offset': 328, 'detected_type': 'Rare', 'text': 'relevant'},
 {'char_offset': 381, 'detected_type': 'Rare', 'text': 'IA'},
 {'char_offset': 381, 'detected_type': 'Accronyme', 'text': 'IA'},
 {'char_offset': 385, 'detected_type': 'Rare', 'text': 'introduisent'},
 {'char_offset': 424, 'detected_type': 'Rare', 'text': 'mythe'},
 {'char_offset': 457, 'detected_type': 'Rare', 'text': 'classée'},
 {'char_offset': 493, 'detected_type': 'Rare', 'text': 'cognitives'},
 {'char_offset': 526, 'detected_type': 'Rare', 'text': 'neurobiologie'},
 {'char_offset': 540, 'detected_type': 'Rare', 'text': 'computationnelle'},
 {'char_offset': 575, 'detected_type': 'Rare', 'text': 'aux'},
 {'char_offset': 587, 'detected_type': 'Rare', 'text': 'neuronaux'},
 {'char_offset': 587, 'detected_type': 'Emprunt Anglais', 'text': 'neuronaux'},
 {'char_offset': 612, 'detected_type': 'Rare', 'text': 'mathématique'},
 {'char_offset': 637, 'detected_type': 'Rare', 'text': 'mathématiques'},
 {'char_offset': 725, 'detected_type': 'Rare', 'text': 'résolution'},
 {'char_offset': 757, 'detected_type': 'Rare', 'text': 'complexité'},
 {'char_offset': 779, 'detected_type': 'Rare', 'text': 'algorithmique'},
 {'char_offset': 813, 'detected_type': 'Rare', 'text': 'désigne'},
 {'char_offset': 863, 'detected_type': 'Rare', 'text': 'imitant'},
 {'char_offset': 940, 'detected_type': 'Rare', 'text': 'cognitives'},
 {'char_offset': 0,
  'detected_type': 'Phrase Longue',
  'text': "L'intelligence artificielle (IA) est « l'ensemble des théories et des techniques mises en œuvre en vue de réaliser des machines capables de simuler l'intelligence humaine »."},
 {'char_offset': 449,
  'detected_type': 'Phrase Longue',
  'text': 'Souvent classée dans le groupe des sciences cognitives, elle fait appel à la neurobiologie computationnelle (particulièrement aux réseaux neuronaux), à la logique mathématique (partie des mathématiques et de la philosophie) et à l'informatique."},
 {'char_offset': 794,
  'detected_type': 'Phrase Longue',
  'text': 'Par extension elle désigne, dans le langage courant, les dispositifs imitant ou remplaçant l'homme dans certaines mises en œuvre de ses fonctions cognitives.'}]
```
