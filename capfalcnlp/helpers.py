import bz2
import gzip
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
from urllib.request import urlretrieve
import zipfile

from tqdm import tqdm


def read_file(filepath):
    def _read_file_with_encoding(filepath, encoding='utf8'):
        with Path(filepath).open('rt', encoding=encoding) as f:
            return f.read()

    for encoding in ['utf8', 'latin-1']:
        try:
            return _read_file_with_encoding(filepath, encoding=encoding)
        except UnicodeDecodeError as e:
            last_exception = e
            pass
    raise last_exception


def yield_lines(filepath, gzipped=False, n_lines=None):
    filepath = Path(filepath)
    open_function = open
    if gzipped or filepath.name.endswith('.gz'):
        open_function = gzip.open
    with open_function(filepath, 'rt') as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip('\n')


TEMP_DIR = None


def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
    percent = int(count * block_size * 100 / total_size)
    msg = f'\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s'
    sys.stdout.write(msg)


def download(url, destination_path=None, overwrite=True):
    if destination_path is None:
        destination_path = get_temp_filepath()
    if not overwrite and destination_path.exists():
        return destination_path
    print('Downloading...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        print('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise
    return destination_path


def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split('/')[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print('Extracting...')
    extracted_paths = extract(compressed_filepath, tmp_dir)
    compressed_filepath.unlink()
    return extracted_paths


def extract(filepath, output_dir):
    output_dir = Path(output_dir)
    # Infer extract function based on extension
    extensions_to_functions = {
        '.tar.gz': untar,
        '.tar.bz2': untar,
        '.tgz': untar,
        '.zip': unzip,
        '.gz': ungzip,
        '.bz2': unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [ext for ext in extensions if filename.endswith(ext)]
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    filename = os.path.basename(filepath)
    extension = get_extension(filename, list(extensions_to_functions))
    extract_function = extensions_to_functions[extension]

    # Extract files in a temporary dir then move the extracted item back to
    # the ouput dir in order to get the details of what was extracted
    tmp_extract_dir = Path(tempfile.mkdtemp())
    # Extract
    extract_function(filepath, output_dir=tmp_extract_dir)
    extracted_items = os.listdir(tmp_extract_dir)
    output_paths = []
    for name in extracted_items:
        extracted_path = tmp_extract_dir / name
        output_path = output_dir / name
        move_with_overwrite(extracted_path, output_path)
        output_paths.append(output_path)
    return output_paths


def move_with_overwrite(source_path, target_path):
    if os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isdir(target_path) and os.path.isdir(source_path):
        shutil.rmtree(target_path)
    shutil.move(source_path, target_path)


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        f.extractall(output_dir)


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, 'r') as f:
        f.extractall(output_dir)


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith('.gz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def unbz2(compressed_path, output_dir):
    extract_filename = os.path.basename(compressed_path).replace('.bz2', '')
    extract_path = os.path.join(output_dir, extract_filename)
    with bz2.BZ2File(compressed_path, 'rb') as compressed_file, open(extract_path, 'wb') as extract_file:
        for data in tqdm(iter(lambda: compressed_file.read(1024 * 1024), b'')):
            extract_file.write(data)


def add_newline_at_end_of_file(file_path):
    with open(file_path, 'r') as f:
        last_character = f.readlines()[-1][-1]
    if last_character == '\n':
        return
    print(f'Adding newline at the end of {file_path}')
    with open(file_path, 'a') as f:
        f.write('\n')


def get_ref_files_prefix(ref_files):
    assert type(ref_files) == list
    if len(ref_files) > 1:
        m = re.match(r'(.+)\.(\d+)', ref_files[0])
        error_msg = 'Reference filenames should be in the form {prefix}.{i}, where i is the index of the file'
        assert m is not None, error_msg
        prefix = m.groups()[0]
        assert all([ref_file == f'{prefix}.{i}' for i, ref_file in enumerate(sorted(ref_files))]), error_msg
    else:
        prefix = ref_files[0]
    return prefix


def replace_lrb_rrb_file(filepath):
    tmp_filepath = filepath + '.tmp'
    with open(filepath, 'r') as input_file, open(tmp_filepath, 'w') as output_file:
        for line in input_file:
            output_file.write(line.replace('-lrb-', '(').replace('-rrb-', ')'))
    os.rename(tmp_filepath, filepath)
