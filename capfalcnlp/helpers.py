import bz2
from contextlib import contextmanager
import gzip
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
from types import MethodType
from urllib.request import urlretrieve
import zipfile

from tqdm import tqdm


@contextmanager
def redirect_streams(source_streams, target_streams):
    # We assign these functions before hand in case a target stream is also a source stream.
    # If it's the case then the write function would be patched leading to infinie recursion
    target_writes = [target_stream.write for target_stream in target_streams]
    target_flushes = [target_stream.flush for target_stream in target_streams]

    def patched_write(self, message):
        for target_write in target_writes:
            target_write(message)

    def patched_flush(self):
        for target_flush in target_flushes:
            target_flush()

    original_source_stream_writes = [source_stream.write for source_stream in source_streams]
    original_source_stream_flushes = [source_stream.flush for source_stream in source_streams]
    try:
        for source_stream in source_streams:
            source_stream.write = MethodType(patched_write, source_stream)
            source_stream.flush = MethodType(patched_flush, source_stream)
        yield
    finally:
        for source_stream, original_source_stream_write, original_source_stream_flush in zip(
            source_streams, original_source_stream_writes, original_source_stream_flushes
        ):
            source_stream.write = original_source_stream_write
            source_stream.flush = original_source_stream_flush


@contextmanager
def mute(mute_stdout=True, mute_stderr=True):
    streams = []
    if mute_stdout:
        streams.append(sys.stdout)
    if mute_stderr:
        streams.append(sys.stderr)
    with redirect_streams(source_streams=streams, target_streams=StringIO()):
        yield


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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output_dir)


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
