import glob
from typing import List

def get_filenames(dir_path: str, specific_name: str, withDirPath=True) -> List[str]:
    '''
    get_filenames
    -----
    This function can find any specific name under the dir_path, even the file inside directories.

    specific_name:
        >>> Can type any word or extension.
        e.g. '*cat*', '*.csv', '*cat*.csv',
    '''
    if dir_path[-1] != '/':
        dir_path += '/'

    filenames = glob.glob(f'{dir_path}**/{specific_name}', recursive=True)

    if '*.' == specific_name[:2]:
        filenames.extend(glob.glob(f'{dir_path}**/{specific_name[1:]}', recursive=True))

    if withDirPath is False:
        dir_path_len = len(dir_path)
        filenames = [filename[dir_path_len:] for filename in filenames]

    return filenames


