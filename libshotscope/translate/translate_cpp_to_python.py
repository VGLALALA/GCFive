import pathlib
from cpp2py.cpp2py import c2py

def translate_all(src_root: str = 'src', dest_root: str = 'libshotscope/translate/output') -> None:
    """Translate all C/C++ files in *src_root* to Python files in *dest_root*.

    Parameters
    ----------
    src_root : str
        Path to search for ``.cpp`` files. If the directory does not exist
        an informative message is printed and the function returns.
    dest_root : str
        Directory where the translated ``.py`` files are written. The
        directory will be created if necessary.
    """
    src_path = pathlib.Path(src_root)
    if not src_path.exists():
        print(f'Source directory {src_root} does not exist. Nothing to translate.')
        return

    dest_path = pathlib.Path(dest_root)
    dest_path.mkdir(parents=True, exist_ok=True)

    cpp_files = list(src_path.rglob('*.cpp'))
    if not cpp_files:
        print(f'No C++ files found in {src_root}.')
        return

    for cpp_file in cpp_files:
        relative = cpp_file.relative_to(src_path)
        output_file = dest_path / relative.with_suffix('.py')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'Translating {cpp_file} -> {output_file}')
        c2py(str(cpp_file), str(output_file))

if __name__ == '__main__':
    translate_all()
