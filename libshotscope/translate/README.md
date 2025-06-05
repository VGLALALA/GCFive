# C++ to Python Translation

This module provides a small utility for translating C++ source files into Python
using the [cpp2py](https://pypi.org/project/cpp2py/) package.

The script `translate_cpp_to_python.py` scans the `src/` directory for `.cpp`
files and converts each one to Python, writing the results under
`libshotscope/translate/output/`.

To run the translation script from the repository root:

```bash
python libshotscope/translate/translate_cpp_to_python.py
```

The script prints informative messages if either the `src/` directory does not
exist or no `.cpp` files are found.
