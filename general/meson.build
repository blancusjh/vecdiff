project('vecdiff-general', 'c', 'fortran', version: '0.1.0',
        default_options: ['buildtype=release'])
py = import('python').find_installation(pure: false)
numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
f2py_inc = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
bindings = custom_target('reference-bindings',
  input: 'native/reference.pyf',
  output: ['_reference_coremodule.c', '_reference_core-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@'])
# OpenMP for the Fortran BEM only: the C wrapper may be built by Apple clang.
fc = meson.get_compiler('fortran')
gomp = fc.find_library('gomp', required: false)
openmp_flags = gomp.found() and fc.has_argument('-fopenmp') ? ['-fopenmp'] : []
py.extension_module('_reference_core',
  ['native/reference_math.f90', 'native/bem.f90', 'native/muller.f90', 'native/fem.f90', bindings, f2py_inc / 'fortranobject.c'],
  include_directories: include_directories(numpy_inc, f2py_inc),
  fortran_args: openmp_flags,
  dependencies: gomp,
  subdir: 'vecdiff_general/propagation/references', install: true)
cache_dirs = run_command(py, ['-c',
  'from pathlib import Path; import sys; root = Path(sys.argv[1]); print(*sorted({str(p.parent.relative_to(root) / "__pycache__") for p in root.rglob("*.py")}), sep=chr(10))',
  meson.current_source_dir() / 'src/vecdiff_general'], check: true).stdout().strip().split('\n')
install_subdir('src/vecdiff_general', install_dir: py.get_install_dir(),
               exclude_directories: cache_dirs)
