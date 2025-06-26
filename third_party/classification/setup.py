"""
Copyright (C) 2025 dsl.
"""
import os
import os.path as osp
import glob
import setuptools
from setuptools.extension import Extension
import sys
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import shutil


def get_extensions():
    this_dir = osp.dirname(osp.abspath(__file__))
    build_dir = osp.join(this_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    # copy fdcv to build/fdcv_py
    out_dir = osp.join(build_dir, 'src/fdcv')
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(osp.join(this_dir, 'fdcv'), out_dir)
    # get py list
    py_list = glob.glob(osp.join(out_dir, '**', '*.py'), recursive=True)
    py_list = [py for py in py_list if 'license_hook' not in py and '__pycache__' not in py]
    code_py_list = [py for py in py_list if '__init__' not in py]
    init_py_list = [py for py in py_list if '__init__' in py]
    # register_hook
    # register_license_check_hook(code_py_list)
    return code_py_list, init_py_list


BUILD_ONLY_SO  = False
for arg in sys.argv[:]:
    if 'BUILD_ONLY_SO' in arg:
        BUILD_ONLY_SO = True
        sys.argv.remove(arg)

exec(open('fdcv/version.py').read())

def get_packages():
    impl_list = glob.glob('./fdcv/**/impl', recursive=True)
    pkgs = ['fdcv'] + [p[2:].replace(osp.sep, '.') for p in impl_list]
    print('pkgs:', pkgs)
    return pkgs

def get_package_data():
    impl_list = glob.glob('./fdcv/**/impl', recursive=True)
    init_list = glob.glob('./fdcv/**/', recursive=True)
    models_list = glob.glob('./fdcv/**/models', recursive=True)
    pkg_data = {'fdcv': [osp.join(p[7:], '*.so') for p in impl_list] +
                        [osp.join(p[7:], '*.pyd') for p in impl_list] +
                        [osp.join(p[7:], '__init__.py') for p in init_list if '__pycache__' not in p] +
                        [osp.join(p[7:], '*.dll') for p in impl_list] +
                        [osp.join(p[7:], '*.pth') for p in models_list]
                        }
    print('pkg_data:', pkg_data)
    return pkg_data

if BUILD_ONLY_SO:
    code_py_list, init_py_list = get_extensions()
    pkg_data = get_package_data()
    cython_ext_modules = cythonize(code_py_list, 
                                   build_dir="build", 
                                   compiler_directives={'language_level': 3})
    setuptools.setup(
        name="fdcv",
        version=__version__,
        author="",
        author_email="",
        description="nn module",
        cmdclass={'build_ext': build_ext},
        packages=get_packages(),
        package_data=pkg_data,
        ext_modules=cython_ext_modules,
        long_description="DeepLearning model visual evaluation.",
        long_description_content_type="text/markdown",
        url="",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
else:
    setuptools.setup(
        name="fdcv",
        version=__version__,
        author="",
        author_email="",
        description="nn module",
        long_description="DeepLearning model.",
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        package_data=get_package_data(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
