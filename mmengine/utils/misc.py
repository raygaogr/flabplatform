# Copyright (c) OpenMMLab. All rights reserved.
import collections.abc
import functools
import itertools
import logging
import re
import subprocess
import textwrap
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec, ismodule
from itertools import repeat
from typing import Any, Callable, Optional, Type, Union


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Optional[Type] = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.

    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         f'match: {sum(lens)} != {len(in_list)}')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def apply_to(data: Any, expr: Callable, apply_func: Callable):
    """Apply function to each element in dict, list or tuple that matches with
    the expression.

    For examples, if you want to convert each element in a list of dict from
    `np.ndarray` to `Tensor`. You can use the following code:

    Examples:
        >>> from mmengine.utils import apply_to
        >>> import numpy as np
        >>> import torch
        >>> data = dict(array=[np.array(1)]) # {'array': [array(1)]}
        >>> result = apply_to(data, lambda x: isinstance(x, np.ndarray), lambda x: torch.from_numpy(x))
        >>> print(result) # {'array': [tensor(1)]}

    Args:
        data (Any): Data to be applied.
        expr (Callable): Expression to tell which data should be applied with
            the function. It should return a boolean.
        apply_func (Callable): Function applied to data.

    Returns:
        Any: The data after applying.
    """  # noqa: E501
    if isinstance(data, dict):
        # Keep the original dict type
        res = type(data)()
        for key, value in data.items():
            res[key] = apply_to(value, expr, apply_func)
        return res
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        # namedtuple
        return type(data)(*(apply_to(sample, expr, apply_func) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, (tuple, list)):
        return type(data)(apply_to(sample, expr, apply_func) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
    elif expr(data):
        return apply_func(data)
    else:
        return data


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
                 'found, please install them first.'):  # yapf: disable
    """A decorator factory to check if prerequisites are satisfied.

    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.

    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """A decorator to check if some python packages are installed.

    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """A decorator to check if some executable files are installed.

    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)


def deprecated_api_warning(name_dict: dict,
                           cls_name: Optional[str] = None) -> Callable:
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f'The expected behavior is to replace '
                            f'the deprecated key `{src_arg_name}` to '
                            f'new key `{dst_arg_name}`, but got them '
                            f'in the arguments at the same time, which '
                            f'is confusing. `{src_arg_name} will be '
                            f'deprecated in the future, please '
                            f'use `{dst_arg_name}` instead.')

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


def is_method_overridden(method: str, base_class: type,
                         derived_class: Union[type, Any]) -> bool:
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method


def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.

    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


def deprecated_function(since: str, removed_in: str,
                        instructions: str) -> Callable:
    """Marks functions as deprecated.

    Throw a warning when a deprecated function is called, and add a note in the
    docstring. Modified from https://github.com/pytorch/pytorch/blob/master/torch/onnx/_deprecation.py

    Args:
        since (str): The version when the function was first deprecated.
        removed_in (str): The version when the function will be removed.
        instructions (str): The action users should take.

    Returns:
        Callable: A new function, which will be deprecated soon.
    """  # noqa: E501
    from flabplatform.core.logging import print_log

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            print_log(
                f"'{function.__module__}.{function.__name__}' "
                f'is deprecated in version {since} and will be '
                f'removed in version {removed_in}. Please {instructions}.',
                logger='current',
                level=logging.WARNING,
            )
            return function(*args, **kwargs)

        indent = '    '
        # Add a deprecation note to the docstring.
        docstring = function.__doc__ or ''
        # Add a note to the docstring.
        deprecation_note = textwrap.dedent(f"""\
            .. deprecated:: {since}
                Deprecated and will be removed in version {removed_in}.
                Please {instructions}.
            """)
        # Split docstring at first occurrence of newline
        pattern = '\n\n'
        summary_and_body = re.split(pattern, docstring, 1)

        if len(summary_and_body) > 1:
            summary, body = summary_and_body
            body = textwrap.indent(textwrap.dedent(body), indent)
            summary = '\n'.join(
                [textwrap.dedent(string) for string in summary.split('\n')])
            summary = textwrap.indent(summary, prefix=indent)
            # Dedent the body. We cannot do this with the presence of the
            # summary because the body contains leading whitespaces when the
            # summary does not.
            new_docstring_parts = [
                deprecation_note, '\n\n', summary, '\n\n', body
            ]
        else:
            summary = summary_and_body[0]
            summary = '\n'.join(
                [textwrap.dedent(string) for string in summary.split('\n')])
            summary = textwrap.indent(summary, prefix=indent)
            new_docstring_parts = [deprecation_note, '\n\n', summary]

        wrapper.__doc__ = ''.join(new_docstring_parts)

        return wrapper

    return decorator


def get_object_from_string(obj_name: str):
    """Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    parts = iter(obj_name.split('.'))
    module_name = next(parts)
    # import module
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            # mmcv.ops has nms.py and nms function at the same time. So the
            # function will have a higher priority
            obj = getattr(module, part, None)
            if obj is not None and not ismodule(obj):
                break
            module_name = f'{module_name}.{part}'
        except StopIteration:
            # if obj is a module
            return module
        except ImportError:
            return None

    # get class or attribute from module
    obj = module
    while True:
        try:
            obj = getattr(obj, part)
            part = next(parts)
        except StopIteration:
            return obj
        except AttributeError:
            return None
