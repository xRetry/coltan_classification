import functools
import numpy as np


def verification(*shapes):
    """
    Function decorator which check the size of dimensions of function arguments. Works with int, float and np.ndarray.
    Each unique dimension needs to be specified by an unique letter, or a number indicating its size.
    Sizes of dimensions with same letters are compared.
    int and float need to specified with an empty string.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Getting argument names of function
            arg_names = func.__code__.co_varnames
            n_args = func.__code__.co_argcount
            # Checking shapes size
            if n_args != len(shapes):
                raise AttributeError('Amount of attributes and shapes must agree! ({} != {})'.format(n_args, len(shapes)))
            # Adding args to kwargs
            kwargs.update({arg_names[i]:args[i] for i in range(len(args))})
            # Checking argument types and shapes
            sizes = {}
            for i in range(n_args):
                # Skip if None
                if shapes[i] is None:
                    continue
                # Check if argument should be numeric
                if shapes[i] == '':
                    if not isinstance(kwargs[arg_names[i]], (int, float)):
                        raise TypeError('Attribute {} needs to be int or float!'.format(arg_names[i]))
                # Otherwise check if argument is numpy array with correct shape
                else:
                    if not isinstance(kwargs[arg_names[i]], np.ndarray):
                        raise TypeError('Invalid type for {} ({} != np.ndarray)'.format(arg_names[i], type(kwargs[arg_names[i]])))
                    # Check if the amount of dimensions is correct
                    shape = kwargs[arg_names[i]].shape
                    if len(shape) != len(shapes[i]):
                        raise TypeError('Incorrect dimensions of argument {}: expected={}, received={}'.format(arg_names[i], len(shapes[i]), len(shape)))
                    # Check if the size of each dimension is correct
                    for j, alias in enumerate(shapes[i]):
                        # Compare size to fixed size if value is provided
                        if alias.isdigit():
                            if shape[j] != int(alias):
                                raise TypeError('Incorrect size of {}[{}]! (expected: {}, received: {})'.format(arg_names[i], j, alias, shape[j]))
                        # Otherwise compare with other dimensions who should have the same size
                        else:
                            size = sizes.get(alias)
                            if size is None:
                                sizes[alias] = shape[j]
                            else:
                                if size != shape[j]:
                                    raise TypeError('Incorrect size of {}[{}]! (expected: {}, received: {})'.format(arg_names[i], j, size, shape[j]))
            # Input checks passed, calling function
            result = func(**kwargs)
            return result
        return wrapper
    return decorator


if __name__ == '__main__':
    pass
