from sre_compile import isstring
from Compiler import library
from Compiler import types


def parse_kv_args(args: list) -> dict:

    tuples = [arg.split("__") for arg in args]
    d = {}
    for t in tuples[1:]:
        assert len(t) == 2
        d[t[0]] = t[1]
    return d




def _is_secret_value_type(value):
    return isinstance(value, types.sint) or \
        isinstance(value, types.sfix) or \
        isinstance(value, types.sfloat) or \
        isinstance(value, types.sgf2n)

 

def _transform_value_to_str(value):
    if isinstance(value, types.MultiArray) or \
        isinstance(value, types.Array):
            return value.reveal_nested()
    elif isstring(value):
        return f"\"{value}\""
    elif _is_secret_value_type(value):
        return value.reveal()
    else:
        return value
        

def output_value_debug(name, value, repeat=False):

    prefix = "###OUTPUT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"repeat\": %s, \"value\": %s }"
    format_str = prefix + the_input + postfix
    format_value = _transform_value_to_str(value)
    repeat_val = None
    if repeat:
        repeat_val = "true"
    else:
        repeat_val = "false"

    library.print_ln(format_str, name, repeat_val, format_value)


def output_value(name, value, party=None):

    prefix = "###OUTPUT_FORMAT:"
    postfix = "###"
    the_input = "{ \"name\": \"%s\", \"value_length\": %s }" % (name, len(value))
    format_str = prefix + the_input + postfix

    library.print_ln(format_str)
    value.reveal_to_binary_output(player=party)

        

