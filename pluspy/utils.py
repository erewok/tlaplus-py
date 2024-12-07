"""
This module contains utility functions and classes used by the compiler.
"""
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: convenient routines
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

def islower(c):
    return c in "abcdefghijklmnopqrstuvwxyz"

def isupper(c):
    return c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def isletter(c):
    return islower(c) or isupper(c)

def isnumeral(c):
    return c in "0123456789"

def isalnum(c):
    return isletter(c) or isnumeral(c)

def isnamechar(c):
    return isalnum(c) or c == "_"

def isprint(c):
    return isinstance(c, str) and len(c) == 1 and (
        isalnum(c) or c in " ~`!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")


# v is either a string, a tuple of values, or a FrozenDict.
# Return a uniform representation such that if two values should
# be equal they have the same representation.  Strings are the preferred
# representation, then tuples, then sets, then records, then nonces.
def simplify(v):
    if len(v) == 0:
        return ""
    if isinstance(v, str):
        return v

    # See if it's a record that can be converted into a tuple
    if isinstance(v, FrozenDict):
        kvs = v.d
        if set(kvs.keys()) == set(range(1, len(v) + 1)):
            t = []
            for i in range(1, len(v) + 1):
                t += [kvs[i]]
            v = tuple(t)

    # See if it's a tuple that can be converted into a string:
    if isinstance(v, tuple) and \
            all(isinstance(c, str) and len(c) == 1 for c in v):
        return "".join(v)

    return v

# Convert a value to something a little more normal and better for printing
def convert(v):
    if isinstance(v, tuple):
        return tuple([ convert(x) for x in v ])
    if isinstance(v, frozenset):
        return [ convert(y) for y in set(v) ]
    if isinstance(v, FrozenDict):
        return { convert(x):convert(y) for (x, y) in v.d.items() }
    return v

def is_tla_id(s):
    if not isinstance(s, str):
        return False
    if any(not isnamechar(c) for c in s):
        return False
    return any(isletter(c) for c in s)

# Defines a sorting order on all values
def key(v):
    if isinstance(v, bool):
        return (0, v)
    if isinstance(v, int):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    if isinstance(v, tuple):
        return (3, [key(x) for x in v])
    if isinstance(v, frozenset):
        lst = [key(x) for x in v]
        return (4, sorted(lst))
    if isinstance(v, FrozenDict):
        lst = [(key(k), key(v)) for (k, v) in v.d.items()]
        return (5, sorted(lst))
    if isinstance(v, Nonce):
        return (6, v.id)
    print(v)
    assert False

# Convert a value to a string in TLA+ format
def val_to_string(v):
    if v == "":
        return "<<>>"
    if v == frozenset():
        return "{}"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, tuple):
        result = ""
        for x in v:
            if result != "":
                result += ", "
            result += val_to_string(x)
        return "<< " + result + " >>"
    if isinstance(v, frozenset):
        lst = sorted(v, key=lambda x: key(x))
        result = ""
        for x in lst:
            if result != "":
                result += ", "
            result += val_to_string(x)
        return "{ " + result + " }"
    if isinstance(v, FrozenDict):
        return v.val_to_string()
    return str(v)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: convenient data structures
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class FrozenDict:
    def __init__(self, d):
        self.d = d

    def __str__(self):
        return "FrozenDict(" + str(self.d) + ")"

    def __hash__(self):
        hash = 0
        for x in self.d.items():
            hash ^= x.__hash__()
        return hash

    # Two dictionaries are the same if they have the same (key, value) pairs
    def __eq__(self, other):
        if not isinstance(other, FrozenDict):
            return False
        if len(self.d.keys()) != len(other.d.keys()):
            return False
        for (k, v) in self.d.items():
            if v != other.d.get(k):
                return False
        return True

    def __len__(self):
        return len(self.d.keys())

    def val_to_string(self):
        result = ""
        keys = sorted(self.d.keys(), key=lambda x: key(x))
        for k in keys:
            if result != "":
                result += ", "
            if is_tla_id(k):
                result += k + " |-> " + val_to_string(self.d[k])
            else:
                result += val_to_string(k) + " |-> " + val_to_string(self.d[k])
        return "[ " + result + " ]"


# A Hashable "nonce" (to implement CHOOSE x: x \notin S)
class Nonce:
    def __init__(self, id):
        self.id = id            # TODO: ideally a cryptographic hash

    def __str__(self):
        return "Nonce(" + str(self.id) + ")"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Nonce) and other.id == self.id
