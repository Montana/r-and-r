import re

def fuzzy_match(a, b):
    assert isinstance(a, str), "The first argument must be a string."
    if isinstance(b, list):
        return max(fuzzy_match(a, m) for m in b)
    else:
        assert isinstance(b, str), "The second argument must be a string or a list of strings."

    a = re.sub(r'[^a-z0-9 ]+', '', a.lower().strip())
    b = re.sub(r'[^a-z0-9 ]+', '', b.lower().strip())

    a_set = set(a.split())
    b_set = set(b.split())

    return int(a_set.issubset(b_set) or b_set.issubset(a_set) or a in b or b in a)
