import sys
import numpy as np

PLACE_HOLDER = '_'

class SequencePattern:
    def __init__(self, sequence, support):
        self.sequence = []
        for s in sequence:
            self.sequence.append(list(s))
        self.support = support

    def append(self, p):
        if p.sequence[0][0] == PLACE_HOLDER:
            first_e = p.sequence[0]
            first_e.remove(PLACE_HOLDER)
            self.sequence[-1].extend(first_e)
            self.sequence.extend(p.sequence[1:])
        else:
            self.sequence.extend(p.sequence)
        self.support = min(self.support, p.support)

def prefixSpan(pattern, S, minSup):
    patterns = []
    f_list = frequentItems(pattern, S, minSup)
    for item in f_list:
        p = SequencePattern(pattern.sequence, pattern.support)
        p.append(item)
        patterns.append(p)

        P_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, P_S, minSup)
        patterns.extend(p_patterns)
    return patterns

def frequentItems(pattern, S, minSup):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    if len(pattern.sequence) != 0:
        last_e = pattern.sequence[-1]
    else:
        last_e = []
    for s in S:
        #class 1
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1:]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1

        #class 2
        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]

        #class 3
        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1

    f_list.extend([SequencePattern([[PLACE_HOLDER, k]], v)
                    for k, v in _items.items()
                    if v >= minSup])
    f_list.extend([SequencePattern([[k]], v)
                   for k, v in items.items()
                   if v >= minSup])
    sorted_list = sorted(f_list, key=lambda p: p.support)
    return sorted_list  

def build_projected_database(S, pattern):
    """
    suppose S is projected database base on pattern's prefix,
    so we only need to use the last element in pattern to
    build projected database
    """
    p_S = []
    last_e = pattern.sequence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.sequence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break

            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1:]
                else:
                    p_s = s[e_index:]
                    index = element.index(last_item)
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break
        if len(p_s) != 0:
            p_S.append(p_s)

    return p_S

def fpGrowth(dataSet, minSup):
    patterns = prefixSpan(SequencePattern([], sys.maxsize), dataSet, minSup)
    return patterns

