from collections import defaultdict
from typing import Sequence

import pyarrow as pa


def to_union_array(arr: Sequence[int | float | bool | str | bytes | None]):
    type_map = defaultdict(list)
    offsets = []
    for item in arr:
        item_t = type(item)
        offsets.append(len(type_map[item_t]))
        type_map[item_t].append(item)

    _types = list(type_map)
    types = pa.array((_types.index(type(i)) for i in arr), type=pa.int8())
    uarr = pa.UnionArray.from_dense(
        types,
        pa.array(offsets, type=pa.int32()),
        list(map(pa.array, type_map.values())),
    )
    return uarr


## TESTS


def test_union_array():
    any_array = [1, 2, None, "a", "b", 2.71, 3.14, "d", "e", True]
    uarr = to_union_array(any_array)
    assert any_array == [i.as_py() for i in uarr]
