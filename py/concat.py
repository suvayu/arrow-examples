from itertools import chain, tee
from typing import Iterable, TypeVar
import pyarrow as pa
import pytest


def merge_schemas(schemas: Iterable[pa.Schema]) -> pa.Schema:
    # overwrites earlier keys
    s1, s2 = tee(schemas)
    fields = list(dict.fromkeys(chain.from_iterable(s1)))
    metadata = {k: v for k, v in chain.from_iterable(sc.metadata.items() for sc in s2)}
    return pa.schema(fields, metadata)


RecBatchTable_t = TypeVar("RecBatchTable_t", pa.RecordBatch, pa.Table)


def replace_schema(batch: RecBatchTable_t, schema) -> RecBatchTable_t:
    rows = batch.shape[0]
    data = [
        batch[field.name] if field in batch.schema else pa.nulls(rows).cast(field.type)
        for field in schema
    ]
    match batch:
        case pa.RecordBatch():
            return pa.record_batch(data, schema=schema)
        case pa.Table():
            return pa.table(data, schema=schema)
        case _:
            raise ValueError(f"{type(batch)}: unknown type")


def concat_w_missing(data: Iterable[RecBatchTable_t]) -> RecBatchTable_t:
    d1, d2, d3 = tee(data, 3)
    item = next(d1)
    schema = merge_schemas(batch.schema for batch in d2)
    match item:
        case pa.RecordBatch():
            return pa.concat_batches(replace_schema(batch, schema) for batch in d3)
        case pa.Table():
            return pa.concat_tables(replace_schema(batch, schema) for batch in d3)
        case _:
            raise ValueError(f"{type(item)}: unknown type")


## TESTS


def make_arrays(*, tables: bool = False):
    iarr = pa.array([i for i in range(5)])
    sarr = pa.array("foo bar baz bla what!".split())
    barr = pa.array([True, False, True, None, False])
    farr = pa.array([i + 0.1 for i in range(5)])

    # rec1 = pa.RecordBatch.from_arrays([iarr, farr, sarr], "f0 f1 f2".split())
    # rec2 = pa.RecordBatch.from_arrays([farr, sarr, barr], "f1 f2 f3".split())
    # recs = pa.concat_batches([rec1, rec2])

    r1 = pa.record_batch(
        {"f0": iarr, "f1": farr, "f2": barr},
        metadata={"f0": "integer", "f1": "number", "f2": "boolean"},
    )
    r2 = pa.record_batch(
        {"f0": iarr, "f3": sarr, "f2": barr},
        metadata={"f0": "integer", "f3": "string", "f2": "boolean"},
    )
    if tables:
        t1 = pa.Table.from_batches([r1, r1])
        t2 = pa.Table.from_batches([r2, r2])
        return t1, t2
    else:
        return r1, r2


@pytest.fixture
def records():
    return make_arrays()


@pytest.fixture
def tables():
    return make_arrays(tables=True)


def test_concat_record_batches(records):
    rec = concat_w_missing(records)
    assert rec.shape == (10, 4)


def test_concat_tables(tables):
    tbl = concat_w_missing(tables)
    assert tbl.shape == (20, 4)
