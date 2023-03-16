import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import hickle as hkl
import numpy as np


def convert_hickle_to_parquet(h5_file, parquet_file, chunksize=100000):

    stream = hkl.load(h5_file)
    schema = pa.schema({'training_data': pa.uint32(),
                        'test_data': pa.uint32(),
                        'training_labels': pa.uint32(),
                        'test_labels': pa.uint32()})
    for key in stream:
        pa_table = pa.table([stream[key]], schema=schema)
        pa.parquet.write_table(pa_table, parquet_file)


convert_hickle_to_parquet('cifar10', 'cifar10.parquet', chunksize=100000)
