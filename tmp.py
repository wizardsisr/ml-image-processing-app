import pyarrow as pa
import hickle as hkl


def convert_hickle_to_parquet(h5_file, parquet_file, chunksize=100000):

    stream = hkl.load(h5_file)
    for key in stream:
        pa_table = pa.table([stream[key].tolist()], names=[key])
        pa.parquet.write_table(pa_table, parquet_file)


convert_hickle_to_parquet('cifar10', 'cifar10.parquet', chunksize=100000)
