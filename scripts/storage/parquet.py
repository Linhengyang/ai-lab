import pyarrow.parquet as pq
import pyarrow as pa


def txt_to_parquet(txt_path, pq_path, delimiter, colname, encoding='utf-8'):
    # 读取整个文件
    with open(txt_path, "r", encoding=encoding) as f:
        content = f.read()
    # 按分隔符分割
    data = content.split(delimiter)
    # 去除空行（如开头的空字符串）
    data = [datapoint.strip() for datapoint in data if datapoint.strip()]
    pq_schema = pa.schema([ pa.field('text', pa.string()) ])

    # overwrite if exists
    with pq.ParquetWriter(pq_path, pq_schema) as writer:
        writer.write_table(
            pa.Table.from_pydict({colname:data}, pq_schema))
        

# 打印parquet文件的前几行
def print_rows_pq(pq_path, num_rows=5):
    parquet_file = pq.ParquetFile(pq_path)
    for batch in parquet_file.iter_batches(batch_size=num_rows):
        print(batch)
        break