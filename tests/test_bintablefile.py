import io
import pickle
import tempfile
from decimal import Decimal
from math import exp
from pathlib import Path
from time import time
from unittest import TestCase, skip

import numpy as np
import pandas as pd
import pyximport;

pyximport.install()
from pyxtension.streams import slist

from bintablefile import BinTableFile, create_stream_opener


class TestBinTableFile(TestCase):
    def setUp(self) -> None:
        self._tmp_dirpath = tempfile.TemporaryDirectory(suffix=None, prefix="test_BRF_", dir=None)
        self.fpath = Path(self._tmp_dirpath.name) / "temp_file.binary"
        self.dz_fpath = self.fpath.with_suffix(".dz")

        for fpath in [self.fpath, self.dz_fpath]:
            if fpath.exists():
                fpath.unlink()

    def tearDown(self) -> None:
        self._tmp_dirpath.cleanup()

    def test_structure(self):
        import ctypes

        def create_structure(fields):
            """
            Dynamically creates a ctypes.Structure subclass with given fields.

            :param fields: A list of tuples where each tuple contains the field name and ctype data type
            :return: A new ctypes.Structure subclass with the specified fields
            """

            class DynamicStructure(ctypes.Structure):
                _fields_ = fields

            return DynamicStructure

        def create_structure_array(struct_cls, n):
            """
            Creates an array of ctypes.Structure of type struct_cls with n elements.

            :param struct_cls: The ctypes.Structure subclass
            :param n: The number of elements in the array
            :return: An instance of the ctypes.Array filled with structures of type struct_cls
            """
            return (struct_cls * n)()

        # Example usage:

        # Define the list of field names and their ctypes types
        fields = [
            ('id', ctypes.c_int),
            ('value', ctypes.c_float),
            ('name', ctypes.c_char * 20)  # Example of a 20-char string
        ]

        # Dynamically create the structure and array of structures
        DynamicStruct = create_structure(fields)
        StructArray = create_structure_array(DynamicStruct, 5)  # Create an array of 5 elements

        # Example of how to use the array
        StructArray[0].id = 1
        StructArray[0].value = 3.14
        StructArray[0].name = b"Example"

        print(StructArray[0].id, StructArray[0].value, StructArray[0].name)

    def test_nominal_read_write_from_the_sameobject(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open)
        high_prec_decimal = -Decimal(1) / Decimal(7)
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        e = Decimal.from_float(ef)
        records = [(MAX_INT, True, ef, high_prec_decimal), (MIN_INT, False, -1.1, e)]
        expected_records = [
            (MAX_INT, True, ef, Decimal("-0.142857142857142857")),
            (MIN_INT, False, -1.1, Decimal("2.71828182845904509")),
        ]
        record_file.extend(records)
        record_file.flush()

        extracted_records = list(record_file)
        self.assertListEqual(expected_records, extracted_records)

    def test_nominal_write_file_read_file(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open)
        high_prec_decimal = -Decimal(1) / Decimal(7)
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        e = Decimal.from_float(ef)
        records = [(MAX_INT, True, ef, high_prec_decimal), (MIN_INT, False, -1.1, e)]
        expected_records = [
            (MAX_INT, True, ef, Decimal("-0.142857142857142857")),
            (MIN_INT, False, -1.1, Decimal("2.71828182845904509")),
        ]
        record_file.extend(records)
        record_file.flush()

        new_record_file = BinTableFile(self.fpath)
        extracted_records = list(new_record_file)
        self.assertListEqual(expected_records, extracted_records)


    def test_nominal_write_file_with_metadata_read_file(self):
        record_format = (int, bool, float, Decimal)
        metadata = {"b": "test_mock", "a": 1, "c": 2.0, "d": [1, 2]}

        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open, metadata=metadata)
        high_prec_decimal = -Decimal(1) / Decimal(7)
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        e = Decimal.from_float(ef)
        records = [(MAX_INT, True, ef, high_prec_decimal), (MIN_INT, False, -1.1, e)]
        expected_records = [
            (MAX_INT, True, ef, Decimal("-0.142857142857142857")),
            (MIN_INT, False, -1.1, Decimal("2.71828182845904509")),
        ]
        record_file.extend(records)
        record_file.flush()

        # Open and read the same file
        new_record_file = BinTableFile(self.fpath)
        extracted_records = list(new_record_file)
        self.assertListEqual(expected_records, extracted_records)
        self.assertDictEqual(metadata, new_record_file.metadata)

    def test_decimal_is_rounded(self):
        record_format = (Decimal,)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=("mock_column",), opener=open)
        high_prec_decimal = Decimal(2) / Decimal(3)
        records = [(high_prec_decimal,), (-high_prec_decimal,)]
        expected_records = [(Decimal("0.666666666666666667"),), (Decimal("-0.666666666666666667"),)]
        record_file.extend(records)
        record_file.flush()
        extracted_records = list(iter(record_file))
        self.assertListEqual(expected_records, extracted_records)

    def test_low_val_decimal_is_upscaled(self):
        """
        We store 18 significant digits, so the # of digits after point can go further of 18
        """
        record_format = (Decimal,)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=("mock_column",), opener=open)
        high_prec_decimal = Decimal("0.00000000001234567890123456789012345")
        records = [(high_prec_decimal,), (-high_prec_decimal,)]
        expected_records = [(Decimal("0.0000000000123456789012345679"),), (Decimal("-0.0000000000123456789012345679"),)]
        record_file.extend(records)
        record_file.flush()
        extracted_records = list(iter(record_file))
        self.assertListEqual(expected_records, extracted_records)

    def test_high_val_decimal_is_downscaled(self):
        """
        We store 18 significant digits, so the # of digits can go further of 18 if number ends with 0-es
        """
        record_format = (Decimal,)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=("mock_column",), opener=open)
        high_prec_decimal = Decimal("12345678901234567800000")
        records = [(high_prec_decimal,), (-high_prec_decimal,)]
        expected_records = [(Decimal("12345678901234567800000"),), (Decimal("-12345678901234567800000"),)]
        record_file.extend(records)
        record_file.flush()
        extracted_records = list(record_file)
        self.assertListEqual(expected_records, extracted_records)

    def test_overflow_decimal_is_truncated(self):
        """
        We store 18 significant digits, so the # of digits can go further of 18 if number ends with 0-es
        """
        record_format = (Decimal,)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=("mock_column",), opener=open)
        high_prec_decimal = Decimal("18446744073709551616000")
        records = [(high_prec_decimal,), (-high_prec_decimal,)]
        # Observe, the last 2 digits (16) are zeroed as it would mean we store 20 digits instead of 18
        expected_records = [(Decimal("18446744073709551600000"),), (Decimal("-18446744073709551600000"),)]
        record_file.extend(records)
        record_file.flush()
        extracted_records = list(record_file)
        self.assertListEqual(expected_records, extracted_records)

    def test_extend(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend([records[0]])
        record_file.flush()
        first_extracted_record = list(record_file)
        self.assertListEqual([records[0]], first_extracted_record)
        record_file.extend([records[1]])
        record_file.flush()
        extracted_records = list(record_file)
        self.assertListEqual(records, extracted_records)

    def test_len(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open)
        e = Decimal(exp(1))
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend(records)
        record_file.flush()
        record_size = len(record_file)
        self.assertEqual(len(records), record_size)

    def test_get_last_record(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend(records)
        record_file.flush()
        last_record = record_file[-1]
        self.assertEqual(records[-1], last_record)

    def test_slice(self):
        record_format = (int,)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=("int",), opener=open)
        records = [(i,) for i in range(10)]
        record_file.extend(records)
        record_file.flush()
        sliced_records = list(record_file[2:5])
        self.assertListEqual(sliced_records, records[2:5])
        self.assertListEqual(list(record_file[2:-2]), records[2:-2])
        self.assertListEqual(list(record_file[:5]), records[:5])
        self.assertListEqual(list(record_file[5:]), records[5:])

    def test_retrieve_header(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "float_col", "Decimal_col")
        metadata = {"mock": "test_mock"}
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, metadata=metadata,
                                   opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend(records)
        record_file.flush()
        record_file_no_columns = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                              opener=open)
        self.assertTupleEqual(record_file_no_columns.columns, columns)
        self.assertTupleEqual(record_file_no_columns.record_format, record_format)
        self.assertDictEqual(record_file_no_columns.metadata, metadata)

    def test_retrieve_header_without_metadata(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "float_col", "Decimal_col")
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend(records)
        record_file.flush()
        record_file_no_columns = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                              opener=open)
        self.assertTupleEqual(record_file_no_columns.columns, columns)
        self.assertTupleEqual(record_file_no_columns.record_format, record_format)
        self.assertIsNone(record_file_no_columns.metadata)

    def test_retrieve_header_with_big_metadata(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "float_col", "Decimal_col")
        big_value = [str(i) for i in range(4096)]
        metadata = {"data": big_value}
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, metadata=metadata,
                                   opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, e)]
        record_file.extend(records)
        record_file.flush()
        record_file_no_columns = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                              opener=open)
        self.assertTupleEqual(record_file_no_columns.columns, columns)
        self.assertTupleEqual(record_file_no_columns.record_format, record_format)
        self.assertDictEqual(record_file_no_columns.metadata, metadata)

    def test_init_file_with_incomplete_file(self):
        corrupt_header_data = b'eSAMbin\x03\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06' \
                              b'\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x17\x00\x00' \
                              b'\x00\x00\x00\x00\x00time,oint,in'
        with open(self.fpath, 'wb') as f:
            f.write(corrupt_header_data)

        with self.assertRaises(EOFError):
            record_file = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                       opener=open)

    def test_init_file_with_wrong_version(self):
        corrupt_header_data = b'eSAMbin\x02\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06' \
                              b'\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x17\x00\x00' \
                              b'\x00\x00\x00\x00\x00time,oint,in'
        with open(self.fpath, 'wb') as f:
            f.write(corrupt_header_data)

        with self.assertRaises(ValueError):
            record_file = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                       opener=open)

    def test_init_file_with_memory_stream(self):
        buf = io.BytesIO()
        opener = create_stream_opener(buf)
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"), opener=opener)
        high_prec_decimal = -Decimal(1) / Decimal(7)
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        e = Decimal.from_float(ef)
        records = [(MAX_INT, True, ef, high_prec_decimal), (MIN_INT, False, -1.1, e)]
        expected_records = [
            (MAX_INT, True, ef, Decimal("-0.142857142857142857")),
            (MIN_INT, False, -1.1, Decimal("2.71828182845904509")),
        ]
        record_file.extend(records)
        record_file.flush()
        bin_data = buf.getvalue()
        self.assertEqual(bin_data[0:8], b'eSAMbin\x03')
        buf = io.BytesIO(bin_data)
        opener = create_stream_opener(buf)
        record_file = BinTableFile(self.fpath, record_format=None, columns=None, opener=opener)
        self.assertTupleEqual(record_file.columns, ("int", "bool", "float", "Decimal"))
        self.assertTupleEqual(record_file.record_format, (int, bool, float, Decimal))
        self.assertListEqual(list(record_file), expected_records)

    def test_retrieve_header_with_corrupted_file(self):
        corrupt_header_data = b'eSAMbin\x02\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06' \
                              b'\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x17\x00\x00' \
                              b'\x00\x00\x00\x00\x00time,oint,int{}'
        with open(self.fpath, 'wb') as f:
            f.write(corrupt_header_data)

        with self.assertRaises(ValueError):
            record_file = BinTableFile(self.fpath, record_format=None, columns=None, metadata=None,
                                       opener=open)
            metadata = record_file.metadata

    def test_as_named_tuple_stream(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "def_", "id")
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, opener=open)
        e = Decimal("2.71828182845904509")
        records = [(1, True, 0.1, e), (2, False, -1.1, -e)]
        record_file.extend(records)
        record_file.flush()
        l = list(record_file.as_named_tuple_stream())
        self.assertEqual(l[0].id, e)
        self.assertEqual(l[-1].id, -e)

    def test_append_after_flush(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "def_", "id")
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, opener=open)
        e = Decimal("2.71828182845904509")
        records = slist([(1, True, 0.1, e), (2, False, -1.1, -e)])
        records.map(record_file.append).toList()
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = list(record_file)
        self.assertListEqual(records, extracted_records)

    def test_append_no_flush_small_buffer(self):
        record_format = (int, bool, float, Decimal)
        columns = ("int_col", "bool_col", "def_", "id")
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=1,
                                   opener=open)
        e = Decimal("2.71828182845904509")
        records = slist([(1, True, 0.1, e), (2, False, -1.1, -e)])
        records.map(record_file.append).toList()
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = list(record_file)
        self.assertListEqual(records, extracted_records)

    def test_extend_after_append_buffer_sharing(self):
        record_format = (int,)
        columns = ("int_col",)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file.append((0,))
        record_file.extend([(i,) for i in range(1, 10)])
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = [t[0] for t in list(record_file)]
        self.assertListEqual(list(range(10)), extracted_records)

    def test_iadd(self):
        record_format = (int,)
        columns = ("int_col",)
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file.append((0,))
        record_file += ((i,) for i in range(1, 10))
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = [t[0] for t in list(record_file)]
        self.assertListEqual(list(range(10)), extracted_records)

    def test_truncate_nominal(self):
        record_format = (int,)
        columns = ("int_col",)
        truncate_N = 5
        expected_records = [(i,) for i in range(10)]
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file += expected_records
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        record_file.truncate(truncate_N)
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = list(record_file)
        self.assertListEqual(expected_records[:truncate_N], extracted_records)

    def test_truncate_to_more_than_exists(self):
        record_format = (int,)
        columns = ("int_col",)
        expected_records = [(i,) for i in range(10)]
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file += expected_records
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        record_file.truncate(11)
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = list(record_file)
        self.assertListEqual(expected_records, extracted_records)

    def test_truncate_to_0(self):
        record_format = (int,)
        columns = ("int_col",)
        expected_records = [(i,) for i in range(10)]
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file += expected_records
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        record_file.truncate(0)
        record_file = BinTableFile(self.fpath, opener=open)
        extracted_records = list(record_file)
        self.assertListEqual([], extracted_records)

    def test_truncate_consistency(self):
        record_format = (int,)
        columns = ("int_col",)
        expected_records = [(i,) for i in range(10)]
        record_file = BinTableFile(self.fpath, record_format=record_format, columns=columns, buf_size=10000,
                                   opener=open)
        record_file += expected_records
        record_file.flush()
        record_file = BinTableFile(self.fpath, opener=open)
        record_file.truncate(5)
        record_file = BinTableFile(self.fpath, opener=open)
        record_file.append((0,))
        record_file.flush()
        extracted_records = list(record_file)
        self.assertListEqual(expected_records[:5] + [(0,)], extracted_records)

    def test_save_df_with_idzip(self):
        try:
            import idzip
        except ImportError:
            return
        records = [(1, True, 0.1), (2, False, -0.1), (3, True, 0.2)]
        df = pd.DataFrame(records, columns=["int_col", "bool_col", "float_col"])
        idzip.compressor.COMPRESSION_LEVEL = 1
        path = str(self.dz_fpath)
        # IDZipWriter enforces 'dz' extension
        # To disable this, set idzip.compressor.IdzipWriter.enforce_extension = False
        BinTableFile.save_df(df, fpath=path, opener=idzip.open)

        record_file = BinTableFile(path, opener=idzip.open)
        self.assertEqual(3, len(record_file))  # The length works with idzip
        self.assertTupleEqual(record_file[-1], records[-1])  # The last record is correct
        extracted_records = list(record_file)
        self.assertListEqual(records, extracted_records)

    def test_save_df_with_np_ints(self):
        buf = io.BytesIO()
        write_opener = create_stream_opener(buf)

        records = [(1, 1, 1, 1., 1.), (2, 2, 2, 2., 2.), (3, 3, 3, 3., 3.)]
        df = pd.DataFrame(records, columns=["int_col", "int8_col", "int64_col", "float_col", "float64_col"])
        df = df.astype({"int8_col": "int8", "int64_col": "int64", "float64_col": "float64"})
        BinTableFile.save_df(df, fpath='', opener=write_opener)
        bin_data = buf.getvalue()

        written_buf = io.BytesIO(bin_data)
        read_opener = create_stream_opener(written_buf)
        record_file = BinTableFile(self.fpath, record_format=None, columns=None, opener=read_opener)
        read_df = record_file.as_df()
        self.assertListEqual(list(df.dtypes), list(read_df.dtypes))

        read_opener = create_stream_opener(written_buf)
        record_file = BinTableFile(self.fpath, record_format=None, columns=None, opener=read_opener)
        read_records = list(record_file)
        self.assertEqual(3, len(read_records))
        self.assertTupleEqual(read_records[-1], read_records[-1])
        self.assertIsInstance(read_records[-1][1], np.int8)
        self.assertIsInstance(read_records[-1][2], np.int64)
        self.assertIsInstance(read_records[-1][3], np.float64)
        self.assertIsInstance(read_records[-1][4], np.float64)
        self.assertEqual(read_records[-1][1], np.int8(3))

    def test_as_df_without_decimal(self):
        record_format = (int, bool, float,)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float",))
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        records = [(MAX_INT, True, ef,), (MIN_INT, False, -1.1,)]
        expected_records = [
            [MAX_INT, True, ef],
            [MIN_INT, False, -1.1],
        ]
        record_file.extend(records)
        record_file.flush()
        record_file = BinTableFile(self.fpath)
        df = record_file.as_df()
        self.assertListEqual(expected_records, df.values.tolist())

    @skip("Decimal not implemented yet in as_df")
    def test_as_df_with_decimal(self):
        record_format = (int, bool, float, Decimal)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float", "Decimal"))
        high_prec_decimal = -Decimal(1) / Decimal(7)
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        e = Decimal.from_float(ef)
        records = [(MAX_INT, True, ef, high_prec_decimal), (MIN_INT, False, -1.1, e)]
        expected_records = [
            [MAX_INT, True, ef, Decimal("-0.142857142857142857")],
            [MIN_INT, False, -1.1, Decimal("2.71828182845904509")],
        ]
        record_file.extend(records)
        record_file.flush()
        record_file = BinTableFile(self.fpath)
        df = record_file.as_df()
        self.assertListEqual(expected_records, df.values.tolist())

    def test_as_df_no_dec(self):
        record_format = (int, bool, float)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float"))
        MIN_INT = -(2 ** 63)
        MAX_INT = 2 ** 63 - 1
        ef = exp(1)
        records = [(MAX_INT, True, ef,), (MIN_INT, False, -1.1,)]
        expected_records = [
            [MAX_INT, True, ef, ],
            [MIN_INT, False, -1.1, ],
        ]
        record_file.extend(records)
        record_file.flush()
        record_file = BinTableFile(self.fpath)
        df = record_file.as_df()
        self.assertListEqual(expected_records, df.values.tolist())

    def test_default_opener_opens_idzip(self):
        try:
            import idzip
        except ImportError:
            return

        records = [(1, True, 0.1), (2, False, -0.1), (3, True, 0.2)]
        df = pd.DataFrame(records, columns=["int_col", "bool_col", "float_col"])
        idzip.compressor.COMPRESSION_LEVEL = 1
        path = str(self.dz_fpath)
        # IDZipWriter enforces 'dz' extension
        # To disable this, set idzip.compressor.IdzipWriter.enforce_extension = False
        BinTableFile.save_df(df, fpath=path, opener=idzip.open)

        record_file = BinTableFile(path)  # Here we expect idzip.open to be used
        extracted_records = list(record_file)
        self.assertListEqual(records, extracted_records)

    def test_performance_read_write(self):
        record_format = (int, bool, float)
        record_file = BinTableFile(self.fpath, record_format=record_format,
                                   columns=("int", "bool", "float"))
        N = 1_000
        records = [(i, True, i + 0.1) for i in range(N)]
        t0 = time()
        record_file.extend(records)
        record_file.flush()
        dt = time() - t0
        print(f"Writing {N} records took {dt:.4f} seconds")
        fsize = self.fpath.stat().st_size
        print(f"File size: {fsize} bytes")
        t0 = time()
        record_file = BinTableFile(self.fpath)
        extracted_records = list(record_file)
        dt = time() - t0
        self.assertEqual(len(extracted_records), N)
        print(f"Reading {N} records took {dt:.4f} seconds")

    def test_performance_df_read_write(self):
        N = 1_000
        records = [(i, True, i + 0.1) for i in range(N)]
        df = pd.DataFrame(records, columns=["int_col", "bool_col", "float_col"])
        t0 = time()
        BinTableFile.save_df(df, fpath=self.fpath, buf_size=4096)
        dt = time() - t0
        print(f"Writing DF {N} records took {dt:.4f} seconds")
        t0 = time()
        record_file = BinTableFile(self.fpath)

        df = record_file.as_df()
        dt = time() - t0
        print(f"Reading {N} records took {dt:.4f} seconds")
        self.assertEqual(len(df.index), N)
        t0 = time()
        with open(self.fpath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(df, f)
        dt = time() - t0
        print(f"Pickling {N} records took {dt:.4f} seconds")
        t0 = time()
        with open(self.fpath.with_suffix(".pkl"), "rb") as f:
            unpickled_df = pickle.load(f)
        dt = time() - t0
        print(f"Unpickling {N} records took {dt:.4f} seconds")
