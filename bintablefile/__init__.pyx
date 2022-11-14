import gzip
import io
import itertools
import json
import struct
from decimal import Decimal
from pathlib import Path
from typing import (
    IO, Iterable,
    Optional,
    Generator,
    Tuple,
    Union,
    Iterator,
    Any,
    TypeVar,
    Dict,
    Type,
    NamedTuple, )

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator
from pydantic import PositiveInt, conint, validate_arguments
from pyxtension import validate

_K = TypeVar('_K')
RecType = TypeVar("RecType", int, Decimal, float, bool, np.int64, np.float64, np.bool_)
RecTypeType = Union[Type[int], Type[float], Type[Decimal], Type[bool], Type[np.int64], Type[np.float64], Type[np.bool_]]
Record = Tuple[RecType, ...]

def open_by_extension(file: Path, *args, **kwargs) -> IO:
    if not isinstance(file, Path):
        file = Path(file)
    if file.suffix == ".gz":
        return gzip.open(file, *args, **kwargs)
    elif file.suffix == ".dz":
        from idzip import IdzipFile
        return IdzipFile(file, *args, **kwargs)
    else:
        return open(file, *args, **kwargs)


class Header(BaseModel):
    columns: Tuple[str, ...]
    types: Tuple[RecTypeType, ...]
    record_size: PositiveInt
    records_nr: conint(ge=-1)  # by convention, if it's -1, then the #records will be estimated from filesize
    metadata: Optional[Dict[str, Any]] = None

    # annotations are necessary for Cython to work
    # https://stackoverflow.com/questions/56079419/using-dataclasses-with-cython
    __annotations__ = {
        'columns':     Tuple[str, ...],
        'types':       Tuple[RecTypeType, ...],
        'record_size': PositiveInt,
        'records_nr':  conint(ge=-1),  # by convention, if it's -1, then the #records will be estimated from filesize
        'metadata':    Optional[Dict[str, Any]],
    }

    @validator('types')
    def _columns_match_types(cls, v, values, **kwargs):
        if len(v) != len(values['columns']):
            raise ValueError("Header.columns and Header.types must have same length")
        return v

    class Config:
        allow_mutation = False
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def encode(self) -> bytes:
        validate(',' not in ''.join(self.columns))
        size_format = "<Q"  # unsigned long long
        version_format = "<B"  # unsigned char
        encoded_columns = bytes(",".join(self.columns), encoding="ascii")
        encoded_format = bytes(",".join(t.__name__ for t in self.types), encoding="ascii")
        signature = bytes(BinTableFile.SIGNATURE, encoding="ascii")
        binary_format = bytes(BinTableFile.BINARY_FORMAT, encoding="ascii")
        version = struct.pack(version_format, BinTableFile.VERSION)
        record_len = BinTableFile._compute_record_size_in_bytes(self.types)
        record_size = struct.pack(size_format, record_len)
        records_nr = struct.pack("<q", self.records_nr)  # long long as it might be -1

        metadata = self.metadata or {}
        encoded_metadata = json.dumps(metadata).encode('utf-8')
        metadata_size = struct.pack(size_format, len(encoded_metadata))
        encoded_columns_size = struct.pack(size_format, len(encoded_columns))
        encoded_format_size = struct.pack(size_format, len(encoded_format))

        header = (
                signature + binary_format + version + record_size + records_nr +
                encoded_columns_size + encoded_format_size + metadata_size +
                encoded_columns + encoded_format + encoded_metadata
        )

        return header


class BinTableFile(list):
    """
    Binary Table File format - is basically a typed CSV, but binary and indexed,
        supporting only a subset of the types: int, float, bool, Decimal.
    The table is backed by a file on disk, and is NOT memory-mapped.
    This class just interfaces the reading/writing of the file through a `list` interface.
    All the data is indexed and can be accessed as the list of records, where each record is a tuple of values.
    It's main purpose are big files that don't fit into memory, or the file that are read partially.
    It can efficiently slice through the file and read just last N records, or just the first N records.
    The footprint of the file is quite small as the data is binary encoded.
    """
    SIZE_BY_TYPE = {int: 8, bool: 1, float: 8, Decimal: 9}
    DECIMAL_STORE_PREC = 18
    SCALE_FACTOR = 10 ** DECIMAL_STORE_PREC
    TYPE_DECODING: Dict[str, RecType] = {k.__name__: k for k in (int, float, Decimal, bool)}
    BUILTIN_TYPES_MAP = {np.int64: int, np.float64: float, np.bool_: bool}
    BINARY_FORMAT = 'bin'
    VERSION = 2

    SIGNATURE = 'eSAM'
    BUF_SZ = 4096

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, fpath: Path, record_format: Optional[Tuple[RecTypeType, ...]] = None,
                 columns: Optional[Tuple[str, ...]] = None, metadata: Optional[Dict[str, Any]] = None,
                 buf_size: int = 4 * 1024, opener: Any = open_by_extension, records_nr: int = -1):
        """
        :param record_format: string describing structure of expecting record
        :param opener: Callable[..., IO] should accept next parameters ([filename], mode:str, newline:str)
                The filename argument can be an actual filename (a str or bytes object),
                or an existing file object to read from or write to.
        :param records_nr: number of records in file.
                If -1, then this is a write-able file, and the size will be estimated from file size;
                otherwise, if records_nr is not -1, then the file is read-only
        """
        if columns is not None and record_format is not None:
            self._columns = columns
            self._record_format = record_format
            self._metadata = metadata
            record_size = self._compute_record_size_in_bytes(self._record_format)
            self._header = Header(columns=columns, types=record_format, record_size=record_size,
                                  records_nr=records_nr, metadata=metadata)
            self._header_data = self._header.encode()
            self._header_sz = len(self._header_data)
        else:
            self._header, self._header_sz = self._read_header(fpath, opener)
            self._columns = self._header.columns
            self._record_format = self._header.types
            self._metadata = self._header.metadata

        self._format = self._build_pack_format(self._record_format)

        self._namedtuple_class_value: Optional[Type[NamedTuple]] = None

        self._fpath = fpath
        self._opener = opener
        self._fh = None
        self._record_size = self._header.record_size
        self._buf_size = self._record_size * (buf_size // self._record_size)
        self._buffer = bytearray()

    @staticmethod
    def _read_header(fpath: Path, opener: Any = gzip.open) -> Tuple[Header, int]:
        with opener(fpath, mode="rb") as file:
            header_data = file.read(BinTableFile.BUF_SZ)
            signature = header_data[:4].decode(encoding="ascii")
            binary_format = header_data[4:7].decode(encoding="ascii")
            validate(signature == BinTableFile.SIGNATURE)
            validate(binary_format == BinTableFile.BINARY_FORMAT)
            NUM_H_SZ = 1 + 8 * 5
            NUM_H_START = 7
            VAR_H_START = NUM_H_START + NUM_H_SZ
            version, record_size, records_nr, name_header_sz, type_header_sz, meta_sz = struct.unpack_from(
                '<BQqQQQ', header_data[NUM_H_START:NUM_H_START + NUM_H_SZ]
            )
            validate(version == BinTableFile.VERSION)

            header_sz = (VAR_H_START + name_header_sz + type_header_sz + meta_sz)
            if len(header_data) < header_sz:
                rest_buf = file.read(header_sz - len(header_data))
                header_data += rest_buf
                validate(
                    len(header_data) == header_sz,
                    f"Expected {header_sz} bytes to be read but we got only {len(header_data)} bytes",
                    EOFError
                )

            name_header = header_data[VAR_H_START:VAR_H_START + name_header_sz]
            columns = tuple(name_header.decode(encoding="ascii").split(','))
            start_index = VAR_H_START + name_header_sz
            end_index = start_index + type_header_sz
            encoded_types = header_data[start_index: end_index].decode(encoding="ascii").split(',')
            decoded_types: Tuple[RecType, ...] = tuple(BinTableFile.TYPE_DECODING[t] for t in encoded_types)

            validate(len(columns) == len(decoded_types), "#columns != #types", ValueError)

            metadata = json.loads(header_data[end_index: end_index + meta_sz])
            header = Header(columns=columns, types=decoded_types, record_size=record_size,
                            records_nr=records_nr, metadata=metadata)
            return header, header_sz

    @property
    def columns(self) -> Tuple[str]:
        return self._columns

    @property
    def _namedtuple_class(self) -> Type[NamedTuple]:
        if not self._namedtuple_class_value:
            col_type_tuples = [(self._columns[i], self._record_format[i]) for i in range(len(self._columns))]
            self._namedtuple_class_value: Optional[Type[NamedTuple]] = NamedTuple(
                "NamedTuple_%X" % abs(hash(self._header_data)), col_type_tuples
            )
        return self._namedtuple_class_value

    @property
    def record_format(self) -> Tuple[RecTypeType, ...]:
        return self._record_format

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @staticmethod
    def _compute_record_size_in_bytes(record_format: Tuple[RecTypeType, ...]) -> PositiveInt:
        total_bytes = 0
        for _type in record_format:
            if issubclass(_type, Decimal):
                total_bytes += 9
            elif issubclass(_type, (bool, np.bool_)):
                total_bytes += 1
            elif issubclass(_type, (int, float, np.int64, np.float64)):
                # encode as "long long/double/unsigned long long" [use q/d/Q for pack() ]
                total_bytes += 8
            else:
                raise ValueError(f"Unknown type [{_type!s}] for record")
        return total_bytes

    @classmethod
    def _to_18prec_parts(cls, high_prec_decimal: Decimal) -> Tuple[int, int]:
        fp = cls.DECIMAL_STORE_PREC - high_prec_decimal.adjusted() - 1
        scaled_int = round(high_prec_decimal.scaleb(fp))
        return scaled_int, fp

    @classmethod
    def _to_encoded_tpl(cls, rt: RecType) -> Tuple[RecType, ...]:
        if isinstance(rt, (bool, int, float)):
            return (rt,)
        elif isinstance(rt, Decimal):
            scaled_int, fp = cls._to_18prec_parts(rt)
            return (scaled_int, fp)
        else:
            raise ValueError(f"Unknown type [{type(rt)}] for record")

    def _record_as_bytes(self, record: Tuple[RecType, ...]) -> bytes:
        record_for_format = itertools.chain.from_iterable((self._to_encoded_tpl(rt) for rt in record))
        return struct.pack(self._format, *record_for_format)

    def bytes_as_record(self, record_buffer: bytes) -> Tuple[RecType, ...]:
        items = []
        index = 0
        buffer = bytearray(record_buffer)
        for _type in self._record_format:
            if issubclass(_type, bool):
                decoded = struct.unpack_from("?", buffer, index)
                items.append(decoded[0])
                index += 1
            elif issubclass(_type, int):
                decoded = struct.unpack_from("q", buffer, index)
                items.append(decoded[0])
                index += 8
            elif issubclass(_type, float):
                decoded = struct.unpack_from("d", buffer, index)
                items.append(decoded[0])
                index += 8
            elif issubclass(_type, Decimal):
                scaled_int, fp = struct.unpack_from("qb", buffer, index)
                d = Decimal(scaled_int).scaleb(-fp)
                items.append(d)
                index += 9
            else:
                raise ValueError(f"Unknown type [{_type!s}] for record")

        return tuple(items)

    def _buffered_read_records(self, f) -> Generator[bytes, None, None]:
        while True:
            buffer = f.read(self._buf_size)
            if not buffer:
                break
            for i in range(0, len(buffer), self._record_size):
                next_record = buffer[i: i + self._record_size]
                yield next_record

    def _itr(self) -> Generator[Tuple[RecType], None, None]:
        validate(self._fh is None, f"The BinaryRecordFile {self._fpath!s} is already opened.")
        with self._opener(self._fpath, mode="rb") as f:
            self._fh = f
            # Pass header
            f.read(self._header_sz)
            for rec_bytes in self._buffered_read_records(f):
                next_record = self.bytes_as_record(rec_bytes)
                yield next_record
        self._fh = None

    def __iter__(self) -> Iterator[_K]:
        return iter(self._itr())

    def as_named_tuple(self, record: Tuple[RecType, ...]) -> NamedTuple:
        return self._namedtuple_class(*record)

    def as_named_tuple_stream(self) -> Iterable[NamedTuple]:
        return map(self.as_named_tuple, iter(self))

    def __getitem__(self, i: Union[slice, int]) -> Union[Tuple[RecType, ...], Iterator[Tuple[RecType, ...]]]:
        if isinstance(i, slice):
            return self.__getslice(i.start, i.stop, i.step)
        else:
            validate(self._fh is None, f"The BinaryRecordFile {self._fpath!s} is already opened.")
            with self._opener(self._fpath, mode="rb") as f:
                if i < 0:
                    file_len = self._get_len(f)
                    i += file_len
                seek_pos = self._record_size * i + self._header_sz
                f.seek(seek_pos, io.SEEK_SET)
                rec = f.read(self._record_size)
                return self.bytes_as_record(rec)

    def _get_len(self, f) -> int:
        """ Returns number of records in this file. """
        if self._header.records_nr != -1:
            return self._header.records_nr

        f.seek(0, io.SEEK_END)
        file_sz = f.tell() - self._header_sz
        validate(file_sz % self._record_size == 0, f"The file {self._fpath} is corrupted", IOError)
        return file_sz // self._record_size

    def __getslice(
            self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None
    ) -> Generator[Tuple[RecType, ...], None, None]:
        validate(self._fh is None, f"The BinaryRecordFile {self._fpath!s} is already opened.")
        if step is None:
            step = 1
        if start is None:
            start = 0

        with self._opener(self._fpath, mode="rb") as f:
            self._fh = f
            seek_pos = self._record_size * start + self._header_sz
            if stop is None:
                stop = self._get_len(f)
            elif stop < 0:
                stop += self._get_len(f)
            f.seek(seek_pos, io.SEEK_SET)

            for i, rec_bytes in enumerate(self._buffered_read_records(f)):
                if i < stop - start:
                    if i % step == 0:
                        yield self.bytes_as_record(rec_bytes)
                else:
                    break
            self._fh = None

    def __len__(self) -> int:
        if self._header.records_nr != -1:
            return self._header.records_nr
        validate(self._fh is None, f"The BinaryRecordFile {self._fpath!s} is already opened.")
        with self._opener(self._fpath, mode="rb") as f:
            file_len = self._get_len(f)
            return file_len

    def __iadd__(self, other: Iterable[Record]) -> 'BinTableFile[Record]':
        self.extend(other)
        return self

    def extend(self, records: Iterable[Record]):
        for record in records:
            self.append(record)

    def append(self, record: Record) -> Record:
        """
        According to https://www.guyrutenberg.com/2020/04/04/fast-bytes-concatenation-in-python/
        the below buffer adding is the fastest implementation
        """
        validate(self._header.records_nr == -1, "Cannot append to a file with fixed length")
        chunk = self._record_as_bytes(record)
        self._buffer += chunk
        if len(self._buffer) >= self._buf_size:
            self.flush()
        return record

    def truncate(self, n: int) -> None:
        """
        Truncates the file and let to remain first n records.
        If the file has less than n records, does nothing.
        """
        validate(self._header.records_nr == -1, "Cannot append to a file with fixed length")
        if n >= len(self):
            return
        seek_pos = self._record_size * n + self._header_sz
        with self._opener(self._fpath, mode="r+") as f:
            f.truncate(seek_pos)

    def flush(self):
        do_add_header = not self._fpath.exists()
        mode = "wb" if do_add_header else "ab"
        validate(self._fh is None, f"The BinaryRecordFile {self._fpath!s} is already opened.")
        validate(self._header.records_nr == -1, "Cannot write to a file with fixed length")
        with self._opener(self._fpath, mode=mode) as f:
            if do_add_header:
                validate(self._header_data, "When creating new BinaryRecordFile, you must specify header")
                f.write(self._header_data)
            f.write(self._buffer)
        self._buffer = bytearray()

    @staticmethod
    def _build_pack_format(record_format: Tuple[RecTypeType, ...]):
        format = "<"
        for _type in record_format:
            if issubclass(_type, (bool, np.bool_)):
                format += "?"
            elif issubclass(_type, (int, np.int64)):
                format += "q"
            elif issubclass(_type, (float, np.float64)):
                format += "d"
            elif issubclass(_type, Decimal):
                format += "qb"
            else:
                raise ValueError(f"Unknown type [{_type!s}] for record")
        return format

    def as_df(self) -> pd.DataFrame:

        return pd.DataFrame(data=list(self), columns=self._columns, index=None)

    @classmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def save_df(cls, df: pd.DataFrame, fpath: Union[str, Path], opener: Any = open_by_extension,
                metadata: Optional[Dict[str, Any]] = None,
                buf_size: int = BUF_SZ) -> None:
        """
        Saves pandas DataFrame as BinaryRecordFile
        """
        columns = tuple(df.columns)
        record_format = tuple(t.type for t in df.dtypes)
        format = cls._build_pack_format(record_format)
        builtin_record_format = tuple(cls.BUILTIN_TYPES_MAP.get(t, t) for t in record_format)
        record_size = cls._compute_record_size_in_bytes(builtin_record_format)
        records_nr = len(df.index)
        header = Header(columns=columns, types=builtin_record_format, record_size=record_size,
                        records_nr=records_nr, metadata=metadata)
        header_data = header.encode()
        with opener(fpath, mode="wb") as f:
            f.write(header_data)
            data = df.to_numpy()
            records_nr = len(data)
            i = 0
            while i < records_nr:
                chunk_size = min(records_nr - i, buf_size)
                chunk = data[i:i + chunk_size]
                chunk_format = '<' + format[1:] * chunk_size
                f.write(struct.pack(chunk_format, *chunk.flatten()))
                i += chunk_size
            f.flush()
