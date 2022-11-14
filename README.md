# Binary Table File format 

## Binary Table File - efficient binary file format to store and retrieve tabular data

This library is written in Cython for speed efficiency.

- is basically a typed CSV, but binary and indexed, 
    supporting only a subset of the types: `int, float, bool, Decimal`.
- The table is backed by a file on disk, and is _**NOT**_ memory-mapped.
This class just interfaces the reading/writing of the file through a `list` interface. 
- All the data is indexed and can be accessed as the list of records, where each record is a tuple of values. 
- It's main purpose are big files that don't fit into memory, or the files that are read partially. Ex. a few records from the middle of the file. 
- It can efficiently slice through the file and read just last N records, or just the first N records. 
- The footprint of the file is quite small as the data is binary encoded.
- It can be used with a seek-able compression format without losing efficiency on indexing, like [idzip](https://pypi.org/project/python-idzip/) 

### Installation
```bash
pip install bintablefile
```
The library can be found on [PyPi](https://pypi.org/project/bintablefile/): https://pypi.org/project/bintablefile/

 
**Note**: We release directly v2.0 as v1.0 was used just internally and was not released to PyPI. The major improvement of V2 is a full-featured header, that allows to store metadata about the table, as well as store the number of records for ReadOnly compression formats like `idzip`.