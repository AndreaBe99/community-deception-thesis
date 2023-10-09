# Matrix Market

All files are in `.mtx` format.

From [NetworkX - Matrix Market](https://networkx.org/documentation/stable/reference/readwrite/matrix_market.html):
The Matrix Market exchange format is a text-based file format described by NIST. Matrix Market supports both a coordinate format for sparse matrices and an array format for dense matrices. The scipy.io module provides the scipy.io.mmread and scipy.io.mmwrite functions to read and write data in Matrix Market format, respectively. These functions work with either numpy.ndarray or scipy.sparse.coo_matrix objects depending on whether the data is in array or coordinate format. These functions can be combined with those of NetworkX’s convert_matrix module to read and write Graphs in Matrix Market format.