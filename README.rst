#######################################################
WIP: Python-based OverlappingAMR HDF5 reader and writer
#######################################################

This is a **proposal** for an OverlappingAMR HDF5-based file format for VTK. The
implementation is a prototype in Python using ``h5py`` and ``numpy``.

The code an either be imported as a module in a Python program, used as a
command line converter or used as a Python-plugin in recent Paraview
versions (given support for Python is present and the packages ``h5py`` and
``numpy`` is installed).

**Important:** Do not use this code for anything but development and testing!
The file format is neither implemented nor officially endorsed by Kitware in
any way.

The implementation has the following limitations:

1.  Only the first grid on the first level is interrogated to find which data
    arrays to write out. All arrays must be present on all grids on all levels.

2.  Arrays with character datatype are silently skipped (vtk numpy_support
    does not handle character arrays very good).

3.  Arrays with zero elements are silently skipped.

4.  The implementation create a memory buffer for each data array, fills this
    buffer completely and flushes it to disk for each complete array and level
    for maximum performance.

See also the `VTK forum discussion thread <https://discourse.vtk.org/t/overlapping-amr-support-in-vtkhdf/7868>`_
and the associated `merge request <https://gitlab.kitware.com/vtk/vtk/-/merge_requests/9065>`_.