#!/usr/bin/env python3

# Python standard library imports
import argparse
import os
from pathlib import Path
import resource
import timeit

# Third party packages
import h5py as h5
import numpy as np
import vtk
from vtk.util import numpy_support as nps
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase


__version__ = "1.1"


class PythonAMRHDFBase(VTKPythonAlgorithmBase):
    # One place to define certain properties for Paraview decorators
    extensions = "h5 hdf vtkhdf",
    file_description = "VTKHDF OverlappingAMR proposal",

    def __init__(self):
        # HDF5 cache parameters - None means use h5py defaults
        self.rdcc_nbytes = None
        self.rdcc_w0 = None
        self.rdcc_nslots = None

        # Filename to read
        self.filename = None

    def SetFileName(self, filename):
        if isinstance(filename, str):
            if self.filename != filename:
                self.filename = filename
        elif isinstance(filename, Path):
            if self.filename != str(filename):
                self.filename = str(filename)
        else:
            raise RuntimeError(f"Expected str, got {type(filename)}")

        self.Modified()

    def GetFileName(self):
        return self.filename

    def SetRdcc(self, rdcc_nbytes=None, rdcc_w0=None, rdcc_nslots=None):
        """Set the Raw Data Chunk Cache parameters on opening the HDF5 file
        I do not call Modified() here becuase it technically does not
        modify the data provided by the reader"""
        if rdcc_nbytes:
            self.SetRdcc_nbytes(rdcc_nbytes)

        if rdcc_w0:
            self.SetRdcc_w0(rdcc_w0)

        if rdcc_nslots:
            self.SetRdcc_nslots(rdcc_nslots)

    def SetRdcc_nbytes(self, rdcc_nbytes):
        if isinstance(rdcc_nbytes, int):
            self.rdcc_nbytes = rdcc_nbytes
        else:
            raise RuntimeError(f"Expected int, got {type(rdcc_nbytes)}")

    def GetRdcc_nbytes(self):
        return self.rdcc_nbytes

    def SetRdcc_w0(self, rdcc_w0):
        if isinstance(rdcc_w0, float):
            self.rdcc_w0 = rdcc_w0
        else:
            raise RuntimeError(f"Expected float, got {type(rdcc_w0)}")

    def GetRdcc_w0(self):
        return self.rdcc_w0

    def SetRdcc_nslots(self, rdcc_nslots):
        if isinstance(rdcc_nslots, int):
            self.rdcc_nslots = rdcc_nslots
        else:
            raise RuntimeError(f"Expected int, got {type(rdcc_nslots)}")

    def GetRdcc_nslots(self):
        return self.rdcc_nslots

    def amrbox_to_offset(self, amrbox):
        ngrids = amrbox.shape[0]

        # Compute offsets and lengths for each grid
        offset_pt = np.zeros([ngrids, 2], dtype=np.int64)
        offset_cl = np.zeros([ngrids, 2], dtype=np.int64)

        # 2-D datasets (planes) have the same number of points as cells in
        # the plane normal direction. Correct for this.
        cx = 1
        cy = 1
        cz = 1
        if self.desc == vtk.VTK_YZ_PLANE:
            cx = 0
        elif self.desc == vtk.VTK_XZ_PLANE:
            cy = 0
        elif self.desc == vtk.VTK_XY_PLANE:
            cz = 0

        for igrid in range(ngrids):
            nx = amrbox[igrid, 1] - amrbox[igrid, 0] + 1
            ny = amrbox[igrid, 3] - amrbox[igrid, 2] + 1
            nz = amrbox[igrid, 5] - amrbox[igrid, 4] + 1

            # The AMRBOx of a 2-D plane is in the plane normal direction
            # set from 0 to -1 - this cannot be used to compute the length
            if self.desc == vtk.VTK_YZ_PLANE:
                nx = 1
            elif self.desc == vtk.VTK_XZ_PLANE:
                ny = 1
            elif self.desc == vtk.VTK_XY_PLANE:
                nz = 1

            length_pt = (nx+cx)*(ny+cy)*(nz+cz)
            length_cl = nx*ny*nz

            # Lengthength
            offset_pt[igrid, 1] = length_pt
            offset_cl[igrid, 1] = length_cl

            # Offset
            if igrid > 0:
                offset_pt[igrid, 0] = offset_pt[igrid-1, 0] + \
                    offset_pt[igrid-1, 1]
                offset_cl[igrid, 0] = offset_cl[igrid-1, 0] + \
                    offset_cl[igrid-1, 1]

        return offset_pt, offset_cl

    @staticmethod
    def fielddata_offset(ngrids):
        offset_fd = np.column_stack((np.arange(ngrids, dtype=np.int64),
                                     np.ones([ngrids], dtype=np.int64)))
        return offset_fd


class PythonAMRHDFWriter(PythonAMRHDFBase):
    version = (1, 0)
    type = "OverlappingAMR"

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=0,
                                        inputType='vtkOverlappingAMR')
        super().__init__()

        self.N = None

    def SetChunkFactor(self, N):
        """Sets the chunksize to N times the average dataset size on each
        level. If all the datasets are equal in size this ensure that each
        dataset boundary is also a chunk boundary."""
        if isinstance(N, int):
            self.N = N
        else:
            raise RuntimeError(f"Expected int, got {type(N)}")

    def GetChunkFactor(self):
        return self.N

    def Write(self):
        self.Modified()
        self.Update()

    def RequestData(self, request, inInfo, outInfo):
        amr = vtk.vtkOverlappingAMR.GetData(inInfo[0])
        self.write(amr)
        return 1

    def SetInputData(self, data):
        return self.SetInputDataObject(0, data)

    def write(self, amr):
        if not self.filename:
            raise RuntimeError("Input filename is not set")

        # Number of levels in input AMR
        self.nlevels = amr.GetNumberOfLevels()

        # Number of grids per level
        self.grids_per_level = np.zeros([self.nlevels], dtype=np.intc)
        for ilevel in range(self.nlevels):
            self.grids_per_level[ilevel] = amr.GetNumberOfDataSets(ilevel)

        # Coarsest level gridspacing
        self.spacing = np.zeros([3], dtype=np.double)
        amr.GetSpacing(0, self.spacing)

        # Overall AMR bounds
        bounds = np.zeros([6], dtype=np.double)
        amr.GetBounds(bounds)
        self.global_origin = np.array([bounds[0], bounds[2], bounds[4]])

        # Grid description
        self.desc = amr.GetGridDescription()

        # Create HDF5 output file
        with h5.File(self.filename, "w", rdcc_nbytes=self.rdcc_nbytes,
                     rdcc_w0=self.rdcc_w0, rdcc_nslots=self.rdcc_nslots) as fh:
            vtkhdf = fh.create_group("VTKHDF")
            vtkhdf.attrs.create("Version", self.version)

            typeattr = self.type.encode('ascii')
            vtkhdf.attrs.create("Type", typeattr,
                                dtype=h5.string_dtype('ascii', len(typeattr)))
            vtkhdf.attrs.create("Origin", self.global_origin)

            # Write out/convert one level at a time
            for ilevel in range(self.nlevels):
                level_h = vtkhdf.create_group(f"Level{ilevel}")

                # Gridspacing
                spacing = np.zeros([3], dtype=np.double)
                amr.GetSpacing(ilevel, spacing)
                level_h.attrs.create("Spacing", spacing)

                # Refinement ratio
                ratio = amr.GetRefinementRatio(ilevel)
                level_h.attrs.create("RefinementRatio", ratio)

                # Construct and write AMRBox for this level
                amrbox = self.get_amrbox(amr, ilevel)
                level_h.create_dataset("AMRBox", data=amrbox)

                # Write point/cell data for this level
                pointdata = level_h.create_group("PointData")
                celldata = level_h.create_group("CellData")
                fielddata = level_h.create_group("FieldData")
                self.write_data(amr, ilevel, amrbox, pointdata,
                                celldata, fielddata)

    def get_amrbox(self, amr, ilevel):
        ngrids = self.grids_per_level[ilevel]
        amrbox = np.zeros([ngrids, 6], dtype=np.intc)
        for igrid in range(ngrids):
            this_amrbox = amr.GetAMRBox(ilevel, igrid)
            this_amrbox.GetDimensions(amrbox[igrid, :])
        return amrbox

    def write_data(self, amr, ilevel, amrbox, pointdata, celldata, fielddata):
        offset_pt, offset_cl = self.amrbox_to_offset(amrbox)
        offset_fd = self.fielddata_offset(amrbox.shape[0])

        # Query first grid for which arrays it has
        dataset0 = amr.GetDataSet(ilevel, 0)
        point_arrays = self.find_arrays(dataset0.GetPointData())
        cell_arrays = self.find_arrays(dataset0.GetCellData())
        field_arrays = self.find_arrays(dataset0.GetFieldData())

        # Iterate over arrays to write them out
        for arr in point_arrays:
            self.write_array(amr, ilevel, pointdata, offset_pt, arr, "point")

        for arr in cell_arrays:
            self.write_array(amr, ilevel, celldata, offset_cl, arr, "cell")

        for arr in field_arrays:
            self.write_array(amr, ilevel, fielddata, offset_fd, arr, "field")

    @staticmethod
    def find_arrays(data):
        arrays = []
        ndata = data.GetNumberOfArrays()
        for iarr in range(ndata):
            array = data.GetAbstractArray(iarr)
            name = array.GetName()
            typ = array.GetDataType()
            ncmp = array.GetNumberOfComponents()
            ntup = array.GetNumberOfTuples()
            nval = array.GetNumberOfValues()

            # DISCUSS: Maybe a dirty hack, but I don't know exactly how
            # fielddata with zero values should be handled - it's a special
            # corner case. HDF5 support creating a dataset with zero length
            # that is most probably the most efficient way.
            if nval == 0:
                continue

            try:
                arrays.append({
                    "name": name,
                    "dtype": nps.get_numpy_array_type(typ),
                    "ncmp": ncmp,
                    "ntup": ntup,
                    "nval": nval,
                    })
            except KeyError:
                # This skips fields where nps.get_numpy_array_type cannot find
                # a mathcing datatype (strings). Possible to implement but will
                # require some additional programming.
                pass

        return arrays

    def write_array(self, amr, ilevel, fileh, offset, arr, typ):
        name = arr["name"]
        dtype = arr["dtype"]
        ncmp = arr["ncmp"]
        nval = arr["nval"]

        # For fielddata all elements must have same length on all grids, scale
        # with number of values (make deep copy)
        if typ == "field":
            offset = np.array(nval//ncmp*offset)

        # Overall length of data is the offset + length of the last element
        length = offset[-1, 0] + offset[-1, 1]

        # Create in memory storage
        if ncmp > 1:
            shape = (length, ncmp)
        else:
            shape = (length, )
        buffer = np.zeros(shape, dtype=dtype)

        # Populate storage in memory
        ngrids = self.grids_per_level[ilevel]
        for igrid in range(ngrids):
            grid = amr.GetDataSet(ilevel, igrid)

            if typ == "cell":
                data = grid.GetCellData()
            elif typ == "point":
                data = grid.GetPointData()
            elif typ == "field":
                data = grid.GetFieldData()
            else:
                raise RuntimeError(f"Invalid type: {typ}")

            array = data.GetAbstractArray(name)

            this_offset = offset[igrid, 0]
            this_length = offset[igrid, 1]

            buffer[this_offset:this_offset+this_length, ...] = \
                nps.vtk_to_numpy(array)

        # When VTK read the VTH file it changes the blanking information based
        # on the levels overlapping information. This is not desired to be
        # written out again.
        # Only write out HIDDENCELL bits and leave others at 0
        #
        # Update: VTK is updated to fix this behavior, but I keep this code
        # for some time until these changes have propagated into at least
        # one official Paraview release.
        # Ref: https://gitlab.kitware.com/vtk/vtk/-/merge_requests/8986
        if name == vtk.vtkDataSetAttributes.GhostArrayName():
            buffer = self.fix_iblank(buffer)

        # Average dataset size determine chunk size
        chunks = None
        maxshape = None
        if self.N and ngrids >= self.N and (not typ == "field"):
            if len(buffer.shape) == 1:
                chunks = (self.N*length//ncmp//ngrids, )
            else:
                chunks = (self.N*length//ncmp//ngrids, ncmp)
            maxshape = buffer.shape

        # Write to disk
        fileh.create_dataset(name, data=buffer, chunks=chunks,
                             maxshape=maxshape)

    @staticmethod
    def fix_iblank(data):
        """ This preserve the HIDDENCELL bit of a grid and clear other bits"""
        return np.bitwise_and(data, vtk.vtkDataSetAttributes.HIDDENCELL)


class PythonAMRHDFReader(PythonAMRHDFBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1,
                                        outputType='vtkOverlappingAMR')
        super().__init__()

        self.nlevels_max = 0
        self.pt_selection = vtk.vtkDataArraySelection()
        self.pt_selection.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                      self.SelectionModifiedCallback)

        self.cl_selection = vtk.vtkDataArraySelection()
        self.cl_selection.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                      self.SelectionModifiedCallback)

        self.selections_is_init = False

    def SelectionModifiedCallback(self, selection, event):
        self.Modified()

    def SetMaximumLevelsToReadByDefault(self, nlevels_max):
        if isinstance(nlevels_max, int):
            if self.nlevels_max != nlevels_max:
                self.nlevels_max = nlevels_max
                self.Modified()
        else:
            raise RuntimeError(f"Expected int, got {type(nlevels_max)}")

    def GetMaximumLevelsToReadByDefault(self):
        return self.nlevels_max

    def GetCellDataArraySelection(self):
        return self.cl_selection

    def GetPointDataArraySelection(self):
        return self.pt_selection

    def RequestData(self, request, inInfo, outInfo):
        amr = vtk.vtkOverlappingAMR.GetData(outInfo)
        self.read(amr)
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        """ Called on UpdateInformation """

        if not self.selections_is_init:
            self.init_selections()

        return 1

    def GetOutput(self):
        return self.GetOutputDataObject(0)

    def init_selections(self):
        if not self.filename:
            raise RuntimeError("Input filename is not set")

        # Interrogate level 0 to check what arrays are there
        with h5.File(self.filename, "r") as fh:
            level0 = fh["VTKHDF"]["Level0"]

            for arr in level0["PointData"].keys():
                self.pt_selection.AddArray(arr)

            for arr in level0["CellData"].keys():
                self.cl_selection.AddArray(arr)

        self.selections_is_init = True

    def read(self, amr):
        if not self.filename:
            raise RuntimeError("Input filename is not set")

        with h5.File(self.filename, "r", rdcc_nbytes=self.rdcc_nbytes,
                     rdcc_w0=self.rdcc_w0, rdcc_nslots=self.rdcc_nslots) as fh:
            vtkhdf = fh["VTKHDF"]

            self.nlevels, self.grids_per_level = self.levels_and_grids(vtkhdf)
            self.global_origin = vtkhdf.attrs["Origin"]

            # Limit reading to a specific number of levels
            if hasattr(self, 'nlevels_max') and self.nlevels_max > 0:
                self.nlevels = min(self.nlevels, self.nlevels_max)

            # Determine GridDescription from AMRBox at level 0
            self.desc = self.desc_from_amrbox(vtkhdf["Level0"]["AMRBox"])

            amr.Initialize(self.nlevels, self.grids_per_level)
            amr.SetOrigin(self.global_origin)
            amr.SetGridDescription(self.desc)

            # Read level properties
            for ilevel in range(self.nlevels):
                level_h = vtkhdf[f"Level{ilevel}"]

                ratio = level_h.attrs["RefinementRatio"]
                spacing = level_h.attrs["Spacing"]

                amr.SetRefinementRatio(ilevel, ratio)
                amr.SetSpacing(ilevel, spacing)

            # Assemble datasets
            for ilevel in range(self.nlevels):
                level_h = vtkhdf[f"Level{ilevel}"]
                spacing = level_h.attrs["Spacing"]
                self.create_datasets(amr, ilevel, spacing, level_h)

            # Fill datasets with data
            for ilevel in range(self.nlevels):
                level_h = vtkhdf[f"Level{ilevel}"]
                self.fill_datasets(amr, ilevel, level_h)

        amr.Audit()
        vtk.vtkAMRUtilities.BlankCells(amr)

    def create_datasets(self, amr, ilevel, spacing, level_h):
        # Read AMRBox dataset
        amrbox = level_h["AMRBox"][...]
        ngrids = amrbox.shape[0]

        for igrid in range(ngrids):
            this_box = amrbox[igrid, :]

            box = vtk.vtkAMRBox(this_box)
            amr.SetAMRBox(ilevel, igrid, box)

            grid = vtk.vtkUniformGrid()
            grid.SetSpacing(spacing)

            # this_dims are number of cells, SetDimensions is number of points
            this_dims = np.array([this_box[1] - this_box[0] + 1,
                                  this_box[3] - this_box[2] + 1,
                                  this_box[5] - this_box[4] + 1])
            grid.SetDimensions(this_dims + 1)

            # Origin
            origin = self.global_origin + this_box[0::2]*spacing
            grid.SetOrigin(origin)

            # Attach (empty) dataset
            amr.SetDataSet(ilevel, igrid, grid)

    def fill_datasets(self, amr, ilevel, level_h):
        # Read AMRBox dataset
        amrbox = level_h["AMRBox"][...]
        offset_pt, offset_cl = self.amrbox_to_offset(amrbox)
        offset_fd = self.fielddata_offset(amrbox.shape[0])

        # Read data
        if "PointData" in level_h:
            for arr in level_h["PointData"].keys():
                if not self.pt_selection.ArrayIsEnabled(arr):
                    continue

                self.read_array(amr, ilevel, level_h["PointData"],
                                offset_pt, arr, "point")

        if "CellData" in level_h:
            for arr in level_h["CellData"].keys():
                if not self.cl_selection.ArrayIsEnabled(arr):
                    continue

                self.read_array(amr, ilevel, level_h["CellData"],
                                offset_cl, arr, "cell")

        if "FieldData" in level_h:
            for arr in level_h["FieldData"].keys():
                self.read_array(amr, ilevel, level_h["FieldData"],
                                offset_fd, arr, "field")

    def read_array(self, amr, ilevel, fileh, offset, arr, typ):
        ngrids = self.grids_per_level[ilevel]

        # Read the entire dataset in memory buffer
        buffer = fileh[arr][...]

        # Dirty hack for FieldData
        if typ == "field":
            # Here every grid must have the same length (=number of values)
            if buffer.shape[0] % ngrids > 0:
                raise RuntimeError(f"Invalid length of array: {buffer.shape}")

            nval = buffer.shape[0]//ngrids
            offset = np.array(nval*offset)

        # Populate AMR structure in memory
        for igrid in range(ngrids):
            grid = amr.GetDataSet(ilevel, igrid)
            if not grid:
                raise RuntimeError(f"Error getting dataset {igrid} at "
                                   f"level {ilevel} from AMR structure")

            if typ == "cell":
                data = grid.GetCellData()
            elif typ == "point":
                data = grid.GetPointData()
            elif typ == "field":
                data = grid.GetFieldData()
            else:
                raise RuntimeError(f"Invalid type: {typ}")

            this_offset = offset[igrid, 0]
            this_length = offset[igrid, 1]

            view = buffer[this_offset:this_offset+this_length, ...]
            array = nps.numpy_to_vtk(view, deep=1)
            array.SetName(arr)
            data.AddArray(array)

    @staticmethod
    def desc_from_amrbox(amrbox):
        # If first grid is 2D all other grids are assumed to be 2D
        box0 = amrbox[0, ...]

        # AMRBox of 2D datasets are from 0 to -1 in the plane-normal direction,
        # therefore check if the upper coordinate is negative
        #
        # OverlappingAMR does not support 1D datasets so no need to check for
        # that
        if box0[1] < 0:
            return vtk.VTK_YZ_PLANE
        elif box0[3] < 0:
            return vtk.VTK_XZ_PLANE
        elif box0[5] < 0:
            return vtk.VTK_XY_PLANE

        return vtk.VTK_XYZ_GRID

    @staticmethod
    def levels_and_grids(vtkhdf):
        nlevels = 0
        grids_per_level = []
        while f"Level{nlevels}" in vtkhdf:
            amrbox = vtkhdf[f"Level{nlevels}"]["AMRBox"]
            grids_per_level.append(amrbox.shape[0])
            nlevels += 1

        return nlevels, np.array(grids_per_level, dtype=np.intc)


try:
    # These imports are only available if you run the code from within Paraview
    from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, \
        smhint

except ImportError:
    # Not executing from within Paraview/pvpytrhon - so the class
    # definitions below are not needed
    pass

else:
    # Decorate the readers and writers for usage within Paraview
    @smproxy.reader(name="PVPythonAMRHDFReader",
                    label="Python-based AMRHDF5 Reader",
                    extensions=PythonAMRHDFReader.extensions,
                    file_description=PythonAMRHDFReader.file_description,
                    support_reload=False)
    class PVPythonAMRHDFReader(PythonAMRHDFReader):
        @smproperty.stringvector(name="FileName", panel_visibility="never")
        @smdomain.filelist()
        @smhint.filechooser(extensions=PythonAMRHDFReader.extensions,
                            file_description=PythonAMRHDFReader.
                            file_description)
        def SetFileName(self, filename):
            return super().SetFileName(filename)

        @smproperty.intvector(name="DefaultNumberOfLevels",
                              default_values=1)
        @smdomain.intrange(min=0, max=10)
        def SetMaximumLevelsToReadByDefault(self, levels):
            return super().SetMaximumLevelsToReadByDefault(levels)

        @smproperty.dataarrayselection(name="Point Data Arrays")
        def GetPointDataArraySelection(self):
            return super().GetPointDataArraySelection()

        @smproperty.dataarrayselection(name="Cell Data Arrays")
        def GetCellDataArraySelection(self):
            return super().GetCellDataArraySelection()

    @smproxy.writer(name="PVPythonAMRHDFWriter",
                    label="Python-based AMRHDF5 Writer",
                    extensions=PythonAMRHDFWriter.extensions,
                    file_description=PythonAMRHDFWriter.file_description,
                    support_reload=False)
    @smproperty.input(name="Input", port_index=0)
    @smdomain.datatype(dataTypes=["vtkOverlappingAMR"])
    class PVPythonAMRHDFWriter(PythonAMRHDFWriter):
        @smproperty.stringvector(name="FileName", panel_visibility="never")
        @smdomain.filelist()
        def SetFileName(self, filename):
            return super().SetFileName(filename)


def vth_to_hdf5(input, output):
    tic = timeit.default_timer()
    reader = vtk.vtkXMLUniformGridAMRReader()
    reader.SetFileName(input)
    reader.SetMaximumLevelsToReadByDefault(0)

    writer = PythonAMRHDFWriter()
    writer.SetFileName(output)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()
    print(f"Read/write {output} took {timeit.default_timer() - tic} seconds")


def hdf5_to_vth(input, output):
    tic = timeit.default_timer()
    reader = PythonAMRHDFReader()
    reader.SetFileName(input)
    reader.SetMaximumLevelsToReadByDefault(0)

    writer = vtk.vtkXMLUniformGridAMRWriter()
    writer.SetFileName(output)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetEncodeAppendedData(False)
    writer.SetCompressorTypeToNone()
    writer.Write()
    print(f"Read/write {output} took {timeit.default_timer() - tic} seconds")


def main():
    desc = "Tool to convert between vtkOverlappingAMR and VTKHDF"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("input", nargs='?', help="Input file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--version", action="store_true",
                        default=False, help="Print version and quit")
    args = parser.parse_args()

    if args.version:
        print(f"PythonAMRHDF version {__version__}")
        return

    if not os.path.exists(args.input):
        raise RuntimeError("File does not exits: {}".format(args.input))

    input_file = os.path.basename(args.input)
    input_file_array = os.path.splitext(input_file)
    input_file_noext = input_file_array[0]
    input_file_ext = input_file_array[1]

    # If output filename is specified use that
    output_file = args.output if args.output else None

    if input_file_ext.lower() == ".vth":
        if not output_file:
            output_file = f"{input_file_noext}.hdf"
        vth_to_hdf5(args.input, output_file)
    else:
        if not output_file:
            output_file = f"{input_file_noext}.vth"
        hdf5_to_vth(args.input, output_file)

    maxmem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024
    print("Peak memory usage: {} MB".format(maxmem))


if __name__ == "__main__":
    main()
