import PythonAMRHDF
import vtk
from vtk.util import numpy_support as nps
import numpy as np


# Inspired by VTK function Validate in IO/XML/Testing/Cxx/TestAMRXMLIO.cxx
def _validate_amr(original, result):
    original.Audit()
    result.Audit()

    assert original.GetNumberOfLevels() == result.GetNumberOfLevels()
    assert original.GetOrigin()[0] == result.GetOrigin()[0]
    assert original.GetOrigin()[1] == result.GetOrigin()[1]
    assert original.GetOrigin()[2] == result.GetOrigin()[2]
    assert original.GetGridDescription() == result.GetGridDescription()

    nlevels = original.GetNumberOfLevels()
    grids_per_level = np.zeros([nlevels], dtype=np.intc)
    for ilevel in range(nlevels):
        assert original.GetNumberOfDataSets(ilevel) == \
            result.GetNumberOfDataSets(ilevel)
        grids_per_level[ilevel] = original.GetNumberOfDataSets(ilevel)

    for ilevel in range(nlevels):
        ngrids = grids_per_level[ilevel]
        for igrid in range(ngrids):
            original_grid = original.GetDataSet(ilevel, igrid)
            result_grid = result.GetDataSet(ilevel, igrid)
            assert _validate_grid(original_grid, result_grid)

    return True


def _validate_grid(original, result):
    assert original.GetOrigin()[0] == result.GetOrigin()[0]
    assert original.GetOrigin()[1] == result.GetOrigin()[1]
    assert original.GetOrigin()[2] == result.GetOrigin()[2]

    assert original.GetSpacing()[0] == result.GetSpacing()[0]
    assert original.GetSpacing()[1] == result.GetSpacing()[1]
    assert original.GetSpacing()[2] == result.GetSpacing()[2]

    for i in range(6):
        assert original.GetExtent()[i] == result.GetExtent()[i]

    assert _validate_data(original.GetCellData(), result.GetCellData())
    assert _validate_data(original.GetPointData(), result.GetPointData())
    assert _validate_data(original.GetFieldData(), result.GetFieldData())

    return True


def _validate_data(original, result):
    assert original.GetNumberOfArrays() == result.GetNumberOfArrays()
    ndata = original.GetNumberOfArrays()

    # Order of arrays is not important
    for iarr in range(ndata):
        original_arr = original.GetAbstractArray(iarr)
        name = original_arr.GetName()

        # Look up resulting array by name
        result_arr = result.GetAbstractArray(name)

        assert original_arr.GetDataType() == result_arr.GetDataType()
        assert original_arr.GetNumberOfComponents() == \
            result_arr.GetNumberOfComponents()
        assert original_arr.GetNumberOfTuples() == \
            result_arr.GetNumberOfTuples()
        assert original_arr.GetNumberOfValues() == \
            result_arr.GetNumberOfValues()

        original_data = nps.vtk_to_numpy(original_arr)
        result_data = nps.vtk_to_numpy(result_arr)
        assert np.array_equal(original_data, result_data)

    return True


def _add_grid_to_amr(amr, ilevel, igrid, amrbox):
    # Adds another grid to the AMR, assumes that the global origin
    # is 0.0, 0.0, 0.0. The grid has a random cell array "Phi".
    nx = (amrbox[1] - amrbox[0]) + 1
    ny = (amrbox[3] - amrbox[2]) + 1
    nz = (amrbox[5] - amrbox[4]) + 1

    box = vtk.vtkAMRBox(amrbox)
    amr.SetAMRBox(ilevel, igrid, box)

    dx = np.zeros((3, ), dtype=np.double)
    amr.GetSpacing(ilevel, dx)

    grid = vtk.vtkUniformGrid()
    grid.SetSpacing(dx)
    grid.SetDimensions((nx+1, ny+1, nz+1))
    grid.SetOrigin((dx[0]*amrbox[0], dx[1]*amrbox[2], dx[2]*amrbox[4], ))

    rng = np.random.default_rng(0)
    phi = rng.random((nx, ny, nz)).flatten()
    array = nps.numpy_to_vtk(phi, deep=1)
    array.SetName("Phi")
    grid.GetCellData().AddArray(array)

    amr.SetDataSet(ilevel, igrid, grid)


def test_pulse1(tmp_path):
    """Test reading/writing roundtrip with the Python reader/writer, check that
    the VTK implemented reader gives the same result"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    reader1.SetMaximumLevelsToReadByDefault(0)
    reader1.Update()

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    reader2.SetMaximumLevelsToReadByDefaultForAMR(0)
    reader2.Update()

    assert _validate_amr(pulse.GetOutput(), reader1.GetOutput())
    assert _validate_amr(pulse.GetOutput(), reader2.GetOutput())


def test_pulse2(tmp_path):
    """Test reading/writing roundtrip with the Python reader/writer,
    using a data object as input instead of pipeline port"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()
    pulse.Update()
    pulsedata = pulse.GetOutput()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputData(pulsedata)
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    reader1.SetMaximumLevelsToReadByDefault(0)
    reader1.Update()

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    reader2.SetMaximumLevelsToReadByDefaultForAMR(0)
    reader2.Update()

    assert _validate_amr(pulsedata, reader1.GetOutput())
    assert _validate_amr(pulsedata, reader2.GetOutput())


def test_pulse_chunked(tmp_path):
    """Test reading/writing roundtrip with the Python reader/writer,
    using chunking in the HDF5 file"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()
    pulse.Update()
    pulsedata = pulse.GetOutput()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputData(pulsedata)
    writer.SetChunkFactor(2)
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    reader1.SetMaximumLevelsToReadByDefault(0)
    reader1.Update()

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    reader2.SetMaximumLevelsToReadByDefaultForAMR(0)
    reader2.Update()

    assert _validate_amr(pulsedata, reader1.GetOutput())
    assert _validate_amr(pulsedata, reader2.GetOutput())


def test_pulse_2d(tmp_path):
    """Test reading/writing roundtrip with the Python reader/writer
    in 2D (the VTK implementation cannot read 2D HDF5 AMR's)"""
    filename = tmp_path / "gaussian_pulse_2d.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()
    pulse.SetDimension(2)

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)
    reader.SetMaximumLevelsToReadByDefault(0)
    reader.Update()

    assert _validate_amr(pulse.GetOutput(), reader.GetOutput())


def test_maximumlevels(tmp_path):
    """Test reading N levels"""
    filename = tmp_path / "gaussian_pulse.hdf"
    nlevels = 2

    pulse = vtk.vtkAMRGaussianPulseSource()
    # This has actually no effect
    pulse.SetNumberOfLevels(nlevels)

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    for levels in range(nlevels):
        reader1.SetMaximumLevelsToReadByDefault(levels + 1)
        reader1.Update()
        assert reader1.GetOutput().GetNumberOfLevels() == levels + 1

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    for levels in range(nlevels):
        reader2.SetMaximumLevelsToReadByDefaultForAMR(levels + 1)
        reader2.Update()
        assert reader2.GetOutput().GetNumberOfLevels() == levels + 1


def test_arrayselection(tmp_path):
    """Test cell array selection on reader"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)

    reader1.UpdateInformation()
    reader1.GetCellDataArraySelection().DisableAllArrays()
    reader1.GetCellDataArraySelection().EnableArray("Centroid")
    reader1.Update()

    first_dataset = reader1.GetOutput().GetDataSet(0, 0)
    assert not first_dataset.GetCellData().GetArray("Gaussian-Pulse")
    assert first_dataset.GetCellData().GetArray("Centroid")

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)

    reader2.UpdateInformation()
    reader2.GetCellDataArraySelection().DisableAllArrays()
    reader2.GetCellDataArraySelection().EnableArray("Centroid")
    reader2.Update()

    first_dataset = reader2.GetOutput().GetDataSet(0, 0)
    assert not first_dataset.GetCellData().GetArray("Gaussian-Pulse")
    assert first_dataset.GetCellData().GetArray("Centroid")


def test_pulse_fielddata(tmp_path):
    """Test reading/writing fielddata of various lengths"""
    filename = tmp_path / "gaussian_pulse_fielddata.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()
    pulse.Update()
    pulsedata = pulse.GetOutput()

    # Add some fielddata to all datasets
    freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                     dtype=np.double)
    nfreqs = freqs.shape[0]
    igrid = 0
    iter = pulsedata.NewIterator()
    while not iter.IsDoneWithTraversal():
        grid = iter.GetCurrentDataObject()
        igrid += 1

        igrid_arr = vtk.vtkIntArray()
        igrid_arr.SetNumberOfValues(1)
        igrid_arr.SetValue(0, igrid)
        igrid_arr.SetName('IGRID')
        grid.GetFieldData().AddArray(igrid_arr)

        freq_arr = vtk.vtkDoubleArray()
        freq_arr.SetNumberOfValues(nfreqs)
        for i in range(nfreqs):
            freq_arr.SetValue(i, freqs[i])
        freq_arr.SetName('FREQUENCY')
        grid.GetFieldData().AddArray(freq_arr)

        iter.GoToNextItem()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputData(pulsedata)
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    reader1.Update()

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    reader2.Update()

    assert _validate_amr(pulse.GetOutput(), reader1.GetOutput())
    assert _validate_amr(pulse.GetOutput(), reader2.GetOutput())


def test_read_varlength(tmp_path):
    """Test reading/writing arrays of various lengths"""
    filename = tmp_path / "varlength.hdf"

    amr = vtk.vtkOverlappingAMR()

    amr.Initialize(3, (1, 3, 1, ))
    amr.SetOrigin((0.0, 0.0, 0.0))

    amr.SetRefinementRatio(0, 2)
    amr.SetRefinementRatio(1, 2)
    amr.SetRefinementRatio(2, 2)

    amr.SetSpacing(0, (1.0/32.0, 1.0/32.0, 1.0/32.0, ))
    amr.SetSpacing(1, (1.0/64.0, 1.0/64.0, 1.0/64.0, ))
    amr.SetSpacing(2, (1.0/128.0, 1.0/128.0, 1.0/128.0, ))

    # Add grids with random data
    _add_grid_to_amr(amr, 0, 0, (0, 31, 0, 15, 0, 23))
    _add_grid_to_amr(amr, 1, 0, (0, 31, 0, 15, 0, 23))
    _add_grid_to_amr(amr, 1, 1, (32, 47, 16, 23, 24, 35))
    _add_grid_to_amr(amr, 1, 2, (32, 63, 0, 13, 24, 35))
    _add_grid_to_amr(amr, 2, 0, (64, 95, 32, 39, 48, 71))

    amr.Audit()
    vtk.vtkAMRUtilities.BlankCells(amr)

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputData(amr)
    writer.Write()

    reader1 = PythonAMRHDF.PythonAMRHDFReader()
    reader1.SetFileName(filename)
    reader1.Update()

    reader2 = vtk.vtkHDFReader()
    reader2.SetFileName(filename)
    reader2.Update()

    assert _validate_amr(amr, reader1.GetOutput())
    assert _validate_amr(amr, reader2.GetOutput())
