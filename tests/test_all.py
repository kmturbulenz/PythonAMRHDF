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


def test_pulse1(tmp_path):
    """Test reading/writing roundtrip with the Python reader/writer"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)
    reader.SetMaximumLevelsToReadByDefault(0)
    reader.Update()

    assert _validate_amr(pulse.GetOutput(), reader.GetOutput())


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

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)
    reader.SetMaximumLevelsToReadByDefault(0)
    reader.Update()

    assert _validate_amr(pulsedata, reader.GetOutput())


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

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)
    reader.SetMaximumLevelsToReadByDefault(0)
    reader.Update()

    assert _validate_amr(pulsedata, reader.GetOutput())


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

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)

    for levels in range(nlevels):
        reader.SetMaximumLevelsToReadByDefault(levels + 1)
        reader.Update()
        assert reader.GetOutput().GetNumberOfLevels() == levels + 1


def test_arrayselection(tmp_path):
    """Test cell array selection on reader"""
    filename = tmp_path / "gaussian_pulse.hdf"

    pulse = vtk.vtkAMRGaussianPulseSource()

    writer = PythonAMRHDF.PythonAMRHDFWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(pulse.GetOutputPort())
    writer.Write()

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)

    reader.UpdateInformation()
    reader.GetCellDataArraySelection().DisableAllArrays()
    reader.GetCellDataArraySelection().EnableArray("Centroid")
    reader.Update()

    first_dataset = reader.GetOutput().GetDataSet(0, 0)
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

    reader = PythonAMRHDF.PythonAMRHDFReader()
    reader.SetFileName(filename)
    reader.Update()

    assert _validate_amr(pulse.GetOutput(), reader.GetOutput())
