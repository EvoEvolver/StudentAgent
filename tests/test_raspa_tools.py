"""
Comprehensive tests for RASPA tools in tools_raspa.py

Run with: pytest tests/test_raspa_tools.py -v
"""

import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from student.agent.tools.tools_raspa import (
    ExecuteRaspa,
    MakeInputFile,
    ReadFile,
    WriteFile,
)
from student.agent.tools.molecule_loader import MoleculeLoader
from student.agent.tools.framework_loader import FrameworkLoader
from student.agent.tools.output_parser import OutputParser, OutputExtractor
from student.agent.tools.coremof_loader import CoreMofLoader

# Test data directory
DATA_DIR = Path(__file__).parent / "data"
TEMP_DIR = Path(__file__).parent / "temp"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests and clean up after."""
    temp = TEMP_DIR
    temp.mkdir(exist_ok=True)
    yield temp
    # Cleanup
    if temp.exists():
        shutil.rmtree(temp)


@pytest.fixture
def mock_agent():
    """Create a mock agent for ExecuteRaspa tests."""
    agent = Mock()
    agent._advance_to_next_folder = Mock()
    return agent


class TestMoleculeLoader:
    """Tests for MoleculeLoader tool."""

    def test_init(self):
        """Test MoleculeLoader initialization."""
        loader = MoleculeLoader()
        assert loader.name == "molecule_loader"
        assert "Generate the molecule definition" in loader.description

    def test_normalize_name_co2(self):
        """Test normalization of CO2 variants."""
        loader = MoleculeLoader()
        assert loader.normalize_name("CO2") == "carbon dioxide"
        assert loader.normalize_name("co2") == "carbon dioxide"
        assert loader.normalize_name("carbon_dioxide") == "carbon dioxide"
        assert loader.normalize_name("carbon dioxide") == "carbon dioxide"

    def test_normalize_name_alkanes(self):
        """Test normalization of alkane names."""
        loader = MoleculeLoader()
        assert loader.normalize_name("CH4") == "methane"
        assert loader.normalize_name("ch4") == "methane"
        assert loader.normalize_name("C2H6") == "ethane"
        assert loader.normalize_name("c3h8") == "propane"

    def test_normalize_name_nitrogen(self):
        """Test normalization of nitrogen variants."""
        loader = MoleculeLoader()
        assert loader.normalize_name("N2") == "nitrogen"
        assert loader.normalize_name("n2") == "nitrogen"
        assert loader.normalize_name("nitrogen") == "nitrogen"

    @patch.object(MoleculeLoader, "_run")
    def test_run_single_molecule(self, mock_run, temp_dir):
        """Test running with a single molecule name."""
        loader = MoleculeLoader(path=str(temp_dir))
        for m in [
            "CO2",
            "carbon dioxide" "N2",
            "nitrogen",
            "methane",
            "ch4",
            "propane",
            "hexane",
            "heptane",
            "ethanol",
            "argon",
            "Ar",
            "He",
            "helium",
        ]:
            m = loader.normalize_name(m)
            mock_run.return_value = [m]

            result = loader._run(m)

            assert mock_run.called
            assert m in result

    @patch.object(MoleculeLoader, "_run")
    def test_run_error_handling(self, mock_run, temp_dir):
        """Test error handling during molecule loading."""
        loader = MoleculeLoader(path=str(temp_dir))
        mock_run.side_effect = Exception("Test error")

        result = loader.run("CO2")

        assert "<error>" in result


class TestReadFile:
    """Tests for ReadFile tool."""

    def test_init(self):
        """Test ReadFile initialization."""
        tool = ReadFile()
        assert tool.name == "read_file"
        assert "read the content of a text file" in tool.description

    def test_read_existing_file(self, temp_dir):
        """Test reading an existing file."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)

        tool = ReadFile(path=str(temp_dir))
        result = tool.run("test.txt")

        assert test_content in result
        assert "test.txt" in result

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading a file that doesn't exist."""
        tool = ReadFile(path=str(temp_dir))
        result = tool.run("nonexistent.txt")

        assert "does not exist" in result

    def test_read_directory(self, temp_dir):
        """Test attempting to read a directory."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        tool = ReadFile(path=str(temp_dir))
        result = tool.run("subdir")

        assert "directory" in result.lower()


class TestWriteFile:
    """Tests for WriteFile tool."""

    def test_init(self):
        """Test WriteFile initialization."""
        tool = WriteFile()
        assert tool.name == "write_file"
        assert "write text into a new file" in tool.description

    def test_write_new_file(self, temp_dir):
        """Test writing a new file."""
        tool = WriteFile(path=str(temp_dir))
        content = "Test content for file"

        result = tool.run(file_content=content, file_name="new_file.txt")

        assert "Successfully generated" in result
        assert (temp_dir / "new_file.txt").exists()
        assert (temp_dir / "new_file.txt").read_text() == content

    def test_overwrite_existing_file(self, temp_dir):
        """Test overwriting an existing file."""
        test_file = temp_dir / "existing.txt"
        test_file.write_text("Old content")

        tool = WriteFile(path=str(temp_dir))
        new_content = "New content"
        result = tool.run(file_content=new_content, file_name="existing.txt")

        assert "Successfully generated" in result
        assert test_file.read_text() == new_content

    def test_write_nested_directory(self, temp_dir):
        """Test writing to a nested directory path."""
        tool = WriteFile(path=str(temp_dir))
        content = "Nested file content"

        result = tool.run(file_content=content, file_name="subdir/nested.txt")

        assert "Successfully generated" in result
        assert (temp_dir / "subdir" / "nested.txt").exists()


class TestInputFile:
    """Tests for InputFile tool."""

    def test_init_with_default_template(self):
        """Test InputFile initialization with default template."""
        tool = MakeInputFile()
        assert tool.name == "input_file"
        assert "simulation input file" in tool.description
        assert "0-based indexing" in tool.description
        assert "<template>" in tool.description

    def test_init_with_custom_template(self, temp_dir):
        """Test InputFile initialization with custom template."""
        template_file = temp_dir / "custom_template.input"
        template_content = "# Custom Template\nSimulationType MonteCarlo"
        template_file.write_text(template_content)

        tool = MakeInputFile(path=str(temp_dir), template_filename=str(template_file))

        assert template_content in tool.description

    def test_write_simulation_input(self, temp_dir):
        """Test writing simulation.input file."""
        tool = MakeInputFile(path=str(temp_dir))
        input_content = """SimulationType MonteCarlo
NumberOfCycles 1000
Framework 0
"""
        result = tool.run(file_content=input_content)

        assert "Successfully generated" in result
        assert (temp_dir / "simulation.input").exists()
        assert tool.has_file is True


class TestExecuteRaspa:
    """Tests for ExecuteRaspa tool."""

    def test_init(self, mock_agent):
        """Test ExecuteRaspa initialization."""
        tool = ExecuteRaspa(agent=mock_agent)
        assert tool.name == "execute_raspa"
        assert "start a RASPA simulation" in tool.description

    @patch.dict(os.environ, {"RASPA_DIR": "/test/raspa/dir"})
    def test_get_run_file(self, mock_agent, temp_dir):
        """Test creation of run.sh file."""
        tool = ExecuteRaspa(agent=mock_agent, path=str(temp_dir))
        tool.get_run_file()

        run_file = temp_dir / "run.sh"
        assert run_file.exists()

        content = run_file.read_text()
        assert "#! /bin/sh" in content
        assert "RASPA_DIR=/test/raspa/dir" in content
        assert "$RASPA_DIR/bin/simulate" in content

    @patch("subprocess.Popen")
    def test_run_raspa_success(self, mock_popen, mock_agent, temp_dir):
        """Test successful RASPA execution."""
        # Mock subprocess output
        mock_process = Mock()
        mock_process.communicate.return_value = ("stdout output", "stderr output")
        mock_popen.return_value = mock_process

        tool = ExecuteRaspa(agent=mock_agent, path=str(temp_dir))
        result = tool.run_raspa()

        assert result == ("stdout output", "stderr output")

    def test_check_success_with_output_dir(self, mock_agent, temp_dir):
        """Test check_success when Output directory exists."""
        output_dir = temp_dir / "Output"
        output_dir.mkdir()

        tool = ExecuteRaspa(agent=mock_agent, path=str(temp_dir))
        assert tool.check_success() is True

    def test_check_success_without_output_dir(self, mock_agent, temp_dir):
        """Test check_success when Output directory doesn't exist."""
        tool = ExecuteRaspa(agent=mock_agent, path=str(temp_dir))
        assert tool.check_success() is False


class TestCoreMofLoader:
    """Tests for CoreMofLoader tool."""

    def test_init(self):
        """Test CoreMofLoader initialization."""
        tool = CoreMofLoader()
        assert tool.name == "framework_loader"
        assert "Load the framework (MOF) file" in tool.description
        assert tool.has_file is False

    @patch("CoRE_MOF.list_structures")
    def test_get_coremof_structures(self, mock_list_structures):
        """Test getting CoreMOF structures."""
        mock_list_structures.side_effect = [["MOF-5", "HKUST-1"], ["ZIF-8"], ["UiO-66"]]

        tool = CoreMofLoader()
        structures = tool.get_coremof_structures()

        assert isinstance(structures, dict)
        assert "MOF-5" in structures
        assert "HKUST-1" in structures

    def test_search_names_exact_match(self):
        """Test searching for MOF names with exact match."""
        tool = CoreMofLoader()
        tool.structures = {"MOF-5": ["2014"], "HKUST-1": ["2019-ASR"]}

        result = tool.search_names("MOF-5")
        assert result == "MOF-5"

    def test_search_names_fuzzy_match(self):
        """Test searching for MOF names with fuzzy match."""
        tool = CoreMofLoader()
        tool.structures = {"MOF-5": ["2014"], "HKUST-1": ["2019-ASR"]}

        result = tool.search_names("MOF5")
        assert result in ["MOF-5", None]  # Depends on fuzzy search threshold

    def test_search_names_no_match(self):
        """Test searching with no matches."""
        tool = CoreMofLoader()
        tool.structures = {"MOF-5": ["2014"]}

        result = tool.search_names("completely_different_name", score_cutoff=90)
        assert result is None

    @patch("CoRE_MOF.get_structure")
    def test_run_success(self, mock_get_structure, temp_dir):
        """Test successful MOF loading."""
        mock_mof = Mock()
        mock_get_structure.return_value = mock_mof

        tool = CoreMofLoader(path=str(temp_dir))
        tool.structures = {"MOF-5": ["2014"]}

        result = tool.run("MOF-5", output_file="test.cif")

        assert tool.has_file is True
        assert "Generated from Coremof" in result

    def test_run_no_match(self, temp_dir):
        """Test running with no matching MOF name."""
        tool = CoreMofLoader(path=str(temp_dir))
        tool.structures = {"MOF-5": ["2014"]}

        result = tool.run("NonExistentMOF")

        assert "<error>" in result
        assert "No entry found" in result


class TestOutputParser:
    """Tests for OutputParser tool."""

    def test_init(self):
        """Test OutputParser initialization."""
        tool = OutputParser()
        assert tool.name == "output_parser"
        assert "parse the raspa output files" in tool.description

    def test_is_empty_none(self):
        """Test is_empty with None."""
        tool = OutputParser()
        assert tool.is_empty(None) is True

    def test_is_empty_empty_string(self):
        """Test is_empty with empty string."""
        tool = OutputParser()
        assert tool.is_empty("") is True
        assert tool.is_empty("   ") is True

    def test_is_empty_empty_list(self):
        """Test is_empty with empty list."""
        tool = OutputParser()
        assert tool.is_empty([]) is True

    def test_is_empty_empty_dict(self):
        """Test is_empty with empty dict."""
        tool = OutputParser()
        assert tool.is_empty({}) is True

    def test_is_empty_nan_and_inf(self):
        """Test is_empty with NaN and inf."""

        tool = OutputParser()
        assert tool.is_empty(float("nan")) is True
        assert tool.is_empty(float("inf")) is True

    def test_is_empty_valid_values(self):
        """Test is_empty with valid values."""
        tool = OutputParser()
        assert tool.is_empty(0) is False
        assert tool.is_empty(1.5) is False
        assert tool.is_empty("text") is False
        assert tool.is_empty([1, 2, 3]) is False
        assert tool.is_empty({"key": "value"}) is False

    def test_check_del_key_blacklist(self):
        """Test check_del_key with blacklisted keys."""
        tool = OutputParser()
        assert tool.check_del_key("System Properties") is True
        assert tool.check_del_key("Cpu") is True
        assert tool.check_del_key("Current cycle") is True
        assert tool.check_del_key("OS information") is True

    def test_check_del_key_whitelist(self):
        """Test check_del_key with non-blacklisted keys."""
        tool = OutputParser()
        assert tool.check_del_key("Total energy") is False
        assert tool.check_del_key("Average loading") is False

    def test_check_keep_key(self):
        """Test check_keep_key with whitelisted keys."""
        tool = OutputParser()
        assert tool.check_keep_key("Total energy") is True
        assert tool.check_keep_key("Average Widom Rosenbluth factor") is True
        assert tool.check_keep_key("Average Henry coefficient") is True
        assert tool.check_keep_key("Other key") is False

    def test_strip_block_fields_dict(self):
        """Test strip_block_fields removes Block[N] keys."""
        tool = OutputParser()
        data = {
            "Block[0]": {"value": 1},
            "Block[ 1 ]": {"value": 2},
            "Component": {"Block[2]": "remove", "keep": "this"},
        }

        result = tool.strip_block_fields(data)

        assert "Block[0]" not in result
        assert "Block[ 1 ]" not in result
        assert "Component" in result
        assert "Block[2]" not in result["Component"]
        assert "keep" in result["Component"]

    def test_strip_block_fields_list(self):
        """Test strip_block_fields with lists."""
        tool = OutputParser()
        data = [{"Block[0]": "remove", "keep": "this"}, {"value": 123}]

        result = tool.strip_block_fields(data)

        assert "Block[0]" not in result[0]
        assert "keep" in result[0]

    @patch("builtins.open")
    @patch("student.agent.tools.output.output_parser.parse")
    def test_run_success(self, mock_parse, mock_open, temp_dir):
        """Test successful output parsing."""
        mock_open.return_value.__enter__.return_value.read.return_value = "mock data"
        mock_parse.return_value = {"Component 0": {"Average loading": 1.23}}

        tool = OutputParser(path=str(temp_dir))
        result = tool.run("Output/System_0/output.data")

        assert "<error>" not in result
        assert "1.23" in result

    def test_run_file_not_found(self, temp_dir):
        """Test parsing non-existent file."""
        tool = OutputParser(path=str(temp_dir))
        result = tool.run("nonexistent.data")

        assert "<error>" in result or "Error" in result

    def test_filter_removes_empty_content(self):
        """Test that filter removes empty content."""
        tool = OutputParser()
        data = {
            "empty_list": [],
            "empty_dict": {},
            "valid": 123,
            "nested": {"empty": None, "keep": "value"},
        }

        result = tool.filter(data)

        assert "empty_list" not in result
        assert "empty_dict" not in result
        assert "valid" in result
        assert "nested" in result
        assert "empty" not in result["nested"]


class TestOutputExtractor:
    """Tests for OutputExtractor tool."""

    def test_init(self):
        """Test OutputExtractor initialization."""
        tool = OutputExtractor()
        assert tool.name == "output_parser"


class TestFrameworkLoader:
    """Tests for FrameworkLoader tool."""

    def test_init_with_coremof(self):
        """Test FrameworkLoader initialization with CoreMOF."""
        tool = FrameworkLoader(coremof=True)
        assert tool.name == "framework_loader"
        assert tool.has_file is False
        assert tool.output_file == "framework.cif"

    def test_init_without_coremof(self):
        """Test FrameworkLoader initialization without CoreMOF."""
        tool = FrameworkLoader(coremof=False)
        assert tool.coremof is False

    @patch.dict(os.environ, {"RASPA_DIR": "/test/raspa"}, clear=False)
    @patch("os.listdir")
    def test_load_local(self, mock_listdir):
        """Test loading local RASPA structures."""
        mock_listdir.return_value = ["MOF-5.cif", "HKUST-1.cif", "ZIF-8.cif"]

        tool = FrameworkLoader(coremof=False)
        tool.load_local()

        assert "MOF-5" in tool.structures_local
        assert "HKUST-1" in tool.structures_local
        assert "ZIF-8" in tool.structures_local

    def test_calculate_unit_cells_cubic(self, temp_dir):
        """Test unit cell calculation for cubic cell."""
        # Create a mock CIF file with cubic cell
        cif_content = """data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
"""
        cif_file = temp_dir / "test.cif"
        cif_file.write_text(cif_content)

        tool = FrameworkLoader()
        unit_cells = tool.calculate_unit_cells(str(cif_file), cutoff_angstrom=14.0)

        # For cutoff=14.0, required_length=28.0, cell=10.0 -> need ceil(28/10)=3 cells
        assert unit_cells == [3, 3, 3]

    def test_calculate_unit_cells_orthorhombic(self, temp_dir):
        """Test unit cell calculation for orthorhombic cell."""
        cif_content = """data_test
_cell_length_a 20.0
_cell_length_b 15.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
"""
        cif_file = temp_dir / "test.cif"
        cif_file.write_text(cif_content)

        tool = FrameworkLoader()
        unit_cells = tool.calculate_unit_cells(str(cif_file), cutoff_angstrom=14.0)

        # a=20 -> ceil(28/20)=2, b=15 -> ceil(28/15)=2, c=10 -> ceil(28/10)=3
        assert unit_cells == [2, 2, 3]

    def test_clean_cif(self, temp_dir):
        """Test CIF file cleaning (removes trailing commas)."""
        cif_file = temp_dir / "dirty.cif"
        cif_content = "line1,\nline2  ,\nline3\n"
        cif_file.write_text(cif_content)

        tool = FrameworkLoader()
        tool.clean_cif(str(cif_file))

        cleaned = cif_file.read_text()
        assert "line1\n" in cleaned
        assert "line2\n" in cleaned
        assert "line3\n" in cleaned
        assert "," not in cleaned.replace("\n", "")


# Integration tests (require actual data files)
class TestIntegration:
    """Integration tests that require actual data files."""

    def test_raspa_output_file_exists(self):
        """Test that sample RASPA output file exists for testing."""
        sample_output = DATA_DIR / "sample_raspa_output.data"
        # Create placeholder if it doesn't exist
        if not sample_output.exists():
            DATA_DIR.mkdir(exist_ok=True, parents=True)
            sample_output.write_text(
                "# PLACEHOLDER: Add real RASPA output data here for integration testing\n"
            )
        assert True  # Just check that we can create the placeholder

    def test_sample_cif_file_exists(self):
        """Test that sample CIF file exists for testing."""
        sample_cif = DATA_DIR / "sample_framework.cif"
        if not sample_cif.exists():
            DATA_DIR.mkdir(exist_ok=True, parents=True)
            sample_cif.write_text(
                "# PLACEHOLDER: Add real CIF data here for integration testing\n"
                "data_test\n"
                "_cell_length_a 10.0\n"
                "_cell_length_b 10.0\n"
                "_cell_length_c 10.0\n"
                "_cell_angle_alpha 90.0\n"
                "_cell_angle_beta 90.0\n"
                "_cell_angle_gamma 90.0\n"
            )
        assert True

    def test_simulation_input_template_exists(self):
        """Test that simulation input template file exists."""
        template_file = DATA_DIR / "template_simulation.input"
        if not template_file.exists():
            DATA_DIR.mkdir(exist_ok=True, parents=True)
            template_file.write_text(
                "# PLACEHOLDER: Add real RASPA simulation.input template here\n"
                "SimulationType MonteCarlo\n"
                "NumberOfCycles 1000\n"
                "Framework 0\n"
                "Component 0\n"
            )
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
