"""
Shared test fixtures for quantum-chemistry package tests.
"""

import pytest
from pyscf import gto, scf

from quantum.chemistry.active_space import find_active_space_avas


@pytest.fixture
def h2_molecule():
    """Create a simple H2 molecule for testing."""
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.build()
    return mol


@pytest.fixture
def h2_scf(h2_molecule):
    """Create converged RHF calculation for H2."""
    mf = scf.RHF(h2_molecule)
    mf.kernel()
    return mf


@pytest.fixture
def h2o_molecule():
    """Create a water molecule for testing."""
    mol = gto.Mole()
    mol.atom = """
    O  0.0000  0.0000  0.0000
    H  0.7571  0.0000  0.5861
    H -0.7571  0.0000  0.5861
    """
    mol.basis = "sto-3g"
    mol.build()
    return mol


@pytest.fixture
def h2o_scf(h2o_molecule):
    """Create converged RHF calculation for water."""
    mf = scf.RHF(h2o_molecule)
    mf.kernel()
    return mf


@pytest.fixture
def fe_complex_molecule():
    """Create a simple iron complex for testing transition metal methods."""
    mol = gto.Mole()
    mol.atom = """
    Fe 0.0000  0.0000  0.0000
    O  1.5000  0.0000  0.0000
    O -1.5000  0.0000  0.0000
    O  0.0000  1.5000  0.0000
    O  0.0000 -1.5000  0.0000
    """
    mol.basis = "minao"  # Use minimal basis for fast testing
    mol.spin = 4  # High-spin Fe2+
    mol.build()
    return mol


@pytest.fixture
def fe_scf(fe_complex_molecule):
    """Create converged UHF calculation for iron complex."""
    mf = scf.UHF(fe_complex_molecule)
    mf.kernel()
    return mf


@pytest.fixture
def sample_active_space_result(h2o_scf, h2o_molecule):
    """Create a sample active space result for testing."""
    return find_active_space_avas(h2o_scf, threshold=0.2)
