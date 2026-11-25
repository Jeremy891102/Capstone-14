"""Recording bootstrap and invalid-list guard; offline only."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('argon_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_argon_oracle.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_cluster_rate_keeps_recording_multiplicity():
    c=mod.Cluster(['a','a','b'],draws=2000,seed=1)
    estimate,bs=c.rate(np.array([1,1,0],bool),np.ones(3,bool))
    assert estimate==pytest.approx(2/3)
    assert set(np.round(np.unique(bs),8))=={0.,round(2/3,8),1.}


def test_invalid_or_abstention_is_not_empty_error_list():
    assert mod.labels([])==set()
    assert mod.labels(None) is None
    assert mod.labels('not_visible') is None
    assert mod.labels(['F','G'])==set()
    assert mod.labels(['Z']) is None
