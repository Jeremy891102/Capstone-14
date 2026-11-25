"""No API calls: reservations prevent concurrent overspend and unknown usage halts work."""
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

spec = importlib.util.spec_from_file_location('argon_full',Path(__file__).resolve().parents[1]/'scripts/run_argon_oracle_full.py')
mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_concurrent_reservations_cannot_exceed_budget(tmp_path):
    budget=mod.Budget(tmp_path/'ledger.json',limit=1,spent=.2)
    def reserve(i):
        try:
            budget.reserve(str(i),.3)
            return True
        except mod.BudgetStop:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve,range(4)))==2
    assert budget.spent+sum(budget.reservations.values())<=1


def test_settled_cost_releases_reservation_and_unknown_halts(tmp_path):
    budget=mod.Budget(tmp_path/'ledger.json',limit=1,spent=.2)
    budget.reserve('a',.5);budget.settle('a',.1)
    assert budget.spent==pytest.approx(.3)
    assert budget.reservations=={}
    budget.reserve('b',.5);budget.settle('b',None)
    with pytest.raises(mod.BudgetStop):budget.reserve('c',.1)


def test_billable_usage_includes_thinking_and_cache():
    assert mod.gross_cost({'promptTokenCount':1000,'cachedContentTokenCount':200,'candidatesTokenCount':50,'thoughtsTokenCount':100})==pytest.approx(.00624)
    assert mod.gross_cost({}) is None
