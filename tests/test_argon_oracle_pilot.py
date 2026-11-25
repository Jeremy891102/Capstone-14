"""Pilot must stop on failed calls; no paid calls."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('argon_pilot', Path(__file__).resolve().parents[1]/'scripts/run_argon_oracle_pilot.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_failed_pilot_stops_before_next_call():
    with pytest.raises(KeyboardInterrupt):
        mod.on_done({'call_id':'x','status':'failed'})


def test_succeeded_pilot_continues():
    mod.on_done({'call_id':'x','status':'succeeded'})


@pytest.mark.parametrize('usage', [
    {'promptTokenCount':500000,'candidatesTokenCount':0},
    {},
])
def test_spend_or_unknown_usage_stops_next_request(tmp_path,monkeypatch,usage):
    from types import SimpleNamespace
    class FakeProvider:
        def __init__(self,*args):
            self.http=SimpleNamespace(post=lambda *a:{'usageMetadata':usage})
        def generate(self,request,*,timeout_s):
            return self.http.post('https://example.test',{'x-goog-api-key':'DO_NOT_SAVE'},
                                  {'contents':[{'parts':[{'text':'test'}]}]},timeout_s,1000)
    monkeypatch.setattr(mod,'VertexGeminiProvider',FakeProvider)
    cfg=SimpleNamespace(provider=SimpleNamespace(model='test',gemini=None))
    provider=mod.factory(cfg,SimpleNamespace(path=tmp_path))
    request=SimpleNamespace(call_id='test-call')
    provider.generate(request,timeout_s=1)
    with pytest.raises(KeyboardInterrupt):
        provider.generate(request,timeout_s=1)
    assert len(list((tmp_path/'api_raw').glob('*_request.json')))==1
    assert 'DO_NOT_SAVE' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
