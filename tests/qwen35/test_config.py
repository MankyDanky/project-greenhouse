"""Exercise custom-plan support and reject incompatible native-model switches."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reranker/engines/qwen35'))
import json
from shared import GroupSampler, summarize_plan, validate_native_config
from train import packing_signature

cfg = json.loads((Path(__file__).resolve().parents[2] / 'reranker/configs/backbones/qwen35_4b_lr5e-5.json').read_text())
validate_native_config(cfg)
for key, value in [('lora', True), ('bidirectional', True), ('query_factor', 2),
                   ('backbone_impl', 'upstream'), ('optimizer_state_sharding', True)]:
    try: validate_native_config({**cfg, key: value})
    except ValueError: pass
    else: raise AssertionError(f'{key} silently accepted')
custom = {**cfg, 'expected_steps': None, 'expected_plan_hash': None,
          'global_batch_groups': 4, 'pack_world': 1}
sampler = GroupSampler([7]*8, [[10]*7]*8, [3]*8, max_tokens=39312,
                       max_groups=2, global_groups=4, scaffold=14, query_factor=1)
plan = summarize_plan(sampler, custom)
assert plan['steps'] == 2 and plan['groups'] == 8 and plan['pairs'] == 64
try: summarize_plan(sampler, {**custom, 'expected_plan_hash': 'wrong'})
except ValueError: pass
else: raise AssertionError('Invalid reference plan accepted')
assert packing_signature(cfg, 2)['objective'] == 'lce_qwen35_tied_gap'
assert packing_signature(cfg, 2)['plan_hash'] == cfg['expected_plan_hash']
print('QWEN_CUSTOM_CONFIG_AND_PLAN_CHECKS_PASSED')
