"""Audit every real RLHN250K step against the frozen co-author LCE recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reranker/engines/ce'))
from dataset_nc import PairGroups,make_plan,plan_digest
from prompt_builder_nc import PairBuilder
from modeling_nanochat_ce import load_tokenizer

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--config', default=str(root/'configs/G_climbmix0924_rlhn250k.yaml'))
parser.add_argument('--train-parquet', help='Prepared training data; overrides the config path')
parser.add_argument('--model-name', help='Optional local copy of the pinned model assets for offline auditing')
args=parser.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text())
if args.train_parquet: cfg['train_parquet']=args.train_parquet
if args.model_name: cfg['model_name']=args.model_name
tok=load_tokenizer(cfg)
builder=PairBuilder(tok,prompt='nochat',max_doc_len=512,max_query_len=128)
groups=PairGroups(cfg['train_parquet'],builder,seed=17)
plan=make_plan(groups,builder,cfg)
assert len(plan)==1931
assert plan_digest(plan)=='04c557c35b37662326d3d96873aa13c7137cab4b614d5f9ebe7ecb272e86977e'
# Execute the co-author's actual sampler/packing source in a separate process,
# avoiding a comparison that accidentally uses our own implementation twice.
import subprocess
reference=str(root.parent/'nanochat_lce/src')
script='''import sys,json,hashlib
sys.path.insert(0,sys.argv[1])
from dataset_nc import PairGroups,GroupSampler
from stepping import pack_steps
from prompt_builder_nc import PairBuilder,setup_tokenizer
from transformers import AutoTokenizer
tok=setup_tokenizer(AutoTokenizer.from_pretrained(sys.argv[3], revision=sys.argv[4] or None))
b=PairBuilder(tok,prompt='nochat',max_doc_len=512,max_query_len=128)
d=PairGroups(sys.argv[2],b,seed=17)
s=GroupSampler(d.avail,d.neg_lens,d.query_lens,max_tokens=39312,max_groups=32,max_doc_len=512,max_query_len=128,k_choices=[7,15,23],seed=17,world_size=2,global_groups=128,scaffold=b.overhead)
plan=[[x for mb in step for x in mb] for step in pack_steps(s._build(0),128,2)]
print('REFERENCE='+json.dumps({'steps':len(plan),'hash':hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()}))
'''
text=subprocess.check_output([sys.executable,'-c',script,reference,str(Path(cfg['train_parquet']).resolve()),cfg['model_name'],cfg.get('model_revision') or ''],text=True)
ref=json.loads(next(line.removeprefix('REFERENCE=') for line in text.splitlines() if line.startswith('REFERENCE=')))
assert ref=={'steps':len(plan),'hash':plan_digest(plan)},ref
result={'steps':len(plan),'rows_in_dataset':len(groups),'groups_in_plan':sum(map(len,plan)),
        'sampled_pairs':sum(k+1 for step in plan for _,k,_ in step),
        'plan_sha256':plan_digest(plan),'reference_source':reference,'matched_all_step_memberships':True}
print('REAL_PLAN_AUDIT_PASSED '+json.dumps(result),flush=True)
