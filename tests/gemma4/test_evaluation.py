"""CPU checks for native checkpoint scoring and exact shared LCE evaluation protocol."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Gemma4TextConfig, Gemma4ForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "reranker/engines/gemma4"
sys.path.insert(0, str(PACKAGE))
from modeling_gemma4_lce import Gemma4Reranker

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

evaluation = load("gemma_evaluation", REPO / "eval/backends/gemma4.py")


def native_check(root):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Gemma4TextConfig(vocab_size=128, vocab_size_per_layer_input=128,
        hidden_size=64, intermediate_size=128, num_hidden_layers=10,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=8, num_kv_shared_layers=5,
        layer_types=["sliding_attention"]*4+["full_attention"]+["sliding_attention"]*4+["full_attention"],
        sliding_window=32, use_double_wide_mlp=True, final_logit_softcapping=30.,
        tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    native = Gemma4ForCausalLM(config).float().eval()
    adapter = Gemma4Reranker(full=copy.deepcopy(native), yes_id=7, no_id=9)
    words = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "Query:", "Passage:", "Relevant:", "true", "x", "false"]
    words += [f"word{i}" for i in range(118)]
    backend = Tokenizer(WordLevel({s:i for i,s in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    tok.save_pretrained(root / "tokenizer")
    (root / "gemma_text_config.json").write_text(config.to_json_string())
    cfg = json.loads((REPO / "reranker/configs/backbones/gemma4_e2b_lr5e-5.json").read_text())
    checkpoint = root / "ckpt_2.pt"
    state = {"cfg": cfg, "step": 2, "packing": {"objective": "lce_gemma4_native_tied_softcap_gap",
        "plan_hash": cfg["expected_plan_hash"]}, "model": adapter.state_dict()}
    torch.save(state, checkpoint)
    docs = [" ".join(f"word{j%118}" for j in range(n)) for n in [1,4,17,70,3,9,25,5,60]]
    for dtype in [torch.float32, torch.bfloat16]:
        scorer = evaluation.GemmaScorer(checkpoint, "cpu", dtype)
        got = scorer("word3 word7", docs, 8, 512)
        expected = []
        # Independent full-vocabulary calculation using original trained FP32 rows,
        # matching the former evaluator's bf16-backbone/fp32-readout protocol.
        decoder = copy.deepcopy(native.model).to(dtype=dtype).eval()
        for start in range(0, len(docs), 8):
            seqs = [scorer.builder.build("word3 word7", d) for d in docs[start:start+8]]
            ids = torch.zeros((len(seqs), max(map(len, seqs))), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i,:len(seq)] = torch.tensor(seq); mask[i,:len(seq)] = 1
            with torch.inference_mode():
                hidden = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                last = hidden[torch.arange(len(seqs)), mask.sum(1)-1].float()
                full = torch.nn.functional.linear(last, native.lm_head.weight)
                full = 30 * torch.tanh(full / 30)
                expected.extend((full[:,7]-full[:,9]).tolist())
        torch.testing.assert_close(torch.tensor(got), torch.tensor(expected), atol=1e-4, rtol=1e-4)
        assert scorer.validated
    bad = copy.deepcopy(state); bad["model"].pop(next(iter(bad["model"])))
    torch.save(bad, root / "bad.pt")
    try: evaluation.GemmaScorer(root / "bad.pt", "cpu", torch.float32)
    except RuntimeError: pass
    else: raise AssertionError("Incomplete state accepted")
    print("GEMMA_NATIVE_EVAL_PARITY_PASSED", flush=True)


def harness_check(root, assets=None):
    path = REPO / "eval/rerank_eval.py"
    original = load("original_lce_harness", path)
    wrapped = evaluation.load_harness()
    assert original.main.__code__.co_code == wrapped.main.__code__.co_code
    assert Path(wrapped.main.__code__.co_filename) == path
    class FakeScorer:
        def __init__(self, *args): pass
        def __call__(self, query, docs, bs, max_len):
            # Many ties exercise the harness's actual TREC tie/rounding behavior.
            return [float(len(doc)%23)/7 for doc in docs]
    if assets is None:
        assets = root / "assets"
        assets.mkdir()
        rows = [
            {"query": {"qid": "1", "text": "test query"}, "candidates": [
                {"docid": str(i), "score": float(100-i), "doc": {"title": "Title", "text": "word " * (i%29+1)}}
                for i in range(100)]},
            {"query": {"qid": "2", "text": "short query"}, "candidates": [
                {"docid": "100", "score": 1., "doc": {"contents": "short passage"}}]},
            {"query": {"qid": "3", "text": "zero hits"}, "candidates": []},
        ]
        for ds, filename in original.QRELS.items():
            (assets / f"retrieve_results_{ds}_top100.jsonl").write_text(
                "".join(json.dumps(row)+"\n" for row in rows))
            (assets / filename).write_text("1 0 1 2\n1 0 4 1\n2 0 100 1\n3 0 200 1\n")
    assets = Path(assets).resolve()
    os.chdir(root)
    for label, harness in [("original",original),("gemma",wrapped)]:
        harness.BACKENDS = {"ours": FakeScorer}
        harness.HERE = str(root/label)
        harness.RUNS = str(assets)
        harness.QRELS_DIR = str(assets)
        for ds in ["dl19","dl20","dl21","dl22","dl23","covid","news","robust04","nfc","scifact","scidocs","fiqa"]:
            sys.argv = ["rerank_eval.py","--backend","ours","--dataset",ds,"--model","fixture",
                "--tag","parity","--out",str(root/label/"runs")]
            with contextlib.redirect_stdout(io.StringIO()):
                harness.main()
    for source in (root/"original/runs").glob("*.trec"):
        assert source.read_bytes() == (root/"gemma/runs"/source.name).read_bytes(),source.name
    compared = {}
    for source in (root/"original/results").glob("*.json"):
        x=json.loads(source.read_text()); y=json.loads((root/"gemma/results"/source.name).read_text())
        assert all(k in x for k in ("ndcg_cut_10", "map", "recip_rank")), x
        x.pop("seconds"); y.pop("seconds")
        assert x==y,source.name
        compared[x["dataset"]] = x
    assert len(list((root/"original/runs").glob("*.trec")))==12
    if os.environ.get("GEMMA_RUN_DIR"):
        (Path(os.environ["GEMMA_RUN_DIR"])/"eval_protocol_validation.json").write_text(json.dumps(compared, indent=2)+"\n")
    print("LCE_HARNESS_ALL_12_DATASETS_BYTE_IDENTICAL_PASSED", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, help='Optional directory of all 12 frozen candidate/qrels pairs')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        native_check(root)
        harness_check(root, args.assets)
    print("GEMMA_EVALUATION_CHECKS_PASSED", flush=True)
