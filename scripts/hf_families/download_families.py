"""Download the App. D families at pinned revisions and verify every file against the Hub's sha256.

Weights only (model*.safetensors when present, else pytorch_model*.bin) plus configs and
tokenizer files. Writes ROOT/models/<org>__<name>/ and ROOT/download_manifest.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

FAMILIES = {
    "mistral_7b": {
        "mistralai/Mistral-7B-v0.1": "27d67f1b5f", "mistralai/Mistral-7B-Instruct-v0.1": "ec5deb64f2",
        "alignment-handbook/zephyr-7b-sft-full": "92f9fac452", "HuggingFaceH4/zephyr-7b-beta": "892b3d7a7b",
        "openchat/openchat_3.5": "0fc98e3242", "berkeley-nest/Starling-LM-7B-alpha": "1dddf3b95b",
        "openchat/openchat-3.5-0106": "ff058fda49", "Nexusflow/Starling-LM-7B-beta": "39a4d50147",
        "teknium/OpenHermes-2-Mistral-7B": "4c6e34123b", "teknium/OpenHermes-2.5-Mistral-7B": "24c0bea14d",
    },
    "llama2_7b": {
        "meta-llama/Llama-2-7b-hf": "01c7f73d77", "meta-llama/Llama-2-7b-chat-hf": "f5db02db72",
        "NousResearch/Nous-Hermes-llama-2-7b": "cbd525ad0d", "lmsys/vicuna-7b-v1.5": "3321f76e3f",
        "Atulit23/meta-llama-indian-constitution-chat": "9fe0b5e808",
        "BLACKBUN/llama-2-7b-pubmed-qa-211k": "1614303337", "QIAIUNCC/EYE-Llama_gqa": "60135e54cd",
        "QIAIUNCC/EYE-Llama_qa": "ef5d2b6ddc", "bineric/NorskGPT-Llama-7B-v0.1": "4772717d0c",
        "whatdhack/Llama-2-7b-chat-hf-oasst1-ft-sg": "4476982869",
    },
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()

def download_repo(api, repo: str, prefix: str, dest: Path) -> dict:
    """Download one repo at its pinned revision into ``dest`` and verify every LFS file."""
    info = api.model_info(repo, files_metadata=True)
    assert info.sha.startswith(prefix), f"{repo} moved: head {info.sha} != pinned {prefix}"
    names = {s.rfilename: s for s in info.siblings}
    st = sorted(n for n in names if n.startswith("model") and n.endswith(".safetensors"))
    weights = st or sorted(n for n in names if n.startswith("pytorch_model") and n.endswith(".bin"))
    extras = sorted(n for n in names if "/" not in n and (n.endswith(".json") or n == "tokenizer.model"))
    files = weights + extras
    t0 = time.time()
    snapshot_download(repo, revision=info.sha, local_dir=dest, allow_patterns=files)
    checks = {}
    for n in files:
        lfs = names[n].lfs
        if lfs is not None:
            got = sha256(dest / n)
            assert got == lfs.sha256, f"{repo}/{n}: sha256 {got} != hub {lfs.sha256}"
            checks[n] = got
    return {
        "revision": info.sha, "weight_format": "safetensors" if st else "bin",
        "files": files, "lfs_sha256_verified": checks, "path": str(dest),
        "bytes": sum((dest / n).stat().st_size for n in files),
        "seconds": round(time.time() - t0, 1),
    }


def main(root: Path) -> None:
    api = HfApi()
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "families": {}}
    for family, repos in FAMILIES.items():
        manifest["families"][family] = {}
        for repo, prefix in repos.items():
            record = download_repo(api, repo, prefix, root / "models" / repo.replace("/", "__"))
            manifest["families"][family][repo] = record
            print(f"{repo}: {len(record['lfs_sha256_verified'])} LFS files verified, "
                  f"{record['bytes']/1e9:.1f} GB in {record['seconds']:.0f}s", flush=True)
            (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("ALL DOWNLOADED AND VERIFIED", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
