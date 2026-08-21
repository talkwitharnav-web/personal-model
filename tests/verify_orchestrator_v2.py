"""Verify the orchestrator can serve BOTH model generations.

chat.py currently serves combined_best_model.pt, so the overriding requirement is that v1 still
loads exactly as before. The v2 support is additive.

Detection is structural, not filename-based: v2 uses RoPE and therefore never creates
position_embedding, so that key is absent from a v2 state_dict and present in every v1 one. A
renamed or copied checkpoint still resolves correctly.

Does NOT touch the FAISS index (3.19 GB) - only the model-loading path, so this is safe to run
while a training job holds most of the machine's RAM.
"""
import sys
from pathlib import Path

import torch

from orchestrator import config
from orchestrator.generation import Generator, load_checkpoint

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== detection ===")
    v1_path = Path("checkpoints/combined_best_model.pt")
    v2_path = Path("checkpoints/pretrain_v2_best.pt")

    if v1_path.exists():
        state, is_v2 = load_checkpoint(v1_path, "cpu")
        check("v1 checkpoint detected as v1", not is_v2)
        check("v1 state_dict has a position table",
              any(k.startswith("position_embedding.") for k in state))
        del state
    else:
        print(f"  SKIP  {v1_path} missing")

    if v2_path.exists():
        state, is_v2 = load_checkpoint(v2_path, "cpu")
        check("v2 checkpoint detected as v2", is_v2)
        check("v2 state_dict has NO position table",
              not any(k.startswith("position_embedding.") for k in state))
        check("v2 training dict was unwrapped to the weights",
              any(k.startswith("token_embedding.") for k in state))
        del state
    else:
        print(f"  SKIP  {v2_path} missing (the pretrain has not written a best checkpoint yet)")

    print("\n=== v1 still serves, unchanged ===")
    if v1_path.exists():
        generator = Generator(device="cpu")
        check("Generator picked a checkpoint", generator.checkpoint is not None,
              str(generator.checkpoint))
        expected_v2 = "finetune_v2" in str(generator.checkpoint)
        check("architecture matches the checkpoint", generator.is_v2 == expected_v2,
              f"is_v2={generator.is_v2}")
        expected_tokenizer = (config.TOKENIZER_PATH_V2 if generator.is_v2
                              else config.TOKENIZER_PATH)
        check("tokenizer matches the checkpoint generation",
              Path(generator.tokenizer_path) == Path(expected_tokenizer),
              str(generator.tokenizer_path))
        check("vocab is 32,000", generator.tokenizer.get_vocab_size() == 32000)

        prompt, used = generator.build_prompt(
            "Who wrote Hamlet?",
            [(0.9, "Hamlet is a tragedy written by William Shakespeare around 1600.")])
        check("build_prompt used the passage", used == 1, f"{used} passage(s)")
        check("prompt has the Assistant marker", prompt.rstrip().endswith("Assistant:"))

        with torch.no_grad():
            pieces = list(generator.stream(prompt, temperature=0.2))
        answer = "".join(pieces).strip()
        print(f"    answer: {answer[:120]!r}")
        check("generation produced text", len(answer) > 0)
        check("generation stopped before inventing a turn",
              not any(stop in answer for stop in config.STOP_SEQUENCES))
    else:
        print(f"  SKIP  {v1_path} missing")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
