"""Is the yes/no polarity failure a FORMAT problem or a KNOWLEDGE problem?

The distinction decides whether we act.

- If it is a **format** problem, the model has the facts but has not yet learned to express them as
  Yes/No. More fine-tuning fixes it for free, because that is exactly what the fine-tune teaches.
- If it is a **knowledge** problem, the model never learned the fact during pretraining. No amount
  of yes/no formatting will install it. The fine-tune will teach it to answer *confidently* and
  still be wrong - which is the §8 result all over again (a repetition penalty removed the loop but
  not the ignorance).

The reason to suspect knowledge: `data/prepare_yesno.py` builds its contrastive pairs from the
cleaned **Wikipedia** corpus, so every pair it teaches looks like "Was Mystery Road released in
1989?" - dates, names, places. It never once contrasts "is the sun hot" against "is the sun cold",
because Wikipedia does not write sentences like that. The wide peek already found everyday
knowledge to be the model's weakest group for the same reason.

So this splits the pairs by that axis and compares. Per-pair numbers are printed, not just a group
mean - averaging populations that differ is the recurring bug of this project's tooling.

CPU-only and read-only, so it is safe to run while the fine-tune has the GPU.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import DarkGLM
from training.finetune_v2 import v2_config

# (question, expected, context) triples. Each pair is the SAME subject asked two ways, one true one
# false, so the only thing that can move the answer is the model reading the predicate.
#
# The context matters enormously and was the whole discovery here: EVERY yes/no training example is
#   User: {question}\n\nContext: {passage}\nAssistant: Yes
# with the passage containing the answer. The set teaches READING COMPREHENSION, not recall. Asking
# these questions bare tests a task the model was never trained on, so both conditions are measured.
ENCYCLOPEDIC = [
    ("Was the Tesla coil invented by Nikola Tesla?", "Yes",
     "The Tesla coil is an electrical resonant transformer circuit designed by inventor "
     "Nikola Tesla in 1891."),
    ("Was the Tesla coil invented by Alexander Fleming?", "No",
     "The Tesla coil is an electrical resonant transformer circuit designed by inventor "
     "Nikola Tesla in 1891."),
    ("Was penicillin discovered by Alexander Fleming?", "Yes",
     "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming."),
    ("Was penicillin discovered by Nikola Tesla?", "No",
     "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming."),
    ("Is Paris in France?", "Yes",
     "Paris is the capital and most populous city of France, situated on the river Seine."),
    ("Is Paris in Japan?", "No",
     "Paris is the capital and most populous city of France, situated on the river Seine."),
    ("Did the Second World War begin in 1939?", "Yes",
     "The Second World War began on 1 September 1939 with the German invasion of Poland."),
    ("Did the Second World War begin in 1975?", "No",
     "The Second World War began on 1 September 1939 with the German invasion of Poland."),
    ("Is the Eiffel Tower in Paris?", "Yes",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."),
    ("Is the Eiffel Tower in Berlin?", "No",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."),
]

EVERYDAY = [
    ("Is the sun hot?", "Yes",
     "The Sun is a star at the centre of the Solar System. Its surface temperature is about "
     "5,500 degrees Celsius, and it is extremely hot."),
    ("Is the sun cold?", "No",
     "The Sun is a star at the centre of the Solar System. Its surface temperature is about "
     "5,500 degrees Celsius, and it is extremely hot."),
    ("Do fish have gills?", "Yes",
     "Fish are aquatic animals that breathe using gills. They do not have lungs."),
    ("Do fish have lungs?", "No",
     "Fish are aquatic animals that breathe using gills. They do not have lungs."),
    ("Is ice cold?", "Yes",
     "Ice is water frozen into a solid state, forming at or below zero degrees Celsius. "
     "It is cold to the touch."),
    ("Is ice hot?", "No",
     "Ice is water frozen into a solid state, forming at or below zero degrees Celsius. "
     "It is cold to the touch."),
    ("Do birds have wings?", "Yes",
     "Birds are warm-blooded vertebrates characterised by feathers and a pair of wings. "
     "They walk on two legs."),
    ("Do birds have four legs?", "No",
     "Birds are warm-blooded vertebrates characterised by feathers and a pair of wings. "
     "They walk on two legs."),
    ("Is water wet?", "Yes",
     "Water is a liquid at room temperature. It is wet and makes other things wet on contact."),
    ("Is water dry?", "No",
     "Water is a liquid at room temperature. It is wet and makes other things wet on contact."),
]


def load(checkpoint_path, device="cpu"):
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    config = v2_config()
    model = DarkGLM(config)
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, tokenizer, blob.get("step", -1)


@torch.no_grad()
def p_yes(model, tokenizer, question, device="cpu", context=None):
    """P(Yes) vs P(No) for the FIRST answer token, normalised against each other.

    Normalising matters: a pretrained model puts almost no mass on either word, and raw
    probabilities of 0.0001 vs 0.0007 say nothing about which way it leans. The ratio does.

    `context` reproduces the exact training format when supplied. Getting this wrong is not a
    detail - without it the prompt is a shape the model has never been trained to answer.
    """
    if context:
        prompt = f"User: {question}\n\nContext: {context}\nAssistant:"
    else:
        prompt = f"User: {question}\nAssistant:"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    logits = model(x)[0, -1]
    probs = F.softmax(logits.float(), dim=-1)

    def mass(word):
        total = 0.0
        for variant in (f" {word}", word, f" {word.lower()}", word.lower()):
            encoded = tokenizer.encode(variant).ids
            if encoded:
                total = max(total, probs[encoded[0]].item())
        return total

    yes, no = mass("Yes"), mass("No")
    return yes / (yes + no) if (yes + no) > 0 else 0.5


def run_group(name, pairs, model, tokenizer, device, with_context):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    gaps = []
    for i in range(0, len(pairs), 2):
        q_true, _, ctx_true = pairs[i]
        q_false, _, ctx_false = pairs[i + 1]
        p_true = p_yes(model, tokenizer, q_true, device, ctx_true if with_context else None)
        p_false = p_yes(model, tokenizer, q_false, device, ctx_false if with_context else None)
        gap = p_true - p_false
        gaps.append(gap)
        verdict = "ok  " if gap > 0.15 else ("weak" if gap > 0.02 else "FAIL")
        print(f"  {verdict}  gap {gap:+.3f}   P(Yes) {p_true:.3f} '{q_true}'")
        print(f"                       P(Yes) {p_false:.3f} '{q_false}'")
    mean = sum(gaps) / len(gaps)
    good = sum(1 for g in gaps if g > 0.15)
    print(f"  -> mean gap {mean:+.3f} | {good}/{len(gaps)} pairs separate cleanly")
    return mean, good, len(gaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, tokenizer, step = load(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint} (step {step:,})")
    print("A 'gap' is P(Yes | true version) - P(Yes | false version).")
    print("Near zero means the model answered the SHAPE of the question, not its content.")

    results = {}
    for with_context in (False, True):
        mode = "WITH CONTEXT (the exact training format - what RAG will supply)" if with_context \
            else "NO CONTEXT (bare recall - a format the yes/no set never trains)"
        print(f"\n\n{'#' * 78}\n### {mode}\n{'#' * 78}")
        enc = run_group("ENCYCLOPEDIC", ENCYCLOPEDIC, model, tokenizer, args.device, with_context)
        eve = run_group("EVERYDAY", EVERYDAY, model, tokenizer, args.device, with_context)
        results[with_context] = (enc, eve)

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    (enc_n, eve_n), (enc_c, eve_c) = results[False], results[True]
    print(f"  {'':<14}{'no context':>22}{'with context':>22}")
    print(f"  {'encyclopedic':<14}{enc_n[0]:>+15.3f} {enc_n[1]}/{enc_n[2]}"
          f"{enc_c[0]:>+16.3f} {enc_c[1]}/{enc_c[2]}")
    print(f"  {'everyday':<14}{eve_n[0]:>+15.3f} {eve_n[1]}/{eve_n[2]}"
          f"{eve_c[0]:>+16.3f} {eve_c[1]}/{eve_c[2]}")

    ctx_mean = (enc_c[0] + eve_c[0]) / 2
    bare_mean = (enc_n[0] + eve_n[0]) / 2
    print()
    if ctx_mean > bare_mean + 0.10:
        print("  -> The model reads the CONTEXT. Separation appears when the prompt matches the\n"
              "     training format, and collapses without it. That is exactly what the yes/no set\n"
              "     teaches - reading comprehension, not recall - and exactly what RAG supplies at\n"
              "     serving time. The bare-prompt failure is a TEST ARTIFACT, not a model defect.")
    elif ctx_mean < 0.05:
        print("  -> Context does NOT help. The model is answering the shape of the question in both\n"
              "     conditions, so this is a real defect and not a prompt-format artifact.\n"
              "     Re-measure later in the run before acting - this may just be immaturity.")
    else:
        print("  -> Context helps somewhat but not decisively. Re-measure later in the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
