"""Behavioural regression for v2 - the metric a fine-tune should actually be selected on.

Section 8's most expensive lesson: SQuAD v2's validation loss fell 3.2648 -> 0.3726, textbook,
while the model had in fact collapsed into refusing every question. "A single validation loss can
hide total behavioural collapse. Select on a behavioural metric."

So this scores the specific documented failure modes, one section of the onboarding each:

  echo        section 9 - "return the noun phrase immediately before 'is'". Two days of work went
              into killing this and it REGRESSED once already ("who is william shakespeare?" went
              from "an English playwright and poet" back to "William Shakespeare").
  polarity    section 9 - opposite questions on the same passage got identical probabilities to
              three decimals. BoolQ can never punish it, so only a probe like this catches it.
  binding     section 9 - two names present in the SAME context, each must bind to its own
              subject. This is the decisive one: it cannot be passed by word-matching.
  natural     section 9 - the skill existed only at trained phrasings. "is coconut high in
              saturated fat" vs "low" was the measured failure.
  length      section 9 - "training context must match deployment context in SHAPE and LENGTH".
              definitions v1 worked on one sentence and failed on three retrieved passages.
  stopping    section 7 - the stop token was never trained in v1, so the model invented its own
              next turn. Checks it terminates instead of rambling into "User:".
  refusal     section 8 - the collapse itself. A model that answers nothing scores well on loss.
  maths       new for v2: GSM8K is in the mix now, so arithmetic is in scope for the first time.

Runs on CPU by default so it can be used while the GPU is busy, and copies the checkpoint before
loading because the trainer rewrites it every 2,000 steps.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

from model.config import v2_config
from model.model import DarkGLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
SCRATCH = Path("tests/behaviour_checkpoint.tmp")
STOP_SEQUENCES = ("\nUser:", "\nContext:", "\nAssistant:")

TESLA_ONE = "A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891."
TESLA_THREE = (
    "The Van de Graaff generator is an electrostatic generator built by Robert Van de Graaff. "
    "A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891. "
    "Penicillin is an antibiotic discovered by Alexander Fleming in 1928.")
FACTS = (
    "A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891. "
    "Penicillin is an antibiotic discovered by Alexander Fleming in 1928. "
    "Rychnowek is a settlement in the administrative district of Gmina Barlinek, Poland.")
SUN = ("The Sun is the star at the centre of the Solar System. Its surface temperature is about "
       "5,500 degrees Celsius, and it is by far the hottest object in the Solar System.")
COCONUT = ("The coconut tree is a member of the palm family. Coconut oil is high in saturated "
           "fat, and many health organizations advise against consuming it in quantity.")
FISH = ("Fish are aquatic animals that breathe using gills. Gills extract dissolved oxygen from "
        "water. Fish do not have lungs, unlike mammals such as whales.")

# (label, question, context, must_contain_any, must_not_contain_any)
OPEN_CASES = [
    ("echo", "What is a Tesla coil?", TESLA_ONE,
     ["resonant", "transformer"], ["a tesla coil is a tesla coil"]),
    ("length", "What is a Tesla coil?", TESLA_THREE,
     ["resonant", "transformer"], ["van de graaff", "penicillin"]),
    ("echo", "Who is William Shakespeare?",
     "William Shakespeare was an English playwright, poet and actor, widely regarded as the "
     "greatest writer in the English language.",
     ["playwright", "poet", "writer"], []),
    ("binding", "Who invented the Tesla coil?", FACTS,
     ["tesla"], ["fleming"]),
    ("binding", "Who discovered penicillin?", FACTS,
     ["fleming"], ["tesla"]),
    ("length", "When was the Tesla coil invented?", TESLA_THREE,
     ["1891"], ["1928"]),
    ("maths", "Natalia sold clips to 48 friends in April, then half as many in May. "
              "How many clips did she sell altogether?", "",
     ["72"], []),
    ("maths", "Weng earns $12 an hour. She babysat for 30 minutes. How much did she earn?", "",
     ["6"], []),
]

# (label, question, context, expected) - scored on P(Yes) vs P(No), no generation needed.
YESNO_CASES = [
    ("polarity", "Is the sun hot?", SUN, "Yes"),
    ("polarity", "Is the sun cold?", SUN, "No"),
    ("polarity", "Do fish have gills?", FISH, "Yes"),
    ("polarity", "Do fish have lungs?", FISH, "No"),
    ("natural", "is coconut oil high in saturated fat", COCONUT, "Yes"),
    ("natural", "is coconut oil low in saturated fat", COCONUT, "No"),
    ("binding", "Was the Tesla coil invented by Nikola Tesla?", FACTS, "Yes"),
    ("binding", "Was the Tesla coil invented by Alexander Fleming?", FACTS, "No"),
    ("natural", "Did Nikola Tesla invent the Tesla coil?", FACTS, "Yes"),
    ("natural", "Did Alexander Fleming invent the Tesla coil?", FACTS, "No"),
]


def load_model(checkpoint_path, device):
    for attempt in (1, 2):
        try:
            shutil.copy2(checkpoint_path, SCRATCH)
            checkpoint = torch.load(SCRATCH, map_location=device, mmap=True, weights_only=True)
            break
        except Exception as error:
            if attempt == 2:
                raise
            print(f"  checkpoint busy ({error}); retrying in 30s")
            time.sleep(30)
    model = DarkGLM(v2_config()).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    step = int(checkpoint.get("step", -1))
    best = checkpoint.get("best_val", float("nan"))
    del checkpoint
    return model, step, best


def build_prompt(question, context):
    if context:
        return f"User: {question}\n\nContext: {context}\nAssistant:"
    return f"User: {question}\nAssistant:"


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=60):
    config = v2_config()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    prompt_length = ids.shape[1]
    stopped_cleanly = False
    for _ in range(max_new_tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)     # greedy: this is a regression test
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == endoftext:
            stopped_cleanly = True
            break
        text = tokenizer.decode(ids[0, prompt_length:].tolist())
        if any(stop in text for stop in STOP_SEQUENCES):
            break
    answer = tokenizer.decode(ids[0, prompt_length:].tolist())
    for stop in STOP_SEQUENCES:
        if stop in answer:
            answer = answer.split(stop)[0]
    return answer.strip(), stopped_cleanly


@torch.no_grad()
def yes_no_probabilities(model, tokenizer, prompt, device):
    yes_id = tokenizer.encode(" Yes").ids[0]
    no_id = tokenizer.encode(" No").ids[0]
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    logits = model(ids)[0, -1]
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities[yes_id].item(), probabilities[no_id].item()


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"missing {path}. Pass --checkpoint checkpoints/pretrain_v2_best.pt to "
                         f"score the pretrained model instead (it will do badly on the "
                         f"instruction format - it has never seen it).")

    model, step, best = load_model(path, args.device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    print(f"checkpoint {path} | step {step:,} | validation {best:.4f}")

    scores = {}

    def record(label, ok):
        hit, total = scores.get(label, (0, 0))
        scores[label] = (hit + (1 if ok else 0), total + 1)

    print("\n=== open-ended (greedy) ===")
    clean_stops = 0
    for label, question, context, wanted, forbidden in OPEN_CASES:
        answer, stopped = generate(model, tokenizer, build_prompt(question, context),
                                   args.device)
        lowered = answer.lower()
        ok = (any(w.lower() in lowered for w in wanted) if wanted else True)
        if forbidden and any(f.lower() in lowered for f in forbidden):
            ok = False
        record(label, ok)
        clean_stops += stopped
        # A blank or near-blank answer is the refusal-collapse signature.
        record("refusal", len(answer.strip()) >= 2)
        print(f"  [{label:8}] {'ok  ' if ok else 'MISS'} {question[:52]:52}")
        print(f"             -> {answer[:110]!r}")

    total_open = len(OPEN_CASES)
    scores["stopping"] = (clean_stops, total_open)

    print("\n=== yes/no (P(Yes) vs P(No)) ===")
    # If BOTH probabilities are negligible the model is not answering a yes/no question at all -
    # it is continuing text and happens to rank one marginally above the other. Scoring that as
    # a hit produces exactly the kind of flattering partial credit section 8 warns about: the
    # pretrained checkpoint scored 2/4 on polarity while its separation gap was 0.0000.
    ANSWERING_THRESHOLD = 0.01
    not_answering = 0
    for label, question, context, want in YESNO_CASES:
        p_yes, p_no = yes_no_probabilities(model, tokenizer,
                                           build_prompt(question, context), args.device)
        got = "Yes" if p_yes > p_no else "No"
        engaged = max(p_yes, p_no) >= ANSWERING_THRESHOLD
        ok = engaged and got == want
        if not engaged:
            not_answering += 1
        record(label, ok)
        flag = "ok  " if ok else ("n/a " if not engaged else "MISS")
        print(f"  [{label:8}] {flag} {question[:46]:46} "
              f"want {want:3} | Yes {p_yes:.4f} No {p_no:.4f}"
              + ("   <- not answering yes/no" if not engaged else ""))
    if not_answering:
        print(f"  ** {not_answering}/{len(YESNO_CASES)} cases: the model is not producing a "
              f"yes/no answer at all (both probabilities < {ANSWERING_THRESHOLD}).")
        print("     Expected for a PRETRAINED checkpoint - it has never seen this format.")

    # The polarity check that a per-question score cannot express: opposite questions on the
    # same passage must produce DIFFERENT probabilities. v1 gave 0.7988 and 0.8003 - identical
    # to three decimals - while scoring 1/2 on the pair, which looks like partial credit rather
    # than the total failure it was.
    print("\n=== polarity separation (the defect a per-question score hides) ===")
    pairs = [("Is the sun hot?", "Is the sun cold?", SUN),
             ("Do fish have gills?", "Do fish have lungs?", FISH),
             ("is coconut oil high in saturated fat", "is coconut oil low in saturated fat",
              COCONUT)]
    separated = 0
    for a, b, context in pairs:
        pa, _ = yes_no_probabilities(model, tokenizer, build_prompt(a, context), args.device)
        pb, _ = yes_no_probabilities(model, tokenizer, build_prompt(b, context), args.device)
        gap = abs(pa - pb)
        ok = gap > 0.10
        separated += ok
        print(f"  {'ok  ' if ok else 'MISS'} P(Yes) {pa:.4f} vs {pb:.4f} | gap {gap:.4f}  "
              f"{a[:34]!r} vs {b[:34]!r}")
    scores["separation"] = (separated, len(pairs))

    print("\n=== summary ===")
    order = ["echo", "polarity", "separation", "binding", "natural", "length", "stopping",
             "refusal", "maths"]
    total_hit = total_all = 0
    for label in order:
        if label not in scores:
            continue
        hit, total = scores[label]
        total_hit += hit
        total_all += total
        bar = "#" * int(hit / max(total, 1) * 20)
        print(f"  {label:11} {hit:>2}/{total:<2} {bar}")
    print(f"  {'OVERALL':11} {total_hit:>2}/{total_all:<2} "
          f"({total_hit/max(total_all,1)*100:.0f}%)")

    SCRATCH.unlink(missing_ok=True)
    print("\nSelect fine-tune checkpoints on THIS, not on validation loss alone (section 8).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
