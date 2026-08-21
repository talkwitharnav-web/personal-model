"""Extensive peek: the SAME questions asked with and without a retrieved passage.

DarkGLM is served behind RAG, so the question that matters is not "what does the model know?" but
"what does the model do with a passage when it is given one, versus when it is not?". Those are
different capabilities and they fail differently:

  no context   -> tests parametric recall. An 80M model trained on 1.6B tokens knows little, and
                  section 9's finding was that what it does know it learned as STRINGS, correct
                  only at Wikipedia's exact opening phrasing.
  with context -> tests reading comprehension, which is what the fine-tune mix mostly teaches
                  (75% of yes/no and definitions rows, all of SQuAD, SciQ, BoolQ, RACE).

Showing them side by side is the only way to tell "the model is dumb" apart from "the model was
asked to do the one thing it is not for". Both are printed for every question, with the passage a
retriever would plausibly return.

Generation matches serving: repetition penalty 1.3 (section 8 - it removes looping but not
ignorance, so it is a presentation fix, not a knowledge fix) and greedy plus sampled decoding,
because a single sample cannot tell "got worse" from "rolled badly".
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import DarkGLM
from training.finetune_v2 import v2_config

REPETITION_PENALTY = 1.3

# (group, question, retrieved passage). The passage is what a retriever would hand back - real
# encyclopedic prose, not a tidy restatement of the answer.
CASES = [
    ("factual recall",
     "What is the capital of France?",
     "Paris is the capital and most populous city of France. With an official estimated "
     "population of 2,102,650 residents, Paris is the fourth-most populated city in the "
     "European Union. The City of Paris is the centre of the Ile-de-France region."),

    ("factual recall",
     "Who invented the Tesla coil?",
     "The Tesla coil is an electrical resonant transformer circuit designed by inventor "
     "Nikola Tesla in 1891. It is used to produce high-voltage, low-current, high-frequency "
     "alternating-current electricity."),

    ("factual recall",
     "When did the Second World War begin?",
     "The Second World War began on 1 September 1939 with the German invasion of Poland. "
     "Britain and France declared war on Germany two days later."),

    ("definition",
     "What is photosynthesis?",
     "Photosynthesis is the process used by plants, algae and some bacteria to convert light "
     "energy into chemical energy. Through photosynthesis, these organisms use carbon dioxide "
     "and water to build sugars, releasing oxygen as a by-product."),

    ("definition",
     "What is a mitochondrion?",
     "A mitochondrion is an organelle found in the cells of most eukaryotes. Mitochondria "
     "generate most of the cell's supply of adenosine triphosphate, which is used as a source "
     "of chemical energy. They are often called the powerhouse of the cell."),

    ("definition",
     "What is an earthquake?",
     "An earthquake is the shaking of the surface of the Earth resulting from a sudden release "
     "of energy in the lithosphere that creates seismic waves. Earthquakes are measured using "
     "instruments called seismometers."),

    ("yes/no",
     "Is the sun hot?",
     "The Sun is the star at the centre of the Solar System. Its surface temperature is about "
     "5,500 degrees Celsius and its core reaches around 15 million degrees Celsius."),

    ("yes/no",
     "Do fish have lungs?",
     "Fish are aquatic vertebrates that extract oxygen from water using gills. Unlike mammals, "
     "fish do not have lungs, although lungfish are a notable exception."),

    ("yes/no",
     "Was penicillin discovered by Alexander Fleming?",
     "Penicillin was discovered in 1928 by the Scottish bacteriologist Alexander Fleming at "
     "St Mary's Hospital in London. He noticed that a mould had killed the bacteria around it."),

    ("everyday knowledge",
     "How many legs does a cat have?",
     "The cat is a small domesticated carnivorous mammal. Cats are quadrupeds, walking on four "
     "legs, and have retractable claws on each paw."),

    ("everyday knowledge",
     "What colour is the sky on a clear day?",
     "On a clear day the sky appears blue because molecules in the air scatter blue light from "
     "the Sun more than they scatter red light. This is called Rayleigh scattering."),

    ("maths",
     "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
     "in May. How many clips did Natalia sell altogether in April and May?",
     "To find a total across two periods, compute each period then add them. Half of a quantity "
     "is that quantity divided by two."),

    ("instruction",
     "Write one sentence about a dog.",
     "The domestic dog is a domesticated descendant of the wolf. Dogs are widely kept as pets "
     "and working animals, valued for loyalty and companionship."),

    ("instruction",
     "List three colours.",
     "Colour is the visual perception based on the wavelength of light. Common colours include "
     "red, blue, green, yellow, orange and purple."),

    ("chat",
     "What is your favourite colour?",
     "Assistants do not have personal preferences, but they can discuss colours and explain "
     "why people find particular colours appealing."),
]


def load(checkpoint_path, device):
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    model = DarkGLM(v2_config())
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, tokenizer, blob.get("step", -1), blob.get("best_val", float("nan"))


@torch.no_grad()
def generate(model, tokenizer, prompt, device, tokens=48, greedy=True, temperature=0.8,
             top_p=0.9, seed=0):
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt).ids[-700:]
    # Generator stays on CPU and sampling is done on CPU tensors: torch.multinomial rejects a
    # CPU generator paired with a CUDA tensor, and a CPU generator is what makes `seed` give the
    # same sequence on both devices - which matters, because these probes are compared across
    # runs that may not all have had the GPU free.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    produced = []
    for _ in range(tokens):
        logits = model(torch.tensor([ids], device=device))[0, -1].float()
        for token in set(produced):                       # serving-style repetition penalty
            logits[token] /= REPETITION_PENALTY
        if greedy:
            nxt = int(torch.argmax(logits))
        else:
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            ordered, index = torch.sort(probs, descending=True)
            keep = torch.cumsum(ordered, dim=-1) <= top_p
            keep[0] = True
            ordered, index = ordered[keep], index[keep]
            nxt = int(index[torch.multinomial(ordered / ordered.sum(), 1,
                                              generator=generator)])
        if nxt == endoftext:
            break
        produced.append(nxt)
        ids.append(nxt)
    return tokenizer.decode(produced).strip()


def show(label, text, width=96):
    body = textwrap.fill(text if text else "(empty)", width=width,
                         initial_indent="", subsequent_indent=" " * 16)
    print(f"      {label:<9} {body}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()

    model, tokenizer, step, best = load(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint}  (step {step:,}, validation {best:.4f})")
    print(f"repetition penalty {REPETITION_PENALTY} (serving default), {args.tokens} tokens max\n")

    current = None
    for group, question, passage in CASES:
        if group != current:
            current = group
            print(f"\n{'=' * 104}\n{group.upper()}\n{'=' * 104}")
        print(f"\n  Q: {question if len(question) < 92 else question[:89] + '...'}")

        bare = f"User: {question}\nAssistant:"
        with_ctx = f"User: {question}\n\nContext: {passage}\nAssistant:"

        print("    --- NO CONTEXT (parametric recall) ---")
        show("greedy", generate(model, tokenizer, bare, args.device, args.tokens, greedy=True))
        show("sampled", generate(model, tokenizer, bare, args.device, args.tokens,
                                 greedy=False, seed=1))

        print("    --- WITH CONTEXT (what RAG supplies) ---")
        show("greedy", generate(model, tokenizer, with_ctx, args.device, args.tokens, greedy=True))
        show("sampled", generate(model, tokenizer, with_ctx, args.device, args.tokens,
                                 greedy=False, seed=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
