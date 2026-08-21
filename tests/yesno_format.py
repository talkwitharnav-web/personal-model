"""Does the contrastive yes/no skill exist at the TRAINED phrasing but fail to transfer?

The yesno dataset only ever says "Was X <verb> by <name>?" / "Was X <verb> in <year>?" /
"Is X <description>?". The probe that looked weak used "is the sun cold" - a shape the
model was never trained on. Same lesson as definitions v1: a skill can be real and still
only exist at the width it was taught.
"""
import torch

from orchestrator.generation import Generator, pick_device

TESLA = ("A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891. "
         "Penicillin is an antibiotic discovered by Alexander Fleming in 1928. "
         "Rychnowek is a settlement in the administrative district of Gmina Barlinek, Poland.")
COCONUT = ("The coconut tree is a member of the palm tree family. Coconut is high in saturated "
           "fat. Many health organizations advise against consumption of coconut oil owing to "
           "its high levels of saturated fat.")

# (question, context, expected) - trained phrasing first, then the natural equivalent.
TRAINED = [
    ("Was the Tesla coil invented by Nikola Tesla?", TESLA, "Yes"),
    ("Was the Tesla coil invented by Alexander Fleming?", TESLA, "No"),
    ("Was the Tesla coil invented in 1891?", TESLA, "Yes"),
    ("Was the Tesla coil invented in 1928?", TESLA, "No"),
    ("Was penicillin discovered by Alexander Fleming?", TESLA, "Yes"),
    ("Was penicillin discovered by Nikola Tesla?", TESLA, "No"),
    ("Is Rychnowek a settlement in the administrative district of Gmina Barlinek?", TESLA, "Yes"),
    ("Is Rychnowek an antibiotic?", TESLA, "No"),
]

NATURAL = [
    ("Did Nikola Tesla invent the Tesla coil?", TESLA, "Yes"),
    ("Did Alexander Fleming invent the Tesla coil?", TESLA, "No"),
    ("is coconut high in saturated fat", COCONUT, "Yes"),
    ("is coconut low in saturated fat", COCONUT, "No"),
]


def run(generator, cases, title):
    yes_id = generator.tokenizer.encode(" Yes").ids[0]
    no_id = generator.tokenizer.encode(" No").ids[0]
    print(f"\n{title}")
    print(f"{'question':62} {'want':5} {'P(Yes)':>8} {'P(No)':>8}  verdict")
    hits = 0
    for question, context, want in cases:
        prompt = f"User: {question}\n\nContext: {context}\nAssistant:"
        ids = torch.tensor([generator.tokenizer.encode(prompt).ids], device=generator.device)
        with torch.no_grad():
            logits = generator.model(ids)[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes, p_no = probs[yes_id].item(), probs[no_id].item()
        got = "Yes" if p_yes > p_no else "No"
        hits += got == want
        print(f"{question[:62]:62} {want:5} {p_yes:8.4f} {p_no:8.4f}  "
              f"{'ok ' if got == want else 'MISS'} (said {got})")
    print(f"  -> {hits}/{len(cases)}")
    return hits, len(cases)


def main():
    generator = Generator(device=pick_device())
    print(f"checkpoint: {generator.checkpoint}")
    a = run(generator, TRAINED, "TRAINED PHRASING (matches data/yesno/train.txt)")
    b = run(generator, NATURAL, "NATURAL PHRASING (never seen in this form)")
    print(f"\ntrained {a[0]}/{a[1]}   natural {b[0]}/{b[1]}")


if __name__ == "__main__":
    main()
