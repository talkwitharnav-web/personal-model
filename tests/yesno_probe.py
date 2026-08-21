"""Throwaway probe: does the model still know "No", and does context move it?"""
import torch

from orchestrator.generation import Generator, pick_device

SUN = ("The Sun is a star at the centre of the Solar System. Its surface "
       "temperature is about 5,500 degrees Celsius and its core reaches "
       "15 million degrees Celsius.")
FISH = ("Fish breathe using gills, which extract dissolved oxygen from water. "
        "Unlike mammals, most fish do not have lungs.")
COCONUT = ("Coconut is high in saturated fat. Excessive consumption of saturated "
           "fat is associated with raised cholesterol and heart disease risk.")

CASES = [
    ("is the sun cold", SUN, "No"),
    ("is the sun hot", SUN, "Yes"),
    ("do fish have lungs", FISH, "No"),
    ("do fish have gills", FISH, "Yes"),
    ("are coconuts bad for you", COCONUT, "No"),
    ("is earth triangular?", None, "No"),
    ("is the sun cold", None, "No"),
    ("do fish have lungs", None, "No"),
]


def main():
    gen = Generator(device=pick_device())
    yes_id = gen.tokenizer.encode(" Yes").ids[0]
    no_id = gen.tokenizer.encode(" No").ids[0]
    print(f"token ids -> Yes={yes_id}  No={no_id}\n")
    print(f"{'question':28} {'ctx':4} {'want':5} {'P(Yes)':>8} {'P(No)':>8}  verdict")

    for question, context, want in CASES:
        if context:
            prompt = f"User: {question}\n\nContext: {context}\nAssistant:"
        else:
            prompt = f"User: {question}\nAssistant:"
        ids = torch.tensor([gen.tokenizer.encode(prompt).ids], device=gen.device)
        with torch.no_grad():
            logits = gen.model(ids)[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes, p_no = probs[yes_id].item(), probs[no_id].item()
        got = "Yes" if p_yes > p_no else "No"
        mark = "ok " if got == want else "MISS"
        print(f"{question:28} {'yes' if context else '-':4} {want:5} "
              f"{p_yes:8.4f} {p_no:8.4f}  {mark} (said {got})")

    prompt = "User: is the sun cold\nAssistant:"
    with torch.no_grad():
        logits = gen.model(torch.tensor([gen.tokenizer.encode(prompt).ids], device=gen.device))
    top = torch.softmax(logits[0, -1].float(), dim=-1).topk(8)
    print("\ntop-8 first tokens for 'is the sun cold' with no context:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   {p:6.3f}  {gen.tokenizer.decode([i])!r}")


if __name__ == "__main__":
    main()
