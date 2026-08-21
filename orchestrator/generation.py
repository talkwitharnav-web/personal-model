from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import DarkGLMConfig, v2_config
from model.model import DarkGLM
from orchestrator import config

# Longest stop marker; we hold back this much text so a partially generated
# "\nUser:" is never shown to the user before we know it's a stop.
_LONGEST_STOP = max(len(stop) for stop in config.STOP_SEQUENCES)


def pick_device(prefer_cpu=False):
    if prefer_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def load_checkpoint(path, device):
    """Return (state_dict, is_v2), handling both checkpoint shapes in this repo.

    v1 phases saved a bare state_dict; the v2 scripts save a training dict with model, optimizer
    and scaler under keys. Detecting v2 from the WEIGHTS rather than the filename means a renamed
    or copied checkpoint still loads correctly.

    The signal is structural: v2 uses RoPE and therefore never creates position_embedding, so
    that key is absent from a v2 state_dict and present in every v1 one.
    """
    raw = torch.load(path, map_location=device, weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    is_v2 = not any(key.startswith("position_embedding.") for key in state)
    return state, is_v2


class Generator:
    """Wraps DarkGLM and produces answers a piece at a time."""

    def __init__(self, device="cpu"):
        checkpoint = next(
            (c for c in config.CHECKPOINT_CANDIDATES if Path(c).exists()), None
        )
        if checkpoint is None:
            raise SystemExit(f"No checkpoint found - looked for {config.CHECKPOINT_CANDIDATES}")

        self.device = device
        self.checkpoint = checkpoint

        state, is_v2 = load_checkpoint(checkpoint, device)
        self.is_v2 = is_v2
        self.config = v2_config() if is_v2 else DarkGLMConfig()
        self.model = DarkGLM(self.config).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        del state

        tokenizer_path = config.TOKENIZER_PATH_V2 if is_v2 else config.TOKENIZER_PATH
        if not Path(tokenizer_path).exists():
            raise SystemExit(
                f"{checkpoint} needs {tokenizer_path}, which is missing. A checkpoint and its "
                f"tokenizer are a matched pair - serving one with the other's tokenizer produces "
                f"fluent nonsense.")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer_path = tokenizer_path
        self.endoftext_id = self.tokenizer.token_to_id("<|endoftext|>")

    def build_prompt(self, question, passages, history=()):
        """Assemble the prompt, newest content first in priority order.

        `history` is a list of (user, assistant) pairs from earlier in the conversation. Without
        it every message is turn 1 of a brand new conversation, so the model cannot resolve "he",
        "it" or "that" and has nothing to stay consistent with - which is exactly what a bare
        greeting exposed ("meet my dog, tommy" -> "Thanks, I'm glad you're enjoying your day").
        The model itself is capable of this: measured on 2026-08-19 it tracked "he" back to a dog
        named Rex across three turns when the turns were actually supplied.

        Budget order matters. The current question is non-negotiable, retrieved passages come
        next because they carry the answer, and history gets whatever is left with the OLDEST
        turns dropped first. Multi-turn examples in SODA and PersonaChat use exactly this
        "User: ...\\nAssistant: ..." layout, so the format is what the model trained on.
        """
        head = f"User: {question}"
        tail = "\nAssistant:"
        header = f"{head}\n\nContext: " if passages else head
        budget = (self.config.context_length
                  - config.ANSWER_TOKEN_RESERVE
                  - len(self.tokenizer.encode(header + tail).ids))

        chosen = []
        for _, passage in passages:
            cost = len(self.tokenizer.encode(passage + " ").ids)
            if cost > budget:
                continue
            chosen.append(passage)
            budget -= cost

        # Whatever budget survives goes to history, most recent first.
        prior = []
        for user_turn, assistant_turn in reversed(list(history)):
            block = f"User: {user_turn}\nAssistant: {assistant_turn}\n"
            cost = len(self.tokenizer.encode(block).ids)
            if cost > budget:
                break
            prior.insert(0, block)
            budget -= cost

        preamble = "".join(prior)
        if not chosen:
            return f"{preamble}User: {question}\nAssistant:", 0
        return (f"{preamble}User: {question}\n\nContext: {' '.join(chosen)}\nAssistant:",
                len(chosen))

    @torch.no_grad()
    def stream(self, prompt, temperature=None, recent_reply_ids=()):
        """Yields the answer in pieces as it is produced.

        `recent_reply_ids` are the tokens of the model's own previous answers. They need a
        penalty of their own because of how conversation history interacts with the existing
        one: the loop below deliberately skips prompt tokens, which is right for retrieved
        passages (reusing their words is the point) but wrong once past answers are in the
        prompt too. Measured effect - the model repeated "Oh, I'm Sarah. Nice to meet you too!"
        for nine consecutive turns, then switched to "Asdf" and repeated that, because every
        repeat was appended to history and made the next repeat more likely. A self-reinforcing
        attractor that could not exist before history was added.
        """
        temperature = temperature or config.TEMPERATURE
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt).ids], dtype=torch.long, device=self.device
        )
        prompt_length = input_ids.shape[1]
        # Milder than the within-answer penalty: these words are legitimately reusable across
        # turns ("I", "the", "you"), so the aim is to break verbatim loops, not to forbid
        # ordinary language.
        echo_ids = set(recent_reply_ids)

        text = ""
        emitted = 0

        for _ in range(config.MAX_NEW_TOKENS):
            crop = input_ids[:, -self.config.context_length:]
            logits = self.model(crop)[:, -1, :] / temperature

            # Only what the model itself produced - penalising the prompt would
            # punish it for reusing words from the retrieved passage.
            for token_id in set(input_ids[0, prompt_length:].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= config.REPETITION_PENALTY
                else:
                    logits[0, token_id] *= config.REPETITION_PENALTY

            for token_id in echo_ids:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= config.ECHO_PENALTY
                else:
                    logits[0, token_id] *= config.ECHO_PENALTY

            top_values, top_indices = torch.topk(logits, config.TOP_P_K)
            probabilities = torch.softmax(top_values, dim=-1)
            next_token = top_indices.gather(-1, torch.multinomial(probabilities, num_samples=1))

            if next_token.item() == self.endoftext_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            text = self.tokenizer.decode(input_ids[0, prompt_length:].tolist())

            stop_at = min(
                (text.find(stop) for stop in config.STOP_SEQUENCES if stop in text),
                default=None,
            )
            if stop_at is not None:
                if stop_at > emitted:
                    yield text[emitted:stop_at]
                return

            # Hold back the tail in case it turns into a stop marker next step.
            safe = len(text) - (_LONGEST_STOP - 1)
            if safe > emitted:
                yield text[emitted:safe]
                emitted = safe

        if len(text) > emitted:
            yield text[emitted:]
