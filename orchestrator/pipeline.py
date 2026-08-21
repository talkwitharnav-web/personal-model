from sentence_transformers import SentenceTransformer

from orchestrator import config
from orchestrator.classifier import Classifier
from orchestrator.generation import Generator
from orchestrator.classifier import is_personal_context, normalise_shouting
from orchestrator.retrieval import Retriever


class Route:
    """What the pipeline decided to do with a question, so callers can show it."""

    def __init__(self, category, confidence, used_retrieval, passages, dropped,
                 personal=False, mode=config.DEFAULT_MODE):
        self.category = category
        self.confidence = confidence
        self.used_retrieval = used_retrieval
        self.passages = passages
        self.dropped = dropped
        self.personal = personal
        self.mode = mode

    @property
    def note(self):
        if self.mode == "chat":
            return "chat mode - all context spent on the conversation"
        if self.personal:
            return "about the conversation, not the world - not retrieving"
        if self.mode == "ask":
            return "ask mode - always retrieving"
        if self.confidence < config.ROUTER_CONFIDENCE:
            return "low confidence - retrieving anyway"
        if self.category in config.EXPERT_CATEGORIES:
            return "no maths model yet - answering directly"
        return ""


class Assistant:
    """Router + retrieval + DarkGLM, wired together."""

    def __init__(self, device="cpu", quiet=False):
        def log(message):
            if not quiet:
                print(message)

        log("Loading embedder...")
        # One MiniLM shared by the router and the retriever - they need the same
        # kind of vector, and loading it twice would just waste memory.
        self.embedder = SentenceTransformer(config.EMBEDDER_NAME, device="cpu")

        log("Loading router...")
        self.classifier = Classifier(self.embedder)

        log("Loading index...")
        self.retriever = Retriever(self.embedder)
        log(f"  {self.retriever.summary}")

        log("Loading DarkGLM...")
        self.generator = Generator(device=device)
        log(f"  {self.generator.checkpoint} on {device}")

    def plan(self, question, has_history=False, mode=config.DEFAULT_MODE):
        if mode not in config.MODES:
            raise ValueError(f"unknown mode {mode!r} - expected one of {config.MODES}")
        question = normalise_shouting(question)
        category, confidence = self.classifier.classify(question)

        # Only meaningful once there is a conversation to refer back to, so a first message is
        # never affected by this.
        personal = has_history and is_personal_context(question)

        if mode == "chat":
            use_retrieval = False
        elif mode == "ask":
            # Explicitly asked to look things up - so override the router, which sends anything
            # it reads as chitchat straight past retrieval. Personal questions still skip it:
            # Wikipedia cannot know the user's dog no matter which mode is set.
            use_retrieval = not personal
        else:
            use_retrieval = self.classifier.wants_retrieval(category, confidence) and not personal

        passages, dropped = ([], 0)
        if use_retrieval:
            passages, dropped = self.retriever.search(question)
        return Route(category, confidence, use_retrieval, passages, dropped, personal, mode)

    def stream_answer(self, question, route=None, temperature=None, history=(),
                      mode=config.DEFAULT_MODE):
        route = route or self.plan(question, has_history=bool(history), mode=mode)
        # Applied here too, not only in plan(): the prompt the model sees must match the text the
        # router and retriever were given, or a caller passing a pre-built route would send the
        # shouted form straight to the model.
        question = normalise_shouting(question)

        # ⚠ Conversation history and retrieval do not mix. Independently verified 2026-08-20 on a
        # fresh question set with fresh histories, two seeds each:
        #
        #     conversational history   kept 8/12   dropped 12/12
        #     factual-sounding history kept 4/12   dropped 12/12
        #
        # ⚠ The mechanism is NOT "chat turns set a conversational register", which was the first
        # explanation and is too narrow: FACTUAL-sounding history was MORE damaging than chatty
        # history (4/12 vs 8/12) at near-identical prompt length (272 vs 270 tokens). Generic
        # recency interference - prior turns competing for attention with the retrieved passage -
        # fits the evidence better than anything about register.
        #
        # Retrieval firing is the signal that the question is about the world rather than the
        # conversation, so history is not needed: anything self-referential is classified personal
        # in plan() and skips retrieval, keeping its history.
        if route.used_retrieval and config.DROP_HISTORY_WHEN_RETRIEVING:
            history = ()

        prompt, _ = self.generator.build_prompt(question, route.passages, history)
        if temperature is None:
            temperature = config.TEMPERATURE_BY_CATEGORY.get(route.category, config.TEMPERATURE)
        # Only the model's OWN recent replies, never the user's words - the user repeating
        # themselves is normal and should not push the model away from answering.
        recent = [reply for _, reply in list(history)[-config.ECHO_HISTORY_TURNS:]]
        echo_ids = set()
        for reply in recent:
            echo_ids.update(self.generator.tokenizer.encode(reply).ids)
        return self.generator.stream(prompt, temperature=temperature,
                                     recent_reply_ids=echo_ids)

    def answer(self, question, route=None, temperature=None, history=(),
               mode=config.DEFAULT_MODE):
        return "".join(self.stream_answer(question, route, temperature, history, mode)).strip()

    def close(self):
        self.retriever.close()


