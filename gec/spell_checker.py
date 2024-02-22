import jamspell

class SpellChecker:

    def __init__(self, checker_type: str = "jamspell"):
        """Load Spell Checker"""
        self.checker_type = checker_type
        self.jamspell = jamspell.TSpellCorrector()

    def load(self, **kwargs):
        if self.checker_type == "jamspell":
            return self.load_jamspell(**kwargs)
        raise Exception(f"{self.checker_type} is not supported")

    def load_jamspell(self, load_path: str = 'stats/uk_jamspell_small.bin'):
        self.jamspell.LoadLangModel(load_path)
        return self.jamspell

    def suggest(self, token: str, tokens: list = [], token_pos: int = 0):
        if not self.jamspell:
            raise Exception(
                "JamSpell is not loaded. Please load it with load()")

        if not tokens:
            return self.jamspell.FixFragment(token)

        suggestions = self.jamspell.GetCandidates(tokens, token_pos)
        if not suggestions:
            return token
        suggestion: str = suggestions[0]
        if token[0].isupper():
            return suggestion.capitalize()

        return suggestion
