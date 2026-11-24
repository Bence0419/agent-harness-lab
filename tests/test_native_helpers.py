from agent_harness_lab.metrics.patch import PatchStatError


class FakeNumstat:
    """Fake git probe: returns canned numstat text, or raises the given error."""

    def __init__(self, text="", error: PatchStatError | None = None):
        self.text, self.error, self.calls = text, error, []

    def numstat(self, patch):
        self.calls.append(patch)
        if self.error:
            raise self.error
        return self.text
