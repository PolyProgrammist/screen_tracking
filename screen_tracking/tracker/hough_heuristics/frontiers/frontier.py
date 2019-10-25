class Frontier:
    def __init__(self):
        self.candidates = []

    def top_overall(self, k=None):
        return list(sorted(self.candidates, key=lambda candidate: candidate.overall_score))

    def top_current(self, k=None):
        return list(sorted(self.candidates, key=lambda candidate: candidate.current_score))

    def overall_current(self):
        return min(self.candidates, key=lambda candidate: candidate.overall_score)

    def best_current(self):
        return min(self.candidates, key=lambda candidate: candidate.current_score)

    def show(self):
        pass
