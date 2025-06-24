import random
import logging

logger = logging.getLogger(__name__)

class EloEngine:
    def __init__(self, initial_rating=1000, k=32, allow_draws=True, normalize=False):
        self.k = k
        self.ratings = {
            "human_review": initial_rating,
            "llm_only": initial_rating,
            "rac_pipeline": initial_rating,
        }
        self.history = []
        self.allow_draws = allow_draws
        self.normalize = normalize

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a, player_b, result):
        r_a, r_b = self.ratings[player_a], self.ratings[player_b]
        expected_a = self.expected_score(r_a, r_b)
        expected_b = self.expected_score(r_b, r_a)

        if result == "draw":
            score_a, score_b = 0.5, 0.5
        elif result == player_a:
            score_a, score_b = 1, 0
        else:
            score_a, score_b = 0, 1

        self.ratings[player_a] += self.k * (score_a - expected_a)
        self.ratings[player_b] += self.k * (score_b - expected_b)

    def log_round(self, round_number, a, b, result):
        self.history.append({
            "round": round_number,
            "contestants": [a, b],
            "winner": result,
            "ratings": {k: round(v, 2) for k, v in self.ratings.items()}
        })

    def judge(self, review_a, review_b):
        # Dummy judge – 50% draw, 25% win each
        r = random.random()
        if r < 0.25:
            return "a"
        elif r < 0.5:
            return "b"
        else:
            return "draw"

    def run_tournament(self, dataset, num_rounds=10):
        for i in range(num_rounds):
            entry = random.choice(dataset)

            # Prepare contestants
            contestants = {
                "human_review": self._sample_human_review(entry),
                "llm_only": entry["llm_generated_review"],
                "rac_pipeline": entry["llm_plus_rag_generated_review"],
            }

            a, b = random.sample(list(contestants.keys()), 2)
            review_a = contestants[a]
            review_b = contestants[b]

            result = self.judge(review_a, review_b)
            result_label = "draw" if result == "draw" else a if result == "a" else b

            self.update_ratings(a, b, result_label)
            self.log_round(i + 1, a, b, result_label)

        if self.normalize:
            self._normalize_ratings()

        return {
            "elo_scores": self.ratings,
            "elo_rounds": self.history,
        }

    def _sample_human_review(self, entry):
        review = random.choice(entry["reviews"])
        return review["topic_and_contributions"] + " " + review["reasons_to_accept"] + " " + review["reasons_to_reject"]

    def _normalize_ratings(self):
        mean_rating = sum(self.ratings.values()) / len(self.ratings)
        for key in self.ratings:
            self.ratings[key] = round(self.ratings[key] - mean_rating + 1000, 2)
        values = list(self.ratings.values())
        min_rating, max_rating = min(values), max(values)
        if max_rating == min_rating:
            return  # Avoid division by zero

        for key in self.ratings:
            norm = (self.ratings[key] - min_rating) / (max_rating - min_rating)
            self.ratings[key] = round(norm * 100, 2)


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # Dummy data following your structure
    dummy_dataset = [
        {
            "reviews": [
                {
                    "topic_and_contributions": "A well-structured discussion of reinforcement learning in robotics.",
                    "reasons_to_accept": "Strong empirical results.",
                    "reasons_to_reject": "Limited theoretical grounding.",
                }
            ],
            "llm_generated_review": "This paper performs well empirically but lacks theory.",
            "llm_plus_rag_generated_review": "The study is strong in experiments and outlines a good contribution.",
        }
        for _ in range(10)
    ]

    engine = EloEngine()
    results = engine.run_tournament(dummy_dataset, num_rounds=5)

    print("\n🏁 Final ELO Scores:")
    print(json.dumps(results["elo_scores"], indent=2))

    print("\n📜 Round-by-Round History:")
    for round_info in results["elo_rounds"]:
        print(round_info)