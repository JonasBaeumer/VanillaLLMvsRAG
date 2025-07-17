"""
ELO engine for LLM-based review evaluation.

This module implements the EloEngine class, which runs ELO-based tournaments to compare human, LLM-only, and RAG-generated reviews using LLM-based judging.
"""
import random
import logging
import json
import re
from evaluation.llm_judge.judge_review_prompt import get_judge_review_prompt
from models.openai_models import OpenAILLM

logger = logging.getLogger(__name__)

class EloEngine:
    """
    EloEngine runs ELO-based tournaments to compare review quality between human, LLM-only, and RAG-generated reviews.
    It uses an LLM to judge review pairs and updates ratings accordingly.
    """
    def __init__(self, initial_rating=1000, k=32, allow_draws=True, normalize=False):
        """
        Initialize the EloEngine with rating parameters and LLM judge.

        Args:
            initial_rating (int): Starting ELO rating for each system.
            k (int): K-factor for ELO updates.
            allow_draws (bool): Whether draws are allowed in judging.
            normalize (bool): Whether to normalize ratings after tournament.
        """
        self.k = k
        self.ratings = {
            "human_review": initial_rating,
            "llm_only": initial_rating,
            "rag_pipeline": initial_rating,
        }
        self.history = []
        self.allow_draws = allow_draws
        self.normalize = normalize
        self.llm = OpenAILLM()

    def expected_score(self, rating_a, rating_b):
        """
        Compute the expected score for player A against player B using ELO formula.
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a, player_b, result):
        """
        Update ELO ratings for two players based on the result of a match.

        Args:
            player_a (str): Name of player A.
            player_b (str): Name of player B.
            result (str): 'draw', player_a, or player_b.
        """
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

    def log_round(self, round_number, contestant_a, contestant_b, result, judge_info):
        """
        Log the results and feedback of a single tournament round.

        Args:
            round_number (int): The round number.
            contestant_a (str): Name of contestant A.
            contestant_b (str): Name of contestant B.
            result (str): Winner of the round.
            judge_info (dict): Judging feedback and parsed tags.
        """
        round_record = {
            "round": round_number,
            "contestants": [contestant_a, contestant_b],
            "winner": result,
            "ratings": self.ratings.copy(),
            "judging_feedback": judge_info
        }
        self.history.append(round_record)

    def judge(self, review_a, review_b, paper_title, paper_abstract, paper_text):
        """
        Use the LLM to judge between two reviews for a given paper.

        Args:
            review_a (dict): First review.
            review_b (dict): Second review.
            paper_title (str): Title of the paper.
            paper_abstract (str): Abstract of the paper.
            paper_text (str): Full text of the paper.

        Returns:
            tuple: (winner label, parsed feedback dict)
        """
        prompt = get_judge_review_prompt(
            review0=review_a,
            review1=review_b,
            paper_title=paper_title,
            full_text=paper_text
        )

        response = self.llm.generate_text([
            {"role": "system", "content": "You are scoring peer reviews for scientific papers."},
            {"role": "user", "content": prompt}
        ])

        # Extract the reasoning
        thinking_match = re.search(r"Your thinking process:\s*(.*?)\s*Your choice:", response, re.DOTALL)
        thinking_process = thinking_match.group(1).strip() if thinking_match else ""

        # Parse all tags
        tags = ["clarity", "relevance", "constructiveness", "specificity", "expertise", "final_choice"]
        parsed = {}
        counts = {"a": 0, "b": 0, "draw": 0}
        for tag in tags:
            match = re.search(fr"<{tag}>\s*{{\s*(a|b)\s*}}\s*</{tag}>", response, re.IGNORECASE)
            if match:
                choice = match.group(1).lower()
                parsed[tag] = choice
                if tag != "final_choice":
                    counts[choice] += 1
            else:
                logger.warning(f"⚠️ Could not parse tag <{tag}> correctly. Setting to 'invalid'. Full response:\n{response}")
                parsed[tag] = "invalid"

        winner = parsed.get("final_choice")
        return winner, {
            **parsed,
            "thinking_process": thinking_process
        }

    def run_tournament(self, dataset, num_rounds=10):
        """
        Run an ELO tournament on the dataset for a specified number of rounds.

        Args:
            dataset (list): List of paper entries with reviews.
            num_rounds (int): Number of rounds to run.

        Returns:
            dict: ELO scores and round history.
        """
        for i in range(num_rounds):
            entry = random.choice(dataset)

            # Prepare contestants
            contestants = {
                "human_review": self._sample_human_review(entry, entry["reviews"]),
                "llm_only": entry["llm_generated_review"],
                "rag_pipeline": entry["llm_plus_rag_generated_review"],
            }
            title = entry["metadata"].get("title", "[no title]")
            abstract = entry["metadata"].get("abstract", "")
            full_text = entry["tei_data"].get("full_text", "")

            a, b = random.sample(list(contestants.keys()), 2)
            review_a = contestants[a]
            review_b = contestants[b]

            winner, judge_info = self.judge(review_a, review_b, title, abstract, full_text)
            result_label = "draw" if winner == "draw" else a if winner == "a" else b

            self.update_ratings(a, b, result_label)
            self.log_round(i + 1, a, b, result_label, judge_info)

        if self.normalize:
            self._normalize_ratings()

        return {
            "elo_scores": self.ratings,
            "elo_rounds": self.history,
        }

    def _sample_human_review(self, full_entry, entry):
        """
        Randomly sample a human review from the entry for comparison.

        Args:
            full_entry (dict): The full paper entry.
            entry (list): List of human reviews.

        Returns:
            dict: Selected review fields for judging.
        """
        if not entry:
            logger.warning(f"⚠️ No human reviews available for this entry {full_entry['paper_id']}. Skipping.")
            return "[NO HUMAN REVIEW AVAILABLE]"

        review = random.choice(entry)
        selected_review = {
            "paper_summary": review["paper_summary"],
            "summary_of_strengths": review["summary_of_strengths"],
            "summary_of_weaknesses":review["summary_of_weaknesses"],
            "comments_suggestions_and_typos": review["comments_suggestions_and_typos"],
            "scores": {
                "soundness": str(review["scores"].get("soundness", "")),
                "overall_assessment": str(review["scores"].get("overall_assessment", ""))
            }
        }
        return selected_review

    def _normalize_ratings(self):
        """
        Normalize ELO ratings so that the mean is 1000 and values are scaled to [0, 100].
        """
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


# if __name__ == "__main__":
#     # LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
#     import logging
#     import json
#     logging.basicConfig(level=logging.INFO)
#     dummy_dataset = [
#         {
#             "reviews": [
#                 {
#                     "topic_and_contributions": "Discussion of RL in robotics.",
#                     "reasons_to_accept": "Well-structured with strong empirical results.",
#                     "reasons_to_reject": "Limited theory.",
#                     "typos_and_style": "Local comments: -- What was claimed exactly in previous work with respect to the discussed bias and how does that differ from the work here?\n-- l. 136: you write \"Our work is the 136 first to investigate the effects of finetuning on the 137 correlation between term frequency statistics and 138 factual knowledge of LLMs.\" This was not sufficiently clear to me:  I thought your main claim is about the relation between these stats in pretraining and in model behavior. This should be made clearer. Otherwise, a well written previous work section.\n-- Section 4.2: Why not use the more standard names then like marginal probability, joint probability and PMI?  -- Figures 2b, 3: The graphs do not show much in my opinion. Could they be explained in a sentence in the text? what does the figure here contribute?\n-- Figure 5 (and in other places in the paper): can you also compute correlation w/o binning? what does that turn out to be?\n-- Section 6.1: a more formal definition of the filtering method should be given.\nGrammar: -- l. 442: performances s.b. performance -- l. 501: changes s.b. change ",
#                     "scores": {
#                         "soundness": "3: Good: This study provides sufficient support for its major claims/arguments, some minor points may need extra support or details.",
#                         "overall_assessment": "3: Ambivalent: It has merits (e.g., it reports state-of-the-art results, the idea is nice), but there are key weaknesses (e.g., it describes incremental work), and it can significantly benefit from another round of revision. However, I won't object to accepting it if my co-reviewers champion it."
#                     }
#                 }
#             ],
#             "llm_generated_review": "This paper has solid empirical results but lacks theoretical rigor.",
#             "llm_plus_rag_generated_review": "Strong in experiments, outlines valuable contributions.",
#             "tei_data": {
#                 "title": "Reinforcement Learning for Robotics",
#                 "full_text": (
#                     "This paper investigates the application of reinforcement learning (RL) techniques in the context of robotics. "
#                     "The authors implement three variants of policy gradient methods across two common robotic environments: robotic arm manipulation "
#                     "and bipedal locomotion. A detailed comparison is made between standard PPO, SAC, and a custom modified actor-critic approach. "
#                     "Experimental results demonstrate that while SAC achieves faster convergence in low-dimensional control tasks, the modified actor-critic "
#                     "outperforms others in high-dimensional settings. The paper discusses reward shaping, exploration strategies, and the challenges of sim-to-real "
#                     "transfer. Although the empirical evaluation is solid, the paper lacks a strong theoretical justification for the proposed modifications, "
#                     "and no formal convergence proofs are provided. The authors conclude by suggesting future work on curriculum learning and multi-agent coordination. "
#                     "Overall, the paper provides valuable empirical insights but would benefit from stronger theoretical framing and broader ablation studies."
#                 )
#             },
#             "metadata": {
#                 "abstract": "This paper explores the use of reinforcement learning in robotic systems."
#             }
#         }
#         for _ in range(10)
#     ]
#     engine = EloEngine()
#     results = engine.run_tournament(dummy_dataset, num_rounds=5)
#     print("\n🏁 Final ELO Scores:")
#     print(json.dumps(results["elo_scores"], indent=2))

    print("\n📜 Round-by-Round History:")
    for round_info in results["elo_rounds"]:
        print(round_info)