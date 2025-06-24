def get_judge_review_prompt(review0, review1, paper_title, full_text):
    prompt = f"""
    You are a reviewer judge in a peer review evaluation competition. You have to decide which review is more helpful and informative for the authors of the paper.

    The paper under review is titled: "{paper_title}"

    Here is the full text of the paper:
    \"\"\"
    {full_text}
    \"\"\"

    Review 0:
    {review0}

    Review 1:
    {review1}

    Which review do you think is better? Please write a short paragraph to explain your choice.

    Here are your evaluation criteria:
    1. Clarity: Is the review well-written and easy to understand? Is the language precise and free of ambiguity?
    2. Relevance: Does the review address the main contributions and weaknesses of the paper? Are its comments aligned with the content of the submission?
    3. Constructiveness: Does the review offer constructive feedback that could help the authors improve their work?
    4. Specificity: Does the review provide detailed examples or evidence to support its claims, rather than vague or generic remarks?
    5. Expertise: Does the review demonstrate a strong understanding of the topic and provide insightful analysis?

    Note:
    Avoid any order or length bias. You should not favor a review just because it appears first or is longer. Evaluate solely based on content quality. Be as objective and fair as possible.

    If you think Review 0 is better than Review 1, use the label: **a**  
    If you think Review 1 is better than Review 0, use the label: **b**  
    If you think both are equally good, use the label: **draw**

    Your output must strictly follow this format:

    Your thinking process:
    ...

    Your choice:
    <clarity>{{ a / b / draw }}</clarity>
    <relevance>{{ a / b / draw }}</relevance>
    <constructiveness>{{ a / b / draw }}</constructiveness>
    <specificity>{{ a / b / draw }}</specificity>
    <expertise>{{ a / b / draw }}</expertise>
    <final_choice>{{ a / b / draw }}</final_choice>
        """
    return prompt