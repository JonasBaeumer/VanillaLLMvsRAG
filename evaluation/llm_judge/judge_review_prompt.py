def get_judge_review_prompt(review0, review1, paper_title, full_text):
    prompt = f"""
You are a reviewer judge in a peer review evaluation competition. Your task is to decide which of two reviews is more helpful and informative for the authors of the paper.

The paper under review is titled: "{paper_title}"

Here is the full text of the paper:
\"\"\"
{full_text}
\"\"\"

Review 0:
{review0}

Review 1:
{review1}

Please evaluate the two reviews using the following **five criteria**:

1. **Clarity** – Is the review well-written and easy to understand? Is the language precise and free of ambiguity?
2. **Relevance** – Does the review address the main contributions and weaknesses of the paper? Are its comments aligned with the content of the submission?
3. **Constructiveness** – Does the review offer constructive feedback that could help the authors improve their work?
4. **Specificity** – Does the review provide detailed examples or evidence to support its claims?
5. **Expertise** – Does the review demonstrate a strong understanding of the topic and provide insightful analysis?

---
Your task: Evaluate the two reviews based on the 5 criteria and choose a winner.

**Rules**:
- For each category, select the better review: `a` for Review 0 or `b` for Review 1. You must always a winner even if the difference is small.
- For the **<final_choice>**, select the review with the majority vote across the 5 criteria.
- In case of a tie (e.g., 2 for a, 2 for b, 1 unclear), resolve it by choosing the review that provides more **actionable and specific feedback**.
- ❗ You are **not allowed** to write "draw" in any field. Doing so violates the required output format.
- Be objective, fair, and precise in your reasoning.

Output format (strictly follow this format – no deviations):

---BEGIN OUTPUT---

Your thinking process:
<explanation for each criterion and why one review is better>

Your choice:
<clarity>{{ a / b }}</clarity>  
<relevance>{{ a / b }}</relevance>  
<constructiveness>{{ a / b }}</constructiveness>  
<specificity>{{ a / b }}</specificity>  
<expertise>{{ a / b }}</expertise>  
<final_choice>{{ a / b }}</final_choice>

---END OUTPUT---
    """
    return prompt