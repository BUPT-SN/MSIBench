from __future__ import annotations

from typing import Dict, List

PromptGrid = List[List[str]]


def _grid(text: str) -> PromptGrid:
    return [list(line) for line in text.splitlines()]


PROMPTS_NEWS: Dict[str, PromptGrid] = {
    "neutral_v1": _grid(
        """
Write a straight news brief.

Use ONLY these source ideas:
{semantic_ref_text}

Rules:
- Objective, wire-style tone.
- No copied sentences or line-by-line paraphrase.
- Keep length similar to the source.
- Avoid opinionated adjectives and rhetorical questions.
""".strip("\n")
    ),
    "neutral_v2": _grid(
        """
Produce one English news story.

Content source (only):
{semantic_ref_text}

Structure:
- Lede: one or two sentences summarizing the main event.
- Body: 3–5 sentences with key details and context.
- Closing: 1–2 sentences on next steps or impact.

Constraints:
- No copying.
- Neutral, concise newsroom style.
- Similar length to the source.
""".strip("\n")
    ),
    "neutral_v3": _grid(
        """
Draft a neutral news article.

Input facts (only):
{semantic_ref_text}

Style constraints:
- Plain vocabulary, avoid hype.
- Simple transitions (However, Additionally, In contrast).
- No slang or dramatic flourishes.
- Do not mirror the original sentence order.
- Do not copy any sentence; match the length roughly.
""".strip("\n")
    ),
    "neutral_v4": _grid(
        """
Task: Write a concise news report in a bland, professional voice.

Source of information (only):
{semantic_ref_text}

Process (internal):
1) Note the key facts and quotes.
2) Write the article from those notes using neutral language.

Constraints:
- No copied sentences.
- No line-by-line paraphrasing.
- Keep length close to the source.
""".strip("\n")
    ),
    "neutral_v5": _grid(
        """
Compose a newsroom-style article using ONLY:
{semantic_ref_text}

Make it deliberately neutral:
- Avoid rhetorical questions, asides, or dramatic punctuation.
- Keep sentences balanced and direct.
- Do not add new facts.
- Do not copy any sentence; maintain similar length.
""".strip("\n")
    ),
    "neutral_v6": _grid(
        """
Rewrite the following news piece so the facts stay the same but the personal voice disappears. Do not copy sentences. Keep the length similar:

{semantic_ref_text}
""".strip("\n")
    ),
    "neutral_v7": _grid(
        """
You are rewriting a news article.

Original (keep meaning, change wording):
{semantic_ref_text}

Checklist before writing:
[ ] Same key facts and stance
[ ] Neutral, unbranded newsroom tone
[ ] No sentence copied verbatim
[ ] Length in the same range

Now write the neutral version.
""".strip("\n")
    ),
    "neutral_v8": _grid(
        """
Step 1 (internal): Extract 6–10 core facts/claims from the article below.
Step 2 (output): Write a neutral news story using only those facts.

Article:
{semantic_ref_text}

Constraints:
- No sentence-level mirroring.
- Standard vocabulary and punctuation.
- Similar length to the original.
""".strip("\n")
    ),
    "neutral_v9": _grid(
        """
Paraphrase the article into a plain newsroom voice. Preserve meaning, avoid stylistic quirks, do not copy sentences, and keep a similar length:

{semantic_ref_text}
""".strip("\n")
    ),
    "imitate_v1": _grid(
        """
You will produce a style-transfer news rewrite.

Inputs:
A) STYLE EXEMPLAR (style only)
{style_ref_text}

B) FACT SOURCE (facts only)
{semantic_ref_text}

Output requirements:
- Write a new English news story using ONLY the facts from B.
- Match the stylistic behavior of A (lead type, paragraphing, attribution habits, transition patterns, and level of formality).
- Do not reuse any sentence or distinctive phrasing from A or B.
- Do not preserve B’s sentence order.
- Keep length similar to B.

Style-strength: MAXIMUM.
Output ONLY the article.

""".strip("\n")
    ),
    "imitate_v2": _grid(
        """
### Task

You are generating **style-transfer training data** for a large language model.

Given:

* a **Style Reference**
* a **Semantic Anchor**

Your job is to **transfer writing style only**, not meaning.

---

### Step 1 — Extract a *STRUCTURED* Style Profile (machine-usable)

From the Style Reference, extract a **Style Profile** with the following fixed fields.
Do **not** summarize loosely.
Each field must be explicit, concrete, and reusable.

**Style Profile (output as a bullet list or JSON-like structure):**

1. **Narrative stance**

   * e.g. strictly factual / mildly interpretive / analytical / report-log style

2. **Lead pattern**

   * e.g. immediate fact lead / contextual contrast / delayed identification / summary-first

3. **Sentence statistics**

   * average sentence length (short / medium / long)
   * dominant sentence type (simple / compound / complex)

4. **Paragraph function pattern**

   * e.g. fact → attribution → example → background → impact
   * number of sentences per paragraph (range)

5. **Attribution behavior**

   * frequency of quotes (none / few / frequent)
   * quote placement (early / mid / late)
   * attribution verbs used (said / added / noted / according to…)

6. **Transition behavior**

   * time-based / contrastive / additive / enumerative
   * whether ordinal framing is used (first / second / third / another)

7. **Lexical tone**

   * neutral / bureaucratic / evaluative / technical
   * adjective density (low / medium / high)

8. **Rhetorical constraints**

   * what the text avoids (emotion, metaphor, speculation, narrative framing)

Return **ONLY** the Style Profile.

---

### Step 2 — Generate a new article using the Style Profile

Write a NEW English news article using **ONLY** the facts from the Semantic Anchor.

**Hard constraints:**

* Meaning must stay consistent with the Semantic Anchor.
* Do NOT reuse exact sentences or clause-level phrasing.
* Do NOT mirror sentence order from the anchor.
* The article MUST obey every element of the Style Profile.
* Length should be roughly similar to the anchor.

You are transferring **style parameters**, not copying wording.

---

### Step 3 — Style compliance check (silent self-correction)

Before finalizing, verify internally:

* Lead pattern matches the Style Profile
* Sentence length & paragraph rhythm are consistent
* Attribution frequency matches
* Forbidden rhetorical moves are avoided

If mismatch exists, revise once before output.

---

[Style Reference]
{style_ref_text}

[Semantic Anchor]
{semantic_ref_text}
""".strip("\n")
    ),
    "imitate_v3": _grid(
        """
Task: Rewrite the FACT SOURCE into a new English news article.

STYLE SOURCE (voice template):
{style_ref_text}

FACT SOURCE (only facts allowed):
{semantic_ref_text}

Constraints:
1) Facts must come exclusively from FACT SOURCE; no added details.
2) The first paragraph must follow the same lead strategy as STYLE SOURCE (e.g., summary lead vs. context lead).
3) Each subsequent paragraph must mimic STYLE SOURCE’s paragraph function pattern (fact → attribution → background → consequence, etc.).
4) Use similar attribution density and placement as STYLE SOURCE.
5) No sentence copying and no line-by-line paraphrase.
6) Similar length.

Return ONLY the finished story.

""".strip("\n")
    ),
    "imitate_v4": _grid(
        """
You are doing voice transfer for news writing.

1) Read the STYLE SAMPLE and internalize its voice: cadence, typical transitions, quote handling, paragraph length.
2) Then write a NEW article using ONLY the FACTS from the FACT PASSAGE.

STYLE SAMPLE:
{style_ref_text}

FACT PASSAGE:
{semantic_ref_text}

Rules:
- Do not copy wording from either passage.
- Do not follow the FACT PASSAGE ordering.
- Make the finished piece sound like it could sit next to the STYLE SAMPLE in the same outlet.
- Similar length; output only the article.

""".strip("\n")
    ),
}
