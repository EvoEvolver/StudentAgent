# Benchmark: FictionalQA (https://arxiv.org/pdf/2506.05639)


**Goal:** Evaluate robustness of memory extraction and utilization across structured vs. unstructured sources.  
**Evaluation:** Multiple-choice accuracy (4- and 10-choice). Deterministic selection to ensure reproducibility.  
**Agents compared:** Naive RAG, Agentic RAG, Student (structured memory extraction).

---

## Part 1 — Structured Memory (Fictsheets)
- **Memory:** First 20 fictsheets (`event_000`–`event_019`).  
- **MCQ&A:** For each fictsheet, use the **first 5 MCQs** (from any linked documents, filtered by infeasible-blind dedup). TODO: which questions!? Maybe just encyclopedia
- **Evaluation:** Bar plot (accuracy ± 95% CI across fictsheets) comparing agent types  
- **Interpretation:** Does the Student outperform others when memory is already structured?


---

## Part 2 — Unstructured Memory (Documents by Style)
- **Memory:** First 50 events (`event_000`–`event_049`), pick **lexicographically first style** document per event.  
- **MCQ&A:** For each event:  
  - 5 MCQs from the chosen style (in-distribution).  
  - 5 MCQs from a different style of the same event (cross-style validation).  TODO: important? which?
- **Evaluation:** Grouped bar plot (agent × per-style-accuracy), separated into training-styles and validation-styles
- **Interpretation:** Does the Student extract/utilize better across diverse styles? Does style density (news vs. social vs. corporate) impact performance?

---

## Notes
- Deterministic selection avoids random-seed bias.  
- Results not directly numerically comparable to the paper’s finetuning curves, but **relative patterns across splits** (e.g., fictsheets being harder) can be meaningfully compared.  
