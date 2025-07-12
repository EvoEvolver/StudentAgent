# Final Benchmarking Setup


1. WikiDYK (n = 50 facts)
    - Learning:  Multiple Facts (+ Single Fact)
    - Evaluation:
        - Baselines:
            - answerable ~ LLM + fact + question + answer
            - pretraining ~ LLM + question
            - LLM + fact + question
            - LLM + RAG@learning + question
            - LLM + RAG + question
            
        - Metrics:
            - Accuracy overall
            - Accuracy, if answerable
            - Per question style: boxplot (mean, std, median)
            - Per fact: histogram (mean)

2. FictionalQA (n_e = 20 events) * (n_s = 10 styles)
    - Learning: 
        - n_e = 1, 0.5*n_e, n_s
        - n_s = 1, 0.5*n_s, n_s

    - Evaluation:
        - Baselines:
            - answerable ~ LLM + style(s) + question + answer
            - pretraining ~ LLM + question
            - LLM + style(s) + question
            - LLM + RAG + question
            
        - Metrics:
            - Accuracy overall
            - Accuracy, if answerable
            - Plot: n_e vs accuracy
            - Plot: n_s vs accuracy
            - 2D?