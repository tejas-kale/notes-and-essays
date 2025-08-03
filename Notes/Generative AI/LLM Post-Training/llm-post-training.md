# Post-Training of LLMs

Large Language Models (LLMs) are neural networks (based on the Transformer architecture) that are trained to predict the next token in a given language. In order to get these models to give meaningful answers to queries like "What is the capital of France?" or "Write a Python function that checks if the input string is a palindrome", the LLMs need to be trained once more with a different type of data. The first training of LLMs is called *Pre-Training* and the second training is known as *Post-Training*.

For pre-training, a large amount (ideally in Gigabytes or order of magnitude higher) of text data is required. A LLM is then pre-trained to predict the next token in this data given the previous *n* tokens. Hence, pre-training is a form of self-supervised learning. 

For post-training, labelled data needs to be available. The type of labelled data that is needed for post-training depends on the technique used for this purpose.
- For Supervised Fine Tuning (SFT), the data consists of a question and the correct (desirable) answer to the question.
- For Direct Preference Optimisation (DPO), the data consists of a question, the correct (desirable) answer, and the wrong (undesirable) answer.  
- For Reinforcement Learning, the data consists of a question and the correct (desirable) answer to the question.


