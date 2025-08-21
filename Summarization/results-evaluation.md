## Model : BART-Large-CNN 

## Articles tested on  : 2 Long , 3 Medium , 2 Short 

Commentaires : 
### 1- Long articles (2)

ROUGE-L tends to be lower for long articles because abstractive summaries often rephrase or condense multiple sections, reducing exact sequence overlap with reference summaries.

BERTScore remains high, which is expected—semantic similarity is captured even if wording differs.

### 2 - Medium-length articles (3)

ROUGE-L scores are moderate, reflecting a balance: there’s enough content to summarize meaningfully, but not so much that rephrasing drastically lowers overlap.

These are often the articles where abstractive summarization performs best in both ROUGE and BERTScore.

### 3- Short articles (2)

ROUGE-L is generally higher for short articles if summaries closely mirror key points, or lower if the generated summary adds interpretive phrasing.

BERTScore usually remains high, but small differences in phrasing can disproportionately affect ROUGE-L for short texts.


## Results : 
	
"average_rougeL_f1": 0.28660743863399357 -> Moderate  

"average_bert_f1": 0.8952425122261047 -> High 

## Evaluation : 
These results indicate that the model captures the semantic meaning of the original article . 

The moderately low value of the rougeL score indiacates that there’s not much of a lexical overlap between the reference summary and the generated summary but that is quite understandable since the summarization is abstractive so the ways of summarizing without paraphrasing differs . 


