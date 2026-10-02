---
license: mit
language:
- en
library_name: transformers.js
pipeline_tag: text-classification
base_model: NLP-LTU/bertweet-large-sexism-detector
base_model_relation: finetune
tags:
- bertweet
- onnx
- transformers.js
- gbv-classification
- multi-task-classification
---

This repository contains the quantized ONNX export of an English language text classifier for detecting gender based violence (GBV) and GBV-subtype. 
<br>The model predicts a binary label for GBV `pred_binary`: `0` (non-GBV), `1` (GBV).
<br>The model predicts one or more GBV subtypes `pred_category_labels`:  `IDEOLOGICAL-INEQUALITY`, `STEREOTYPING-DOMINANCE`, `OBJECTIFICATION`, `SEXUAL-VIOLENCE`, `MISOGYNY-NON-SEXUAL-VIOLENCE`.

The model was developed for the Digital Harms, Democratic Costs (DHDC) browser extension. It is intended as a research classifier and content-analysis aid, not as a determination about a person or as a substitute for human review.

## Model details
- **Base model:** [`NLP-LTU/bertweet-large-sexism-detector`](https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector)
- **Architecture:** Bertweet based sexism model as encoder with separate binary and five-class multi-label classification heads. 
- **Language:** English
- **Maximum input length:** 514 tokens (inherited from encoder)
- **Published format:** quantized ONNX, opset 17
- **Runtime target:** Transformers.js 2.17.2 with ONNX Runtime Web (WASM/CPU)
- **Repository ID:** `socsys/gbv-model`

## Using Model 

Python version = 3.12.3

    pip install -r requirements.txt

Should declare environment variables (or use .env file) for

    export TRAIN_DATA='/path/to/EXIST/training/data'
    export VAL_PATH='/path/to/EXIST/dev/data'
    export PROCESSED_DATA_DIR='/path/to/where/to/store/labelled/inference/data'

#### Train and validate model

    python main.py --train 
    [--train_data_name] # Default to EXIST, currently code only written for EXIST 
    [--train_data_path] /path/to/train/data # To override environment variable
    [--no_eval] # Does not perform evaluation of model after training


#### Use model for inference 
Also runs validation automatically.

    python main.py --infr --infr_data_path '/path/to/inf/data' 
    [--resume_infr] # if inference was interupted
    [--no_eval] # Does not perform evaluation of model before inference 

#### Export model for use in Chrome extension
Creates folder containing necessary config, tensor files etc, and subfolder containing quantized onnx model. Currently normalizes tensors to match expected format for extension. Can be commented out. 

    python main.py --export  
    [--output] # Output file if not intended to overright tokenizer.json
    [--no_backup] # Skipping backing up of original tokenizer.json 
    
## Model Card

#### Model Details
Model developed by DHDC team at University of Surrey.
Model intended to detect gender based violence (GBV) and GBV subtype in text-based microblogging platform data (e.g. from Bluesky, X.com).
Multilabel multilevel classifier has a binary head (GBV) and a multiple category head (GBV subtype).
Uses [NLP-LTU/bertweet-large-sexism-detector on Huggingface](https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector) as encoder layer. 
Model is trained and validated on EXIST 2025 training data and evaluated on EXIST 2025 development data (test data is withheld for this task). 

#### Intended Use
The model is intended for use in predicted GBV and GBV subtype for short spans of text. It is a monolingual English model. The intended users are researchers looking to estimate levels of GBV on the platform e.g. for a given set of users. 

#### Factors
Random seed = 51
<br>The GBV subtypes from EXIST are: `IDEOLOGICAL-INEQUALITY`, `STEREOTYPING-DOMINANCE`, `OBJECTIFICATION`, `SEXUAL-VIOLENCE`, `MISOGYNY-NON-SEXUAL-VIOLENCE`.
<br>During training, model only learns GBV subtype classification from true examples of GBV. 
<br>Both heads used focal loss, with γ = 2 to downweight easy examples.

#### Metrics
EXIST uses a custom ICM metric to evaluate performance, "punishing" binary GBV/Not GBV errors more than GBV subtype errors. Gold score = 2.15; best performing model = 0.65, 10th best performing model = 0.38. 

#### Training and Evaluation Data
Encoder model trained on Gab (right-wing microblogging platform) and Reddit data; classifier trained on EXIST Twitter data. 

EXIST training and evaluation data collected by searching Twitter for key terms related to GBV. Tweets collected 2021-2022. Tweets contain 5+ words. 
Data labelled by human annotators. Six annotations per data point. Binary label for GBV applied for majority (4/6) label. Draws (3/6) dropped. Category labels applied if at least two annotators give label. 

#### Quantitative Analyses
The model performs well at binary GBV identification. Binary F1 Score: 0.8614
<br>For subtypes, performance is above 0.63 for all subtypes except `MISOGYNY-NON-SEXUAL-VIOLENCE` where performance is poor (0.47). 
<br>Subtype F1 Scores: `-` (no GBV): 0.88; `IDEOLOGICAL-INEQUALITY`: 0.64, `STEREOTYPING-DOMINANCE`: 0.66; `OBJECTIFICATION`: 0.69; `SEXUAL-VIOLENCE`: 0.67; `MISOGYNY-NON-SEXUAL-VIOLENCE`: 0.47
<br>Category Macro F1 Score: 0.6696
<br>Predicted ICM: 0.4823 (likely comparable to top 10 performance on test set for Plaza et al., 2025)

#### Ethical Considerations
Evaluated using hard labels. Hard (majority vote) labels may "suppress" minoritised voices. Annotation performed by expert annotators, 3 male, 3 female. If there is a perfect gender split in labels, example will be dropped, rather than "privileging" female expertise on GBV through lived experience. 

#### Caveats 
Given training data, likely model may underperform on posts using up-to-date slang etc, posts shorter than 5 words. 
Trained on multiple microblogging platforms but may underperform on other platforms e.g. Bluesky. 
<br>Only tested for English language, uses English language encoder model (though uses multilingual finetuning data as this improved performance).
<br>Likely fails to distinguish reports of GBV from instances of GBV as this remains challenging.
Prediction of `MISOGYNY-NON-SEXUAL-VIOLENCE` is very poor and label should be discarded. 

## License and attribution

This repository is released under the MIT License. The upstream [`NLP-LTU/bertweet-large-sexism-detector`](https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector) model does not provide license information. 

The finetuning dataset EXISt stipulates that:
<ul>
<li> The dataset should only be used for scientific or research purposes. Any other use is explicitly prohibited. </li>
<li> The datasets must not be redistributed or shared in part or full to any third party. Redirect interested parties to this website. </li>
<li> All data released for these tasks are under the CC BY-SA 4.0 License (https://creativecommons.org/licenses/by-sa/4.0/legalcode). </li>
<li> Data must not be used for providing surveillance, analyses or research that isolates a group of individuals or any single individual for any unlawful or discriminatory purpose. </li>
</ul> 

MIT licensing of the model artifacts does not grant rights to third-party input data, platform content, personal data, trademarks, or generated outputs. Downstream users remain responsible for those rights and obligations.

### Citations
The following are citations for the encoder model and finetuning data respectively. 

```bibtex
@inproceedings{Al-Azzawi_Kovács_Nilsson_Adewumi_Liwicki_2023, address={Toronto, Canada}, title={NLP-LTU at SemEval-2023 Task 10: The Impact of Data Augmentation and Semi-Supervised Learning Techniques on Text Classification Performance on an Imbalanced Dataset}, url={https://aclanthology.org/2023.semeval-1.196/}, DOI={10.18653/v1/2023.semeval-1.196}, booktitle={Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)}, publisher={Association for Computational Linguistics}, author={Al-Azzawi, Sana and Kovács, György and Nilsson, Filip and Adewumi, Tosin and Liwicki, Marcus}, editor={Ojha, Atul Kr. and Doğruöz, A. Seza and Da San Martino, Giovanni and Tayyar Madabushi, Harish and Kumar, Ritesh and Sartori, Elisa}, year={2023}, month=july, pages={1421–1427} }


 @article{Plaza_Carrillo-de-Albornoz_Arcos_Rosso_Spina_Amigó_Gonzalo_Morante_2025, address={Madrid, Spain}, title={Overview of EXIST 2025: Learning with Disagreement for Sexism Identification and Characterization in Tweets, Memes, and TikTok Videos (Extended Overview)}, journal={CLEF 2025 Working Notes}, author={Plaza, Laura and Carrillo-de-Albornoz, Jorge and Arcos, Iván and Rosso, Paolo and Spina, Damiano and Amigó, Enrique and Gonzalo, Julio and Morante, Roser}, year={2025}, month=sept, language={en} }

```