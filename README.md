Should declare environment variables for

    export TRAIN_DATA='/path/to/EXIST/training/data'
    export VAL_PATH='/path/to/EXIST/dev/data'
    export PROCESSED_DATA_DIR='/path/to/where/to/store/labelled/inference/data'

#### Train and validate model

    python main.py --train 
    [--train_data_name] '/path/to/train/data' # To override environment variable
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
Uses [NLP-LTU/bertweet-large-sexism-detector on Huggingface](https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector)) as encoder layer. 
Model is trained and validated on EXIST 2025 training data and evaluated on EXIST 2025 development data (test data is withheld for this task). 
EXIST task: Laura Plaza, Jorge Carrillo-de-Albornoz, Iván Arcos, Paolo Rosso, Damiano Spina, Enrique Amigó, Julio Gonzalo, and Roser Morante. 2025. Overview of EXIST 2025: Learning with Disagreement for Sexism Identification and Characterization in Tweets, Memes, and TikTok Videos (Extended Overview). CLEF 2025 Working Notes.


#### Intended Use
The model is intended for use in predicted GBV and GBV subtype for short spans of text. It is a monolingual English model. The intended users are researchers looking to estimate levels of GBV on the platform e.g. for a given set of users. 

#### Factors
The GBV subtypes from EXIST are: 
 "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE".
 During training, model only learns GBV subtype classification from true examples of GBV. 

#### Metrics
EXIST uses a custom ICM metric to evaluate performance, "punishing" binary GBV/Not GBV errors more than GBV subtype errors. Gold score = 2.15; best performing model = 0.65, 10th best performing model = 0.38. 

#### Training and Evaluation Data
EXIST training and evaluation data collected by searching Twitter for key terms related to GBV. Tweets collected 2021-2022. Tweets contain 5+ words. Encoder model trained on Gab (right-wing microblogging platform) and Reddit data; classifier trained on Twitter data. 
Data labelled by human annotators. Six annotations per data point. Binary label for GBV applied for majority (4/6) label. Draws (3/6) dropped. Category labels applied if at least two annotators give label. 

#### Quantitative Analyses
The model performs well at binary GBV identification. Binary F1 Score: 0.8614
For subtypes, performance is above 0.63 for all subtypes except MISOGYNY-NON-SEXUAL-VIOLENCE where performance is poor (0.47). 
Subtype F1 Scores: "-": 0.88; IDEOLOGICAL-INEQUALITY: 0.64, "STEREOTYPING-DOMINANCE": 0.66; "OBJECTIFICATION": 0.69; "SEXUAL-VIOLENCE": 0.67; MISOGYNY-NON-SEXUAL-VIOLENCE: 0.47
Category Macro F1 Score: 0.6696
Predicted ICM: 0.4823 (likely comparable to top 10 performance on test set)

#### Ethical Considerations
Evaluated using hard labels. Hard (majority vote) labels may "suppress" minoritised voices. Annotation performed by expert annotators, 3 male, 3 female. If there is a perfect gender split in labels, example will be dropped, rather than "privileging" female expertise on GBV through lived experience. 

#### Caveats 
Given training data, likely model may underperform on posts using up-to-date slang etc, posts shorter than 5 words. 
Trained on multiple microblogging platforms but may underperform on other platforms e.g. Bluesky. 
