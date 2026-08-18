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
    
