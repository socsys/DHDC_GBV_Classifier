#### Train and validate model
python main.py --train 

#### Use model for inference 
Also runs validation automatically.
    python main.py --inf --inf_data_path ( --resume (if inference interupted))

#### Export model for use in Chrome extension
Creates folder containing necessary config, tensor files etc, and subfolder containing quantized onnx model. 
    python main.py --export  
    
Likely need to normalize tensors to match expected merge style for transformers.js