from lstm import lstm

def get_model(is_training,params):
    if(params["model"]=="lstm"):
        model = lstm( is_training, params)
    return model

def get_model_pretrained(is_training,params):
    model=get_model(is_training,params)
    return model