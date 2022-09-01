from flask import Flask, request, send_file
from torch import autocast, cuda
import torch, os
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

access_token = "enter access token"


    
# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    torch_dtype=torch.float16,
    use_auth_token=access_token
)

if cuda.is_available():
    print("using cuda")
    pipe = pipe.to("cuda")

@app.route("/stable_diffusion")
def get_result():
    prompt = request.args.get("prompt")
    prompt_path = os.path.join(root_dir(), "data",prompt.replace(" ", "_") + ".png")
    print(prompt, prompt_path)
    if cuda.is_available():
        with autocast("cuda"):
            image = pipe(prompt)["sample"][0] 
    else:
        image = pipe(prompt)["sample"][0] 

    image.save(prompt_path)
    torch.cuda.empty_cache()
    
    return send_file(prompt_path, mimetype='image/png')

def root_dir():
    return os.path.abspath(os.path.dirname(__file__))
    
if __name__=="__main__":
    app.run(host="127.0.0.1", port="5555",debug = True)