from flask import Flask, request, send_file
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    use_auth_token=True
)#.to("cuda")

@app.route("/stable_diffusion", )
def get_result():
    prompt = request.args.get("prompt")
    prompt_path = "data/" + prompt.replace(" ", "_") + ".png"
    print(prompt, prompt_path)
    image = pipe(prompt)["sample"][0] 
    image.save(prompt_path)

    return send_file(prompt_path, mimetype='image/png')

if __name__=="__main__":
    app.run(debug = True)