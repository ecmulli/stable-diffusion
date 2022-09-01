# Stable Diffusion

## Set Up
1. Clone repo
2. Create virtual environment 
    * `cd stable-diffusion`
    * `python3 -m venv venv`
    * `source venv/bin/activate`
2. Install requirements `pip install -r requirements.txt` 
3. Sign up for Hugging Face and generate auth token (https://huggingface.co/docs/hub/security-tokens)
4. Request access to stable diffusion model https://huggingface.co/CompVis/stable-diffusion-v1-4
5. Input access token to `access_token` in `server.py` and `stable-diffusion.py`

## Playground

3. Run `stable-diffusion.ipynb` - you can change the prompt and test out different results OR
4. Run `venv/bin/python stable-diffusion.py` to produce a single result. You can change the prompt and run it again. NOTE: loading the model into memory will be done every time, this is definitely not the most efficient way to do it. 

## Server

1. run `venv/bin/python server.py`
2. endpoint available at `<ipaddress>:port/stable-diffusion?prompt=<prompt>` 