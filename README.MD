# Stable Diffusion

## Set Up
1. Clone repo
2. Create virtual environment 
    * `cd stable-diffusion`
    * `python3 -m venv venv`
    * `source venv/bin/activate`
2. Install requirements `pip install -r requirements.txt` 

## Playground

3. Run `stable-diffusion.ipynb` - you can change the prompt and test out different results OR
4. Run `venv/bin/python stable-diffusion.py` to produce a single result. You can change the prompt and run it again. NOTE: loading the model into memory will be done every time, this is definitely not the most efficient way to do it. 

## Server

1. run `venv/bin/python server.py`
2. endpoint available at `<ipaddress>:port/stable-diffusion/?prompt=<prompt>` 