This folder contains the implementation of the validation pipeline of images downloaded from Mapillary (often referred to as mpllry in code).

The pipeline is very simple: 
- we download an image from the Mapillary API;
- we pass it through the graph where a multimodal agent evaluates them based on the prompt in [mpllry_prompt.py](./prompts/mpllry_prompt.py).

The download happens following some basic criteria: 
1. we sample a random point in a specified area
2. we check that that for that point a Mapillary image exists
3. we download the image 
4. we select the next points at least $\Delta$ meters away from the initial point ($\Delta$ is our only hyperparameter at the moment) 