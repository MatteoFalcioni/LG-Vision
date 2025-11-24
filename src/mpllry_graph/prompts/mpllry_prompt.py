prompt = """
Your goal is to accept or discard images given their quality. 
In the following I will list criteria that you need to follow for your evaluation, and show you examples of images that you would accept or reject.

## Evaluation Criteria

Follow these rules to evaluate whether an image is acceptable.

Accept the image ONLY if all of the following parameters are satisfied:

- The image clearly shows a complete view of the scene without being occluded.

- The image is sharp and not blurry.

- The image features significant architectural elements such as buildings, houses, porticoes, churches, etc.

Discard the image if any of the following conditions are met:

- The view is occluded or obstructed by objects.

- The image is excessively blurry or out of focus, to the point where elements of the image are not well distinguishable.

- The camera angle is too tilted downwards, capturing only the street or the ground.

- The image does not contain any significant architectural elements.

The image can be tilted, that is not a problem.

Your answers MUST be composed of two parths: 
* a 'response' that you will fill with 'yes' if you accept the image, or fill with 'no' if you discard it;
* an 'reason' that you will fill with the reason explaining why you accepted or discarded an image. 

## Examples 

You will be shown in the following some examples of acceptable or discardable images.
"""