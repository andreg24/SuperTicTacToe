import numpy as np
from PIL import Image

import ultimatetictactoe.ultimatetictactoe


def gif_from_simulation(path_name, duration):

    env = ultimatetictactoe.env(render_mode="rgb_array")
    env.reset(42)

    imgs_np = []
    for agent in env.agent_iter():
        imgs_np.append(env.render())
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            mask = observation["action_mask"]
            action = env.action_space(agent).sample(mask)

        env.step(action)
    env.close()
    imgs_np.append(env.render())
    numpy_to_gif(imgs_np, path_name, duration)


def numpy_to_gif(imgs_np, path_name, duration=200):
    imgs = [Image.fromarray(img_np) for img_np in imgs_np]
    imgs[0].save(
        path_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0
    )
