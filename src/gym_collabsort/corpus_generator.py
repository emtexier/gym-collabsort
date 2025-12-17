import json
import os
import random
from typing import Dict, List

import numpy as np
from PIL import Image

from gym_collabsort.config import Config
from gym_collabsort.envs.env import CollabSortEnv, RenderMode, Action

# =======================
# PARAMÈTRES, à mettre dans un éventuel fichier config.py
# =======================

N_EXAMPLES = 5 # nombre d'exemples générés pour une phase donnée 
MAX_STEPS = 400 # longueur de l'épisode 
MS_BETWEEN_EACH_FRAME = 250 # fluidité de l'avancement du bras / saccades 

PHASE_NUMBER = 0
N_OBJECTS = 1 # nombre d'objets = 1 dans la phase 0

def serialize_obs(obs: dict) -> dict:
    return {
        "self": obs["self"].tolist() if isinstance(obs["self"], np.ndarray) else obs["self"],
        "robot": obs["robot"].tolist() if isinstance(obs["robot"], np.ndarray) else obs["robot"],
        "objects": [
            {
                "coords": obj["coords"].tolist(),
                "color": int(obj["color"]),
                "shape": int(obj["shape"]),
            }
            for obj in obs["objects"]
        ],
    }

# =======================
# GENERATEUR
# =======================

class Phase0GeneratorAgent:

    def __init__(self, n_examples: int, max_steps: int, seed: int = 42, render_mode: RenderMode = RenderMode.RGB_ARRAY):
        self.n_examples = n_examples
        self.max_steps = max_steps
        self.seed = seed
        self.render_mode = render_mode
        random.seed(seed)

    def choose_agent_action(self, obs, start_row):
        """
        Agent simple mais intelligent :
        - Ne descend jamais en dessous de sa ligne de départ
        """
        possible_actions = list(Action)

        agent_row = obs["self"][0]  # ligne actuelle de l'agent
        if agent_row >= start_row:
            possible_actions = [a for a in possible_actions if a != Action.DOWN]

        # Choix aléatoire parmi les actions autorisées
        return np.random.choice(possible_actions)

    def save_gif(self, frames, episode_id: int):
        folder = f"gifs_episodes/{PHASE_NUMBER}"
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/episode_{episode_id}.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=MS_BETWEEN_EACH_FRAME,
            loop=0,
        )
        print(f"GIF sauvegardé : {path}")

    def generate_episode(self, env, episode_id: int) -> Dict:
        obs, info = env.reset()
        steps = []
        frames = []

        start_row = obs["self"][0]

        # Force le robot à ne rien faire car phase0 = le robot bouge seul 
        original_choose_action = env.robot.choose_action
        env.robot.choose_action = lambda: Action.NONE

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        for t in range(self.max_steps):
            step = {"t": t}
            step["observation"] = serialize_obs(obs)

            action = self.choose_agent_action(obs, start_row)
            step["action"] = action.value 

            obs, reward, terminated, truncated, info = env.step(action)

            step["reward"] = reward
            step["info"] = info
            steps.append(step)

            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

            if terminated or truncated:
                break

        env.robot.choose_action = original_choose_action

        if len(frames) > 1:
            self.save_gif(frames, episode_id)

        return {
            "episode_id": episode_id,
            "phase": PHASE_NUMBER,
            "steps": steps
        }

    def generate_corpus(self) -> List[Dict]:
        corpus = []
        cfg = Config(n_objects=N_OBJECTS)

        for ep in range(self.n_examples):
            env = CollabSortEnv(render_mode=self.render_mode, config=cfg)
            episode = self.generate_episode(env, ep)
            corpus.append(episode)
            env.close()

        return corpus

    def save(self, corpus: List[Dict], filename: str):
        with open(filename, "w") as f:
            json.dump(corpus, f, indent=2)


if __name__ == "__main__":
    generator = Phase0GeneratorAgent(
        n_examples=N_EXAMPLES,
        max_steps=MAX_STEPS,
    )

    corpus = generator.generate_corpus()
    generator.save(corpus, f"corpus_phase_{PHASE_NUMBER}.json")

    print(f"Phase {PHASE_NUMBER} générée et GIFs sauvegardés")