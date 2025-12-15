# A LANCER DEPUIS SRC 
# avec python -m gym_collabsort.corpus_generator

import json
import os
import random
from enum import Enum
from typing import Dict, List

import numpy as np
from PIL import Image

from gym_collabsort.config import Config
from gym_collabsort.envs.env import CollabSortEnv, RenderMode
from gym_collabsort.envs.robot import Robot

N_EXAMPLES = 3 # nb d’épisodes à générer
MAX_STEPS = 40 # nb max d’étapes par épisode
MS_BETWEEN_EACH_FRAME =  250 # ms entre les frames

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Mapping difficulté → configuration
DIFFICULTY_CONFIGS = {
    Difficulty.EASY: {"n_objects": 1},
    Difficulty.MEDIUM: {"n_objects": 5},
    Difficulty.HARD: {"n_objects": 10},
}

# L’environnement renvoie des données qui ne sont pas directement JSON-compatibles
# Cette fonction convertit tout en types simples
def serialize_obs(obs: dict) -> dict:
    """
    Convertit une observation en types JSON-compatibles.
    np.ndarray → list, Enum → int
    """
    return {
        "self": obs["self"].tolist() if isinstance(obs["self"], np.ndarray) else obs["self"],
        "robot": obs["robot"].tolist() if isinstance(obs["robot"], np.ndarray) else obs["robot"],
        "objects": [
            {
                "coords": obj["coords"].tolist() if isinstance(obj["coords"], np.ndarray) else obj["coords"],
                "color": int(obj["color"]),
                "shape": int(obj["shape"]),
            }
            for obj in obs["objects"]
        ],
    }


class CorpusGenerator:
    def __init__(
        self,
        n_examples: int = 200,
        max_steps: int = 20,
        seed: int = 42,
        render_mode: RenderMode = RenderMode.NONE, # mode de rendu (NONE ou RGB)
        include_frames: bool = False,
    ):
        self.n_examples = n_examples
        self.max_steps = max_steps
        self.seed = seed
        self.render_mode = render_mode
        self.include_frames = include_frames
        random.seed(seed)

    # Capture l’image au début de chaque épisode
    def save_initial_frame(self, env: CollabSortEnv, episode_id: int, difficulty: Difficulty):
        frame = env.render()
        if frame is None:
            return

        folder = f"images_episodes/{difficulty.value}"
        os.makedirs(folder, exist_ok=True)

        img = Image.fromarray(frame)
        img.save(f"{folder}/episode_{episode_id}.png")

        print("Render mode =", env.render_mode)
        print("Frame =", type(frame), None if frame is None else frame.shape)


        
    # Chaque épisode est une liste de "steps" enregistrant tout ce qui s’est passé
    # Chaque épisode inclut :
    #  - son ID
    #  - la difficulté
    #  - ses données
    def generate_one_episode(self, env: CollabSortEnv, episode_id: int, difficulty: Difficulty) -> Dict:
        episode = []
        obs, info = env.reset()

        # Liste des frames POUR LE GIF (si render_mode RGB)
        frames = []

        # Frame initiale
        if self.render_mode == RenderMode.RGB_ARRAY:
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

        for t in range(self.max_steps):
            step_data = {"t": t}

            obs_serializable = serialize_obs(obs)
            step_data["observation"] = obs_serializable

            # agent complètement aléatoire
            # action = env.action_space.sample()
            
            # agent qui sait que s'il est en bas, il doit avancer vers le haut par ex
            robotic_agent = Robot(
                board=env.board,
                arm=env.board.agent_arm,
                rewards=env.config.agent_rewards
            )

            action = robotic_agent.choose_action().value

            step_data["action"] = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            step_data["reward"] = reward
            step_data["info"] = info

            # Enregistrer la frame pour le GIF
            if self.render_mode == RenderMode.RGB_ARRAY:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            episode.append(step_data)

            if terminated or truncated:
                break

        # Sauvegarde du GIF complet
        if self.render_mode == RenderMode.RGB_ARRAY:
            self.save_gif(frames, episode_id, difficulty)

        return episode

    def save_gif(self, frames, episode_id: int, difficulty: Difficulty):
        folder = f"gifs_episodes/{difficulty.value}"
        os.makedirs(folder, exist_ok=True)

        gif_path = f"{folder}/episode_{episode_id}.gif"
        
        # Sauvegarder un GIF animé
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=MS_BETWEEN_EACH_FRAME,
            loop=0
        )

        print(f"GIF sauvegardé → {gif_path}")

    def generate_corpus(self, difficulty: Difficulty) -> List[Dict]:
        cfg_kwargs = DIFFICULTY_CONFIGS[difficulty]
        cfg = Config(n_objects=cfg_kwargs["n_objects"])
        env = CollabSortEnv(render_mode=self.render_mode, config=cfg)

        corpus = []
        for episode_id in range(self.n_examples):
            env = CollabSortEnv(render_mode=self.render_mode, config=cfg)  # <-- NEW
            episode = self.generate_one_episode(env, episode_id, difficulty)
            corpus.append({
                "episode_id": episode_id,
                "difficulty": difficulty.value,
                "steps": episode
            })
            env.close()
        return corpus


    def save(self, corpus: List[Dict], filename: str = "corpus.json"):
        with open(filename, "w") as f:
            json.dump(corpus, f, indent=2)


if __name__ == "__main__":
    generator = CorpusGenerator(n_examples=N_EXAMPLES, max_steps=MAX_STEPS, render_mode=RenderMode.RGB_ARRAY)

    full_corpus = []
    for diff in Difficulty:
        print(f"Génération pour la difficulté : {diff.value}")
        corpus = generator.generate_corpus(diff)
        full_corpus.extend(corpus)

    generator.save(full_corpus, "corpus_full.json")
    print("Corpus complet généré et sauvegardé !")
