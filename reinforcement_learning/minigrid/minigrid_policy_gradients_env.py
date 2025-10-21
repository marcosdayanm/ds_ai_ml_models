from __future__ import annotations

import numpy as np
import pandas as pd

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions
import matplotlib.pyplot as plt

import random
import time
import pickle
import argparse

class SoftmaxEnv(MiniGridEnv):
    """Agente Policy Gradient con Softmax para MiniGrid"""
    
    def __init__(
        self,
        size=10,
        episodes=2000,
        max_steps: int | None = None,
        logits: dict = {},     
        **kwargs,
    ):
        self.size = size
        self.logits = logits
        self.history = []
        
        # Hiperparámetros Policy Gradient
        self.gamma = 0.95                # factor de descuento
        self.temperature = 3.0           # temperatura para softmax (exploración)
        self.baseline = 0.0              # baseline para reducir varianza
        self.baseline_alpha = 0.1        # tasa de actualización del baseline
        self.policy_alpha = 0.1          # tasa de aprendizaje de política
        self.temperature_decay = 0.995   # decay de temperatura
        self.temperature_min = 0.3       # temperatura mínima
        self.episodes = episodes

        self.agent_pos = (1, 1)
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 3 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        self._place_agent()
        
        self.mission = "Reach the goal"

    def _place_agent(self):
        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            pos = (x, y)
            
            if (self.grid.get(*pos) is None and pos != self.goal_pos):
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)
                break

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs

    def step(self, action: Actions):
        prev_pos = self.agent_pos
        prev_dir = self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        # Penalización por movimiento normal
        if self.grid.get(*self.agent_pos) is None:
            reward = -0.05
        
        # Recompensa por acercarse a la meta (distancia Manhattan)
        goal_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        prev_distance = abs(prev_pos[0] - self.goal_pos[0]) + abs(prev_pos[1] - self.goal_pos[1])
        
        if goal_distance < prev_distance:
            reward = float(reward) + 0.1  # Bonus por acercarse
        elif goal_distance > prev_distance:
            reward = float(reward) - 0.05  # Penalización por alejarse

        # Penalización por chocar contra pared
        if prev_dir == self.agent_dir and prev_pos == self.agent_pos:
            reward = -1.0
        
        # Recompensa por llegar a la meta
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 100
            terminated = True
            # print("🎯 Reached the goal!")
        
        return obs, reward, terminated, truncated, info
    
    def get_logits(self, state) -> np.ndarray:
        """Obtiene los logits para un estado"""
        if state not in self.logits:
            self.logits[state] = np.zeros(len(self.actions))
        return self.logits[state]

    def get_state(self) -> tuple[int, int, int]:
        return (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convierte logits a probabilidades con softmax"""
        exp_logits = np.exp((logits - np.max(logits)) / self.temperature)
        return exp_logits / np.sum(exp_logits)
    
    def train(self):
        """Entrena el agente con Policy Gradient + Softmax"""
        for ep in range(self.episodes):
            obs = self.reset()
            done = False
            truncated = False
            total_reward = 0.0
            episode_transitions = []  # (estado, acción, recompensa, probs)

            while not (done or truncated):
                state = self.get_state()
                logits = self.get_logits(state)
                probs = self.softmax(logits)
                
                # Muestrear acción según probabilidades softmax
                action = np.random.choice(len(self.actions), p=probs)
                
                obs, reward, done, truncated, info = self.step(action)
                
                episode_transitions.append((state, action, float(reward), probs.copy()))
                total_reward += float(reward)
            
            # Actualizar baseline con media móvil
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * total_reward
            
            # Calcular returns (recompensas acumuladas desde cada paso)
            returns = []
            G = 0.0
            for i in range(len(episode_transitions) - 1, -1, -1):
                _, _, reward, _ = episode_transitions[i]
                G = reward + self.gamma * G
                returns.insert(0, G)
            
            # Normalizar returns para reducir varianza
            if len(returns) > 1:
                returns = np.array(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = np.array(returns)
            
            # Actualizar logits usando returns normalizados
            for idx, (state, action, _, probs) in enumerate(episode_transitions):
                advantage = returns[idx]
                
                # Actualización de gradiente de política
                for a in range(len(self.actions)):
                    if a == action:
                        # Refuerza la acción tomada
                        self.logits[state][a] += self.policy_alpha * advantage * (1 - probs[a])
                    else:
                        # Debilita otras acciones
                        self.logits[state][a] -= self.policy_alpha * advantage * probs[a]
            
            # Decay de temperatura
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
            
            self.history.append({
                "episode": ep + 1,
                "baseline": self.baseline,
                "temperature": self.temperature,
                "total_reward": total_reward,
                "states_learned": len(self.logits),
                "reached_goal": done,
                "truncated": truncated
            })
            status = "GOAL" if done else "TIMEOUT"
            if not done:
                print(f"Ep {ep+1}: Reward={total_reward:.2f}, Temp={self.temperature:.2f}, Baseline={self.baseline:.2f} {status}")

    def plot_history(self):
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(df["episode"], df["total_reward"], label="Total Reward", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Progress (Policy Gradient)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(df["episode"], df["baseline"], label="Baseline", color='red', alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Baseline")
        plt.title("Baseline Evolution")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(df["episode"], df["temperature"], label="Temperature", color='orange', alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Temperature")
        plt.title("Temperature Decay")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def dump_model(self, filename: str = "policy_gradients_data.pkl"):
        """Saves the model"""
        data_to_save = {
            "logits": self.logits,
            "baseline": self.baseline,
            "temperature": self.temperature,
            "size": self.size,
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str = "policy_gradients_data.pkl"):
        """Loads the model"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        if "logits" in data:
            self.logits = data["logits"]
        if "baseline" in data:
            self.baseline = data["baseline"]
        if "temperature" in data:
            self.temperature = data["temperature"]
        
        print(f"Model loaded from {filename}")

    def test_human(self):
        """Prueba el agente con renderizado visual"""
        self.render_mode = "human"
        for _ in range(5):
            done = False
            obs = self.reset()
            time.sleep(0.3)
            while not done:
                state = self.get_state()
                logits = self.get_logits(state)
                probs = self.softmax(logits)
                action = np.random.choice(len(self.actions), p=probs)
                obs, reward, done, truncated, info = self.step(action)
                time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a Policy Gradient (softmax) agent in MiniGrid")
    
    parser.add_argument(
        "--plot-history", "-ph",
        action="store_true",
        help="Plot the training history after training"
    )
    
    parser.add_argument(
        "--skip-testing", "-stt",
        action="store_true",
        help="Skip testing the agent with human-visible rendering"
    )
    
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=10,
        help="Grid size (default: 10)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=2000,
        help="Number of training episodes (default: 2000, softmax needs more episodes)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="policy_gradients_data.pkl",
        help="Filename for saving/loading model (default: policy_gradients_data.pkl)"
    )
    
    parser.add_argument(
        "--skip-training", "-st",
        action="store_true",
        help="Skip training the model"
    )
    
    args = parser.parse_args()

    # Training
    if not args.skip_training:
        print(f"Training Policy Gradient (softmax) agent for {args.episodes} episodes on {args.size}x{args.size} grid...")
        env = SoftmaxEnv(render_mode=None, size=args.size, episodes=args.episodes)
        env.train()
        env.dump_model(args.model)
        
        if args.plot_history:
            env.plot_history()

    # Testing
    if not args.skip_testing:
        print(f"\nTesting agent with human rendering...")
        env_human = SoftmaxEnv(size=args.size)
        env_human.load_model(args.model)
        env_human.test_human()
