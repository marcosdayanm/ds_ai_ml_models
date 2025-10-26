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
        
        # Hiperparámetros Policy Gradient, funcionan para un grid 10x10
        self.gamma = 0.99# factor de descuento
        self.alpha = 0.01  # learning rate
        self.baseline = 0.0  # baseline para reducir varianza
        self.episodes = episodes

        self.agent_pos = (1, 1)
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 5 * size**2
        
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

        # base por movimiento
        reward = -0.01
        

        goal_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        prev_distance = abs(prev_pos[0] - self.goal_pos[0]) + abs(prev_pos[1] - self.goal_pos[1])        
        if goal_distance < prev_distance:
            reward += 0.3
        elif goal_distance > prev_distance:
            reward -= 0.2

        # premio por chocar contra pared
        if prev_dir == self.agent_dir and prev_pos == self.agent_pos:
            reward = -0.5
        
        # premio por llegar a la meta
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 50.0
            terminated = True
            # print("Reached the goal")
        
        return obs, reward, terminated, truncated, info
    
    def get_logits(self, state) -> np.ndarray:
        """Get logits for a given state"""
        if state not in self.logits:
            self.logits[state] = np.zeros(len(self.actions))
        return self.logits[state]

    def get_state(self) -> tuple[int, int, int]:
        return (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def train(self):
        for ep in range(self.episodes):
            self.reset()
            done = False
            truncated = False
            total_reward = 0.0
            episode_data = []  # (estado, acción, recompensa)

            # 1 generar episodio completo
            while not (done or truncated):
                state = self.get_state()
                logits = self.get_logits(state)
                probs = self.softmax(logits)
                
                # Muestrear acción de la política
                action = np.random.choice(len(self.actions), p=probs)
                
                obs, reward, done, truncated, info = self.step(action)  #type: ignore
                
                episode_data.append((state, action, float(reward)))
                total_reward += float(reward)
            
            # 2 calcular returns G_t para cada paso t
            returns = []
            G = 0.0
            for i in range(len(episode_data) - 1, -1, -1):
                _, _, reward = episode_data[i]
                G = reward + self.gamma * G
                returns.insert(0, G)
            
            # 3actualizar baseline (promedio de returns del episodio)
            episode_return = np.mean(returns)
            self.baseline = 0.9 * self.baseline + 0.1 * episode_return
            
            # 4 actualizar parámetros θ (logits) usando policy gradient
            for t, (state, action, _) in enumerate(episode_data):
                G_t = returns[t]
                advantage = G_t - self.baseline  # reducir varianza con baseline
                
                # Recalcular probabilidades para este estado
                logits = self.get_logits(state)
                probs = self.softmax(logits)
                
                for a in range(len(self.actions)):
                    if a == action:
                        grad_log_pi = 1 - probs[a]
                    else:
                        grad_log_pi = -probs[a]
                    
                    self.logits[state][a] += self.alpha * grad_log_pi * advantage
            
            # Registro
            self.history.append({
                "episode": ep + 1,
                "baseline": self.baseline,
                "total_reward": total_reward,
                "episode_return": episode_return,
                "states_learned": len(self.logits),
                "reached_goal": done,
                "truncated": truncated
            })
            status = "GOAL" if done else "TIMEOUT"
            if not done:
                print(f"Ep {ep+1}: Reward={total_reward:.2f}, G_avg={episode_return:.2f}, Baseline={self.baseline:.2f} {status}")

    def plot_history(self):
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(14, 5))
        
        # Reward total
        plt.subplot(1, 3, 1)
        plt.plot(df["episode"], df["total_reward"], label="Total Reward", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Progress (Policy Gradient)")
        plt.legend()
        plt.grid(True)
        
        # Episode returns vs Baseline
        plt.subplot(1, 3, 2)
        plt.plot(df["episode"], df["episode_return"], label="G_t (Return)", alpha=0.6, color='green')
        plt.plot(df["episode"], df["baseline"], label="Baseline (b)", color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Baseline")
        plt.title("Baseline Evolution")
        plt.legend()
        plt.grid(True)
        
        # Tasa de éxito
        plt.subplot(1, 3, 3)
        plt.plot(df["episode"], df["temperature"], label="Temperature", color='orange', alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Temperature")
        plt.title("Temperature Decay")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas finales
        print(f"Total de Episodios: {len(df)}")
        print(f"Tasa de Éxito: {df['reached_goal'].mean() * 100:.2f}%")
        print(f"Recompensa Promedio: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
        print(f"Baseline Final: {self.baseline:.2f}")
        print(f"Estados Aprendidos: {len(self.logits)}")
    
    def dump_model(self, filename: str = "policy_gradients_data.pkl"):
        """Saves the model"""
        data_to_save = {
            "logits": self.logits,
            "baseline": self.baseline,
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
        if "size" in data:
            self.size = data["size"]
        
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
                obs, reward, done, truncated, info = self.step(action)  #type: ignore
                time.sleep(0.05)


if __name__ == "__main__":
    plot_history = False
    skip_training = False
    skip_testing = False
    size = 10
    episodes = 2000

    # Training
    if not skip_training:
        print(f"Training Policy Gradient (softmax) agent for {episodes} episodes on {size}x{size} grid...")
        env = SoftmaxEnv(render_mode=None, size=size, episodes=episodes)
        env.train()
        env.dump_model()
        
        if plot_history:
            env.plot_history()

    # Testing
    if not skip_testing:
        print(f"\nTesting agent with human rendering...")
        env_human = SoftmaxEnv(size=size)
        env_human.load_model()
        env_human.test_human()
