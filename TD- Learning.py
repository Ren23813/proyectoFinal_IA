#Universida del Valle
#Proyecto Final


#TD-learning

import numpy as np
import random
import pickle
from battleshipENV import BattleshipEnv, BOARD_SIZE, HORIZONTAL, VERTICAL


class TDAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.q_table = {}  # Tabla Q para almacenar los valores de acción-estado

        self.v_table = np.zeros((BOARD_SIZE, BOARD_SIZE))  # Tabla V para almacenar los valores de estado
        self.last_action = None
    
    def place_ships(self, env, player_id):
        env.place_ships_randomly(player_id)

    def select_action(self, obs, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            action=random.choice(valid_actions)
        else:
            q_values = []
            for act in valid_actions:
                r, c=divmod(act, BOARD_SIZE)
                q_values.append(self.v_table[r][c])

            max_idx = np.argmax(q_values)
            action = valid_actions[max_idx]
        
        self.last_action = action
        return action
    

    def update(self, reward,  next_obs, done):
        if self.last_action is None:
            return
        
        r_curr, c_curr = divmod(self.last_action, BOARD_SIZE)

        v_curr = self.v_table[r_curr][c_curr]

        if done: 
            target = reward
        else:
            target = reward + self.gamma * np.max(self.v_table)
        self.v_table[r_curr][c_curr] += self.alpha * (target - v_curr)

#entrenamiento

def train_agent(episodes=1000):
    env=BattleshipEnv()
    agent=TDAgent()

    print("Entrenando agente, por {episodes} episodios...".format(episodes=episodes))

    for i in range(episodes):
        obs0, obs1 = env.reset()
        agent.place_ships(env, 0)
        env.place_ships_randomly(1)

        done=False
        while not env.done:
            current_player=env.current_player

            if current_player==0:
                obs=env.get_observation(current_player)
                action=agent.select_action(obs, obs["valid_actions"])

                #ejecuta accion
                _,_, reward, done, info=env.step(action)

                #actualizar
                next_obs=env.get_observation(0)
                agent.update(reward, next_obs, done)
            else:
                valid=env.get_valid_actions(1)
                env.step(random.choice(valid))
        if (i+1) % 100 == 0:
            print(f"Episodio {i+1}/{episodes} completado.")  

            #se guarda el modelo cada 100 episodios
            with open("td_agent.pkl", "wb") as f:
                pickle.dump(agent.v_table, f)
            print("Se finalizó el entrenamiento. Modelo guardado en td_agent.pkl")

if __name__ == "__main__":
    train_agent(episodes=1000)