#Universida del Valle
#Proyecto Final


#TD-learning

import numpy as np
import random
import pickle
from battleshipENV import BattleshipEnv, BOARD_SIZE


class TDAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.997):
        self.alpha         = alpha          # Tasa de aprendizaje
        self.gamma         = gamma          # Factor de descuento
        self.epsilon       = epsilon        # Exploración inicial (decae por episodio)
        self.epsilon_min   = epsilon_min    # Piso de exploración
        self.epsilon_decay = epsilon_decay  # Multiplicador por episodio

        # v_table[r][c]: valor aprendido de disparar a la celda (r, c)
        self.v_table     = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.last_action = None

    def place_ships(self, env, player_id):
        env.place_ships_randomly(player_id)

    def select_action(self, obs, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(valid_actions)
        else:
            q_values = [self.v_table[r][c]
                        for act in valid_actions
                        for r, c in [divmod(act, BOARD_SIZE)]]
            action = valid_actions[int(np.argmax(q_values))]

        self.last_action = action
        return action

    def update(self, reward, next_obs, done):
        if self.last_action is None:
            return

        r_curr, c_curr = divmod(self.last_action, BOARD_SIZE)
        v_curr         = self.v_table[r_curr][c_curr]

        if done:
            target = reward
        else:
            # Solo el máximo sobre las acciones válidas del siguiente estado,
            valid_next = next_obs["valid_actions"]
            best_next  = max(
                (self.v_table[r][c]
                 for act in valid_next
                 for r, c in [divmod(act, BOARD_SIZE)]),
                default=0.0
            )
            target = reward + self.gamma * best_next

        self.v_table[r_curr][c_curr] += self.alpha * (target - v_curr)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO SELF-PLAY
# ══════════════════════════════════════════════════════════════════════════════

def train_agent(
    episodes:              int   = 1000,
    update_clone_every:    int   = 300,
    random_opponent_ratio: float = 0.2,
    log_every:             int   = 100,
    save_prefix:           str   = "td_agent",
):
    """
    Entrena el TDAgent con self-play + fracción de partidas contra agente aleatorio.

    Protocolo (simétrico al de RN.py para comparación justa):
      · Con probabilidad (1 - random_opponent_ratio): el oponente es un clone
        congelado del agente, actualizado cada 'update_clone_every' episodios.
      · Con probabilidad random_opponent_ratio: el oponente es completamente aleatorio.
      · Solo el agente (jugador 0) actualiza su v_table.
      · epsilon decae al final de cada episodio.

    Parámetros
    ----------
    episodes              : número de partidas — hiperparámetro principal
    update_clone_every    : cada cuántos eps sincronizar el clone
    random_opponent_ratio : fracción de eps contra agente aleatorio (0.2 = 20%)
    log_every             : frecuencia de impresión y guardado
    save_prefix           : prefijo del archivo .pkl de salida
    """
    from battleshipENV import RandomAgent

    env          = BattleshipEnv()
    agent        = TDAgent()
    random_agent = RandomAgent("Random-opponent")

    # Clone congelado para self-play (se actualiza periódicamente)
    clone         = TDAgent(epsilon=0.0)   # greedy puro, sin exploración
    clone.v_table = agent.v_table.copy()

    print(f"Entrenando TD-Agent por {episodes} episodios...")
    print(f"  Oponente aleatorio: {random_opponent_ratio:.0%} de episodios")
    print(f"  Clone se actualiza cada: {update_clone_every} eps\n")

    wins_window = 0

    for i in range(episodes):

        env.reset()

        # Elegir oponente del episodio
        use_random = random.random() < random_opponent_ratio
        opponent   = random_agent if use_random else clone

        # Fase de colocación
        agent.place_ships(env, 0)
        opponent.place_ships(env, 1)

        if env.phase != "battle":
            continue

        # Fase de batalla
        winner = -1
        while not env.done:
            current_player = env.current_player

            if current_player == 0:
                obs    = env.get_observation(0)
                action = agent.select_action(obs, obs["valid_actions"])

                _, _, reward, done, info = env.step(action)

                next_obs = env.get_observation(0)
                agent.update(reward, next_obs, done)

                if done:
                    winner = info.get("winner", -1)
            else:
                obs    = env.get_observation(1)
                action = clone.select_action(obs, obs["valid_actions"]) \
                         if not use_random \
                         else random.choice(env.get_valid_actions(1))
                _, _, _, done, info = env.step(action)
                if done:
                    winner = info.get("winner", -1)

        # Decaer epsilon al final de cada episodio
        agent.decay_epsilon()

        if winner == 0:
            wins_window += 1

        # Actualizar clone
        if (i + 1) % update_clone_every == 0:
            clone.v_table = agent.v_table.copy()

        # Log y guardado
        if (i + 1) % log_every == 0:
            win_rate = wins_window / log_every
            print(f"  Episodio {i+1:>5}/{episodes} │ "
                  f"WinRate: {win_rate:.1%} │ "
                  f"ε: {agent.epsilon:.4f}")

            with open(f"{save_prefix}.pkl", "wb") as f:
                pickle.dump(agent.v_table, f)

            wins_window = 0

    print(f"\nEntrenamiento finalizado. Modelo guardado en {save_prefix}.pkl")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA — ejecutar con: python TD_Learning.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from battleshipENV import RandomAgent

    agent = train_agent(
        episodes              = 800,        ##mantener en 800, para que sea "justo" con las "vueltas" que hace la RN
        update_clone_every    = 300,
        random_opponent_ratio = 0.2,
        log_every             = 100,
    )

    # Evaluación vs. agente aleatorio
    print("\nEvaluando vs. Agente Aleatorio (n partidas)...")
    env          = BattleshipEnv()
    random_agent = RandomAgent("Random")

    agent.epsilon = 0.0   # greedy puro para evaluación

    wins, total = 0, 600                #misma cantidad de partidas contra el random que la RN
    for _ in range(total):
        w, _, _ = env.run_game(agent, random_agent, render=False)
        if w == 0:
            wins += 1

    print(f"\n  Victoria: {wins}/{total} ({wins/total:.1%}) vs. agente aleatorio")
    print("  (>55% indica aprendizaje útil)")