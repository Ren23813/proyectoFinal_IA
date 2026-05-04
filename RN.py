"""
RN.py
=====
Agente de Red Neuronal para Battleship 7x7.
Implementado desde cero con NumPy puro (sin PyTorch u otra librería).

Dos redes neuronales:
  · BattleNet    (98 → 128 → 64 → 49)  — Q-values para la fase de disparo
  · PlacementNet (51 → 64  → 32 → 49)  — política para la fase de colocación

Algoritmos de entrenamiento:
  · BattleNet:    DQN (Q-learning + replay buffer + target network)
  · PlacementNet: REINFORCE (policy gradient con retorno del episodio)

Cómo entrenar:
    from RN import NeuralNetAgent, self_play_train
    agent = NeuralNetAgent()
    agent = self_play_train(agent, n_episodes=5000)

Cómo usar en competencia (con battleshipENV.py):
    winner, steps, history = env.run_game(agent, otro_agente)

Cómo guardar y recargar pesos:
    agent.save_weights("mi_agente")
    agent.load_weights("mi_agente")
"""

import numpy as np
import random
from collections import deque
import time

from battleshipENV import (
    BattleshipEnv, BOARD_SIZE, SHIP_SIZES,
    HORIZONTAL, VERTICAL,
)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ACTIVACIÓN Y PÉRDIDA
# (Reutilizadas/adaptadas de labs anteriores de la clase)
# ══════════════════════════════════════════════════════════════════════════════

def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def relu_deriv(Z: np.ndarray) -> np.ndarray:
    """Derivada de ReLU respecto a la pre-activación Z."""
    return (Z > 0).astype(np.float32)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable (resta el máximo antes de exponenciar)."""
    Z = Z - np.max(Z, axis=-1, keepdims=True)
    e = np.exp(Z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-8)

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# RED NEURONAL GENÉRICA
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNet:
    """
    Red neuronal densa con capas ocultas ReLU y salida lineal.
    Soporta dos modos de entrenamiento:
      · backward_q  → gradiente descendente por MSE (para Q-learning)
      · backward_pg → REINFORCE / policy gradient (para colocación)

    Parámetros
    ----------
    layer_sizes : list[int]   e.g. [98, 128, 64, 49]
    lr          : float       tasa de aprendizaje (SGD)
    seed        : int | None  semilla para reproducibilidad
    """

    def __init__(self, layer_sizes: list, lr: float = 0.001, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.lr          = lr
        self.n_layers    = len(layer_sizes) - 1   # número de matrices de pesos

        # Inicialización He (recomendada para ReLU)
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            std = np.sqrt(2.0 / layer_sizes[i])
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]).astype(np.float32) * std)
            self.b.append(np.zeros((1, layer_sizes[i+1]), dtype=np.float32))

    # ── Propagación hacia adelante ────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> tuple:
        """
        Retorna (A_list, Z_list):
          A_list[0]  = X (entrada)
          A_list[i]  = activación ReLU de la capa i  (capas ocultas)
          A_list[-1] = salida lineal (sin activación final)
          Z_list[i]  = pre-activación de la capa i (necesario para backward)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X.astype(np.float32)

        A_list = [X]
        Z_list = []

        for i in range(self.n_layers):
            Z = A_list[-1].dot(self.W[i]) + self.b[i]
            Z_list.append(Z)
            # Todas las capas ocultas usan ReLU; la salida es lineal
            A_list.append(relu(Z) if i < self.n_layers - 1 else Z)

        return A_list, Z_list

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante; retorna solo la salida."""
        A_list, _ = self.forward(X)
        return A_list[-1]

    # ── Backward: Q-learning (MSE) ────────────────────────────────────────────

    def backward_q(self, X: np.ndarray, q_targets: np.ndarray) -> float:
        """
        Actualiza los pesos minimizando el error cuadrático medio entre
        los Q-values predichos y los Q-targets de Bellman.

        El truco de DQN: q_targets[i] == q_pred[i] para todas las acciones
        excepto la acción tomada, de modo que el gradiente solo "empuja"
        la acción relevante.

        Retorna la pérdida escalar.
        """
        A_list, Z_list = self.forward(X)
        out  = A_list[-1]                                   # (B, 49)
        loss = mse_loss(out, q_targets)

        # δ de la capa de salida: d(MSE)/d(out) con salida lineal
        delta = 2.0 * (out - q_targets) / X.shape[0]       # (B, 49)

        # Retropropagación capa a capa (de salida hacia entrada)
        for i in reversed(range(self.n_layers)):
            dW = A_list[i].T.dot(delta)
            db = np.sum(delta, axis=0, keepdims=True)

            if i > 0:   # no hay que propagar más allá de la primera capa
                delta = delta.dot(self.W[i].T) * relu_deriv(Z_list[i-1])

            # Gradient descent
            self.W[i] -= self.lr * dW
            self.b[i]  -= self.lr * db

        return loss

    # ── Backward: REINFORCE (policy gradient) ────────────────────────────────

    def backward_pg(self, states: np.ndarray, actions: np.ndarray,
                    returns: np.ndarray) -> None:
        """
        Actualización REINFORCE:
          θ ← θ + α * G * ∇_θ log π(a|s)

        La red produce logits → softmax → π(a|s).
        Gradiente de log π(a|s) respecto a logits:
          ∂ log π(a|s) / ∂ logits_i = δ(i==a) - π(i|s)

        Parámetros
        ----------
        states  : (T, input_size)  estados al momento de colocar cada barco
        actions : (T,)             índices de celda elegidos [0..48]
        returns : (T,)             retorno G del episodio (+ ganó, - perdió)
        """
        T = len(states)
        if T == 0:
            return

        A_list, Z_list = self.forward(states)
        logits = A_list[-1]                     # (T, 49)
        probs  = softmax(logits)                # (T, 49)

        # Gradiente de la log-probabilidad de la acción tomada
        # δ = π(a|s) − 1[a==acción tomada], escalado por −G (ascenso de gradiente)
        delta = probs.copy()
        delta[np.arange(T), actions] -= 1.0
        delta *= (-returns.reshape(-1, 1))      # queremos maximizar → negamos para SGD
        delta /= T                              # promedio del batch

        for i in reversed(range(self.n_layers)):
            dW = A_list[i].T.dot(delta)
            db = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = delta.dot(self.W[i].T) * relu_deriv(Z_list[i-1])

            self.W[i] -= self.lr * dW
            self.b[i]  -= self.lr * db

    # ── Utilidades ────────────────────────────────────────────────────────────

    def copy_weights_from(self, other: "NeuralNet") -> None:
        """Copia todos los pesos desde otra red (para target network)."""
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def get_weights(self) -> tuple:
        return ([w.copy() for w in self.W], [b.copy() for b in self.b])

    def set_weights(self, weights: tuple) -> None:
        self.W = [w.copy() for w in weights[0]]
        self.b = [b.copy() for b in weights[1]]


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Buffer circular de experiencia para DQN.
    Almacena transiciones (s, a, r, s', done, acciones_válidas_siguiente).
    """

    def __init__(self, capacity: int = 15_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, valid_next: list) -> None:
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
            list(valid_next),
        ))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE DE RED NEURONAL
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetAgent:
    """
    Agente de Battleship que combina dos redes neuronales:

      BattleNet    — aprende Q(s, a) para decidir dónde disparar (DQN)
      PlacementNet — aprende π(posición | tablero, barco) para colocar (REINFORCE)

    Implementa la interfaz requerida por BattleshipEnv.run_game():
      · place_ships(env, player_id)
      · select_action(obs, valid_actions)

    Hiperparámetros
    ---------------
    gamma              : factor de descuento γ para Q-learning
    epsilon_start      : exploración inicial (ε-greedy)
    epsilon_min        : piso de exploración
    epsilon_decay      : multiplicador de decaimiento por episodio
    lr_battle          : tasa de aprendizaje de BattleNet
    lr_placement       : tasa de aprendizaje de PlacementNet
    batch_size         : tamaño del minibatch para DQN
    buffer_capacity    : capacidad del replay buffer
    target_update_freq : cada cuántos pasos copiar battle_net → target_net
    hidden_battle      : lista de neuronas por capa oculta en BattleNet
    hidden_placement   : lista de neuronas por capa oculta en PlacementNet
    name               : identificador del agente (para logs)
    seed               : semilla aleatoria
    """

    def __init__(
        self,
        gamma:              float = 0.95,
        epsilon_start:      float = 1.0,
        epsilon_min:        float = 0.05,
        epsilon_decay:      float = 0.995,
        lr_battle:          float = 0.001,
        lr_placement:       float = 0.005,
        batch_size:         int   = 64,
        buffer_capacity:    int   = 15_000,
        target_update_freq: int   = 200,
        hidden_battle:      list  = None,
        hidden_placement:   list  = None,
        name:               str   = "NeuralNetAgent",
        seed:               int   = None,
    ):
        self.name               = name
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._step_count        = 0
        self.training           = True   # False durante evaluación/competencia

        # ── Arquitecturas ────────────────────────────────────────────────────
        # BattleNet:    [tracking(49) + own(49)] → ocultas → 49 Q-values
        # PlacementNet: [own_grid(49) + ship_size(1) + orientation(1)] → 49 logits
        n_cells   = BOARD_SIZE ** 2                          # 49

        hb = hidden_battle    or [128, 64]
        hp = hidden_placement or [64, 32]

        battle_layers    = [n_cells * 2] + hb + [n_cells]   # 98 → ... → 49
        placement_layers = [n_cells + 2] + hp + [n_cells]   # 51 → ... → 49

        self.battle_net    = NeuralNet(battle_layers,    lr=lr_battle,    seed=seed)
        self.target_net    = NeuralNet(battle_layers,    lr=lr_battle,    seed=seed)
        self.placement_net = NeuralNet(placement_layers, lr=lr_placement, seed=seed)
        self.target_net.copy_weights_from(self.battle_net)

        self.replay = ReplayBuffer(buffer_capacity)

        # Log interno de colocación para REINFORCE (se limpia cada episodio)
        self._placement_states:  list = []
        self._placement_actions: list = []

    # ── Interfaz BattleshipEnv: colocar barcos ────────────────────────────────

    def place_ships(self, env: BattleshipEnv, player_id: int) -> None:
        """
        Coloca todos los barcos pendientes usando PlacementNet.

        Para cada barco:
          1. Construye el vector de estado: [own_grid_flat | ship_size | orientation]
          2. Obtiene logits de PlacementNet → filtra solo posiciones válidas
          3. Durante entrenamiento: muestrea según softmax (exploración)
             Durante evaluación:   elige la posición con mayor logit (greedy)
          4. Prueba ambas orientaciones y elige la que tiene mayor logit máximo

        Guarda (estado, acción) para el update REINFORCE al final del episodio.
        """
        self._placement_states  = []
        self._placement_actions = []

        n_cells = BOARD_SIZE ** 2

        while not env.placement_done(player_id):
            size = env.next_ship_size(player_id)
            own  = env.boards[player_id].own_grid.flatten() / 2.0   # (49,) normalizado

            best_pos     = None
            best_orient  = None
            best_action  = None
            best_logit   = -np.inf
            best_state   = None

            # Evaluar ambas orientaciones y quedarse con la mejor
            for orientation in [HORIZONTAL, VERTICAL]:
                positions = env.get_valid_placements(player_id, orientation)
                if not positions:
                    continue

                # Vector de estado para PlacementNet
                state_vec = np.concatenate([
                    own,
                    [size / max(SHIP_SIZES)],          # normalizado
                    [float(orientation)],
                ]).astype(np.float32)                  # (51,)

                logits    = self.placement_net.predict(state_vec).flatten()   # (49,)
                valid_idx = [r * BOARD_SIZE + c for r, c in positions]

                # Logits solo de posiciones válidas
                valid_logits = logits[valid_idx]

                # Elegir posición
                if self.training and random.random() < max(self.epsilon, 0.2):
                    # Exploración: muestreo según softmax
                    probs        = softmax(valid_logits)
                    chosen_local = np.random.choice(len(valid_idx), p=probs)
                else:
                    # Explotación: mayor logit
                    chosen_local = int(np.argmax(valid_logits))

                top_logit = valid_logits[chosen_local]
                if top_logit > best_logit:
                    best_logit  = top_logit
                    best_pos    = positions[chosen_local]
                    best_orient = orientation
                    best_action = valid_idx[chosen_local]    # índice de celda [0..48]
                    best_state  = state_vec

            # Fallback: si PlacementNet no encontró ninguna posición, colocar aleatoriamente
            if best_pos is None:
                env.place_ships_randomly(player_id)
                return

            row, col = best_pos
            success  = env.place_ship(player_id, row, col, best_orient)

            if not success:
                # Posición inválida en el momento de colocar (race condition rara)
                env.place_ships_randomly(player_id)
                return

            if self.training:
                self._placement_states.append(best_state)
                self._placement_actions.append(best_action)

    # ── Interfaz BattleshipEnv: seleccionar disparo ───────────────────────────

    def select_action(self, obs: dict, valid_actions: list) -> int:
        """
        Elige una celda para disparar usando política ε-greedy sobre BattleNet.

        Durante entrenamiento: con probabilidad ε elige aleatoriamente,
                               con probabilidad 1-ε elige el mayor Q-value.
        Durante evaluación:    siempre elige el mayor Q-value (greedy puro).
        """
        if not valid_actions:
            raise ValueError("No hay acciones válidas.")

        if self.training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state  = obs["flat"]                                     # (98,)
        q_vals = self.battle_net.predict(state).flatten()        # (49,)

        # Enmascarar celdas inválidas con -inf para que nunca sean elegidas
        masked = np.full(BOARD_SIZE ** 2, -np.inf)
        masked[valid_actions] = q_vals[valid_actions]
        return int(np.argmax(masked))

    # ── Almacenar transición ──────────────────────────────────────────────────

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool,
                         valid_next: list) -> None:
        """Guarda una transición de batalla en el replay buffer."""
        self.replay.push(state, action, reward, next_state, done, valid_next)

    # ── Paso de entrenamiento DQN ─────────────────────────────────────────────

    def train_battle_step(self) -> float | None:
        """
        Muestrea un minibatch del replay buffer y realiza un paso de gradiente
        en BattleNet usando la ecuación de Bellman:

            Q_target(s, a) = r + γ · max_{a'} Q_target(s', a')

        La target_net (copia congelada de battle_net) se usa para calcular
        Q(s', a') con mayor estabilidad (técnica DQN clásica).

        Retorna la pérdida del minibatch, o None si el buffer está muy vacío.
        """
        if len(self.replay) < self.batch_size:
            return None

        batch       = self.replay.sample(self.batch_size)
        states      = np.vstack([t[0] for t in batch])             # (B, 98)
        actions     = np.array( [t[1] for t in batch], dtype=int)  # (B,)
        rewards     = np.array( [t[2] for t in batch])             # (B,)
        next_states = np.vstack([t[3] for t in batch])             # (B, 98)
        dones       = np.array( [t[4] for t in batch])             # (B,) bool
        valid_nexts = [t[5] for t in batch]                        # lista de listas

        # Q-values actuales (red principal)
        q_pred   = self.battle_net.predict(states)                  # (B, 49)
        # Q-values del siguiente estado (target network, sin gradiente)
        q_next   = self.target_net.predict(next_states)             # (B, 49)

        # Construir targets: copiar Q actuales y modificar SOLO la acción tomada
        # (el gradiente no "toca" las otras acciones → equivalente a ignorarlas)
        q_target = q_pred.copy()

        for i in range(self.batch_size):
            if dones[i]:
                # Estado terminal: sin valor futuro
                q_target[i, actions[i]] = rewards[i]
            else:
                valid = valid_nexts[i]
                if valid:
                    best_next = float(np.max(q_next[i, valid]))
                else:
                    best_next = 0.0
                q_target[i, actions[i]] = rewards[i] + self.gamma * best_next

        loss = self.battle_net.backward_q(states, q_target)

        # Actualizar target_net cada N pasos
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.battle_net)

        return loss

    # ── Paso de entrenamiento REINFORCE (colocación) ──────────────────────────

    def train_placement_step(self, episode_return: float) -> None:
        """
        Actualiza PlacementNet con REINFORCE al final de cada episodio.

        El retorno del episodio G se usa para todas las decisiones de colocación:
          · G > 0 → se refuerzan las posiciones elegidas (ganamos, buena colocación)
          · G < 0 → se penalizan las posiciones elegidas (perdimos, mala colocación)

        La señal es diferida (sparse reward), lo cual es la motivación de REINFORCE:
        el gradiente solo se aplica al terminar el episodio, usando el retorno global.

        Parámetro
        ---------
        episode_return : float — típicamente +1.0 si ganó, -1.0 si perdió
        """
        if not self._placement_states:
            return

        states  = np.vstack(self._placement_states)                         # (T, 51)
        actions = np.array(self._placement_actions, dtype=int)              # (T,)
        returns = np.full(len(actions), episode_return, dtype=np.float32)   # (T,)

        self.placement_net.backward_pg(states, actions, returns)

        # Limpiar log del episodio
        self._placement_states  = []
        self._placement_actions = []

    # ── Decaer epsilon ────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Reduce ε multiplicativamente, respetando el piso epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Guardar y cargar pesos ────────────────────────────────────────────────

    def save_weights(self, path_prefix: str = "rn_agent") -> None:
        """
        Guarda los pesos de BattleNet y PlacementNet en archivos .npz.
        Ejemplo: save_weights("mi_agente") → mi_agente_battle.npz, mi_agente_placement.npz
        """
        n_b = self.battle_net.n_layers
        n_p = self.placement_net.n_layers

        np.savez(f"{path_prefix}_battle.npz",
                 **{f"W{i}": self.battle_net.W[i] for i in range(n_b)},
                 **{f"b{i}": self.battle_net.b[i] for i in range(n_b)})

        np.savez(f"{path_prefix}_placement.npz",
                 **{f"W{i}": self.placement_net.W[i] for i in range(n_p)},
                 **{f"b{i}": self.placement_net.b[i] for i in range(n_p)})

        print(f"Pesos guardados -> {path_prefix}_battle.npz  |  {path_prefix}_placement.npz")

    def load_weights(self, path_prefix: str = "rn_agent") -> None:
        """
        Carga pesos previamente guardados con save_weights().
        Sincroniza target_net con battle_net automáticamente.
        """
        def _load(net, path):
            data = np.load(path)
            n    = net.n_layers
            net.W = [data[f"W{i}"] for i in range(n)]
            net.b = [data[f"b{i}"] for i in range(n)]

        _load(self.battle_net,    f"{path_prefix}_battle.npz")
        _load(self.placement_net, f"{path_prefix}_placement.npz")
        self.target_net.copy_weights_from(self.battle_net)
        print(f"Pesos cargados ← {path_prefix}_battle.npz  |  {path_prefix}_placement.npz")

    def __repr__(self):
        return (f"NeuralNetAgent(name='{self.name}', "
                f"ε={self.epsilon:.3f}, grad_steps={self._step_count})")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO SELF-PLAY
# ══════════════════════════════════════════════════════════════════════════════

def self_play_train(
    agent:                  NeuralNetAgent,
    n_episodes:             int   = 5000,
    update_opponent_every:  int   = 300,
    random_opponent_ratio:  float = 0.2,
    log_every:              int   = 200,
    max_steps_per_game:     int   = 300,
    placement_return_scale: float = 1.0,
    save_best:              bool  = True,
    save_prefix:            str   = "rn_agent",
) -> NeuralNetAgent:
    """
    Entrena el agente jugando contra una copia congelada de sí mismo (self-play),
    con una fracción de episodios contra un agente aleatorio para mejorar la
    generalización fuera del self-play.

    Protocolo
    ---------
    · En cada episodio, con probabilidad (1 - random_opponent_ratio) el oponente
      es la copia congelada del agente; con probabilidad random_opponent_ratio
      el oponente es un agente completamente aleatorio.
    · Mezclar ambos tipos de oponente evita que el agente solo aprenda a explotar
      su propio estilo y mejora el desempeño contra estrategias desconocidas.
    · El oponente self-play se actualiza cada 'update_opponent_every' episodios.

    Hiperparámetros
    ---------------
    n_episodes             : número total de partidas — el hiperparámetro principal
    update_opponent_every  : cada cuántos episodios sincronizar la copia self-play
    random_opponent_ratio  : fracción de episodios contra agente aleatorio [0.0-1.0]
                             0.0 = solo self-play puro
                             0.2 = 20% aleatorio + 80% self-play (recomendado)
                             1.0 = solo contra aleatorio (sin self-play)
    log_every              : frecuencia de impresión de estadísticas
    max_steps_per_game     : límite de disparos por partida (evita loops infinitos)
    placement_return_scale : escala del retorno G para REINFORCE de colocación
    save_best              : guardar pesos cuando se supera el mejor win rate
    save_prefix            : prefijo de archivo para save_weights()

    Retorna
    -------
    El agente entrenado (mismo objeto, modificado in-place).
    """
    from battleshipENV import RandomAgent

    env           = BattleshipEnv()
    random_agent  = RandomAgent("Random-opponent")

    # Copia congelada del agente para self-play
    clone = NeuralNetAgent(name="Clone (self-play)")
    clone.training = False
    clone.epsilon  = 0.0
    clone.battle_net.copy_weights_from(agent.battle_net)
    clone.placement_net.copy_weights_from(agent.placement_net)

    best_win_rate = 0.0    # se actualiza siempre, independiente de save_best
    wins_window   = 0
    losses_window = []

    print(f"\n{'═'*62}")
    print(f"  Entrenamiento self-play — Red Neuronal")
    print(f"  Episodios:            {n_episodes}")
    print(f"  Oponente aleatorio:   {random_opponent_ratio:.0%} de episodios")
    print(f"  Clone se actualiza cada: {update_opponent_every} eps")
    print(f"  Arquitectura BattleNet:    {agent.battle_net.layer_sizes}")
    print(f"  Arquitectura PlacementNet: {agent.placement_net.layer_sizes}")
    print(f"{'═'*62}\n")

    for episode in range(1, n_episodes + 1):

        env.reset()

        # Elegir oponente para este episodio
        use_random = random.random() < random_opponent_ratio
        opponent   = random_agent if use_random else clone

        # ── Fase de colocación ───────────────────────────────────────────────
        agent.place_ships(env, 0)
        opponent.place_ships(env, 1)

        if env.phase != "battle":
            continue

        # ── Fase de batalla ──────────────────────────────────────────────────
        done   = False
        steps  = 0
        winner = -1

        while not done and steps < max_steps_per_game:
            current  = env.current_player
            is_agent = (current == 0)
            active   = agent if is_agent else opponent

            obs   = env.get_observation(current)
            valid = obs["valid_actions"]
            if not valid:
                break

            action                       = active.select_action(obs, valid)
            obs_s, _, reward, done, info = env.step(action)

            # Solo el agente (jugador 0) almacena experiencia y entrena
            if is_agent:
                next_obs   = env.get_observation(0)
                next_valid = next_obs["valid_actions"]

                agent.store_transition(
                    obs["flat"], action, reward,
                    next_obs["flat"], done, next_valid,
                )
                loss = agent.train_battle_step()
                if loss is not None:
                    losses_window.append(loss)

            if done:
                winner = info.get("winner", -1)

            steps += 1

        # ── Actualizar PlacementNet con REINFORCE ────────────────────────────
        if winner == 0:
            g_return = +1.0 * placement_return_scale
            wins_window += 1
        elif winner == 1:
            g_return = -1.0 * placement_return_scale
        else:
            g_return = 0.0

        agent.train_placement_step(g_return)

        # ── Decaer epsilon ───────────────────────────────────────────────────
        agent.decay_epsilon()

        # ── Actualizar clone ─────────────────────────────────────────────────
        if episode % update_opponent_every == 0:
            clone.battle_net.copy_weights_from(agent.battle_net)
            clone.placement_net.copy_weights_from(agent.placement_net)

        # ── Log y guardado ───────────────────────────────────────────────────
        if episode % log_every == 0:
            win_rate = wins_window / log_every
            avg_loss = float(np.mean(losses_window)) if losses_window else float("nan")

            print(
                f"  Ep {episode:>5}/{n_episodes} │ "
                f"WinRate: {win_rate:.1%} │ "
                f"AvgLoss: {avg_loss:.5f} │ "
                f"ε: {agent.epsilon:.4f} │ "
                f"Buffer: {len(agent.replay)}"
            )

            # Trackear mejor win rate siempre (independiente de save_best)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                if save_best:
                    agent.save_weights(save_prefix + "_best")
                    print(f"    ↑ Nuevo mejor win rate — pesos guardados.")

            wins_window   = 0
            losses_window = []

    print(f"\n{'═'*62}")
    print(f"  Entrenamiento finalizado.")
    print(f"  Pasos de gradiente totales: {agent._step_count}")
    print(f"  Mejor win rate observado:   {best_win_rate:.1%}")
    print(f"{'═'*62}\n")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA — ejecutar con: python RN.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    inicio = time.time()
    from battleshipENV import RandomAgent

    # ── Hiperparámetros (cambiar para la pelea final) ─────────────────────
    # Para resultados significativos usar n_episodes >= 5000, pero tarda bastante.
    HPARAMS = dict(
        n_episodes            = 3000,   # cantidad de entrenamientos
        update_opponent_every = 300,
        random_opponent_ratio = 0.2,    # 20% de partidas contra agente aleatorio
        log_every             = 300,
        placement_return_scale= 1.0,
        save_best             = False,  # True para guardar el mejor modelo en disco
    )

    print("=" * 62)
    print("  TEST RN.py")
    print("=" * 62)

    agent = NeuralNetAgent(
        gamma             = 0.95,
        epsilon_start     = 1.0,
        epsilon_min       = 0.05,
        epsilon_decay     = 0.997,
        lr_battle         = 0.001,
        lr_placement      = 0.005,
        batch_size        = 64,
        hidden_battle     = [128, 64],
        hidden_placement  = [64, 32],
        name              = "RN_agent",
        seed              = 42,
    )

    agent = self_play_train(agent, **HPARAMS)

    # ── Evaluación final contra agente aleatorio ──────────────────────────────
    print("Evaluando vs. Agente Aleatorio (200 partidas)...")
    env          = BattleshipEnv()
    random_agent = RandomAgent("Random")

    agent.training = False
    agent.epsilon  = 0.0

    wins, total = 0, 600
    for _ in range(total):
        w, _, _ = env.run_game(agent, random_agent, render=False)
        if w == 0:
            wins += 1

    print(f"\n  Victoria: {wins}/{total} ({wins/total:.1%}) vs. agente aleatorio")
    print("  (>55% indica aprendizaje útil, >65% es muy bueno para 3000 eps)")
    fin = time.time()
    print("Tardó en ejecutar:", fin-inicio)
