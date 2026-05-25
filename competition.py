"""
competition.py
==============
Archivo principal de la competencia final: TD-Learning vs Red Neuronal en Battleship 7×7.

Funcionalidades
───────────────
  · Entrenamiento de ambos modelos con recolección de métricas
  · Guardado / carga automática (evita reentrenar si los archivos ya existen)
  · Competencia directa: N partidas TD vs RN con turnos alternos
  · Dashboard de visualizaciones:
        1. Métricas individuales de entrenamiento (win rate, reward/loss, epsilon)
        2. Heatmaps de disparos y colocación de barcos (entrenamiento + competencia)
        3. Win rate acumulado, pie chart y distribución de turnos
        4. Evolución del reward durante la competencia
        5. Estrategias emergentes (v-table TD / Q-values iniciales RN)
        6. Replay visual de una partida ejemplo
        7. Comparación de desempeño vs agente aleatorio

Uso rápido
──────────
    python competition.py               ← entrena si no hay checkpoints, luego compite
    python competition.py --retrain     ← fuerza reentrenamiento aunque existan archivos
    python competition.py --no-train    ← sólo carga y compite (debe haber checkpoints)


"""

# ── std-lib ────────────────────────────────────────────────────────────────
import os
import sys
import pickle
import random
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── third-party ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin ventana gráfica; cambiar a "TkAgg" si quieres pop-ups
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── proyecto ───────────────────────────────────────────────────────────────
from battleshipENV import (
    BattleshipEnv, BOARD_SIZE, SHIP_SIZES, NUM_SHIPS,
    HORIZONTAL, VERTICAL, RandomAgent,
    OWN_EMPTY, OWN_SHIP, OWN_HIT,
    TRACK_UNKNOWN, TRACK_MISS, TRACK_HIT, TRACK_SUNK,
)
from TDLearning import TDAgent
from RN import NeuralNetAgent, self_play_train


# RUTAS DE GUARDADO
TD_SAVE_PATH   = "td_agent_vtable.pkl"
RN_SAVE_PREFIX = "rn_agent"

# GUARDAR / CARGAR AGENTES

def save_td_agent(agent: TDAgent, path: str = TD_SAVE_PATH) -> None:
    """Serializa v_table + epsilon del TDAgent en disco."""
    with open(path, "wb") as f:
        pickle.dump({"v_table": agent.v_table, "epsilon": agent.epsilon}, f)
    print(f"  [TD]  Guardado → {path}")


def load_td_agent(path: str = TD_SAVE_PATH) -> TDAgent:
    """Restaura un TDAgent desde disco y lo pone en modo evaluación (ε=0)."""
    agent = TDAgent(epsilon=0.0)
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Compatibilidad con el formato simple que genera TD_Learning.py (solo el array)
    if isinstance(data, dict):
        agent.v_table = data["v_table"]
        agent.epsilon = float(data.get("epsilon", 0.0))
    else:
        agent.v_table = data
    print(f"  [TD]  Cargado ← {path}")
    return agent


def save_rn_agent(agent: NeuralNetAgent, prefix: str = RN_SAVE_PREFIX) -> None:
    """Delega en el método propio de NeuralNetAgent."""
    agent.save_weights(prefix)


def load_rn_agent(prefix: str = RN_SAVE_PREFIX) -> NeuralNetAgent:
    """Restaura un NeuralNetAgent desde disco y lo pone en modo evaluación."""
    agent = NeuralNetAgent(name="RN_agent")
    agent.load_weights(prefix)
    agent.training = False
    agent.epsilon  = 0.0
    return agent


def _rn_files_exist(prefix: str = RN_SAVE_PREFIX) -> bool:
    return (os.path.exists(f"{prefix}_battle.npz") and
            os.path.exists(f"{prefix}_placement.npz"))


# ENTRENAMIENTO CON RECOLECCIÓN DE MÉTRICAS

def train_td_with_metrics(
    episodes:              int   = 800,
    update_clone_every:    int   = 300,
    random_opponent_ratio: float = 0.2,
    log_every:             int   = 100,
    save_path:             str   = TD_SAVE_PATH,
) -> tuple:
    """
    Re-implementa el loop de TD_Learning.train_agent() añadiendo recolección
    de métricas (win rate, reward promedio, epsilon) y heatmaps de disparos
    y colocación durante el entrenamiento.

    Retorna
    -------
    (agent, history_log, shot_heatmap, placement_heatmap)
      history_log : lista de dicts {episode, win_rate, avg_reward, epsilon}
      shot_heatmap      : np.array (7,7) — frecuencia de disparos acumulada
      placement_heatmap : np.array (7,7) — frecuencia de colocación acumulada
    """
    env          = BattleshipEnv()
    agent        = TDAgent()
    random_agent = RandomAgent("Random-opponent")

    # Clone congelado para self-play
    clone         = TDAgent(epsilon=0.0)
    clone.v_table = agent.v_table.copy()

    history_log       = []
    wins_window       = 0
    rewards_window    = []
    shot_heatmap      = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    placement_heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)

    print(f"\n{'─'*58}")
    print(f"  Entrenando TD-Agent  ({episodes} episodios)")
    print(f"  Oponente aleatorio: {random_opponent_ratio:.0%}  │  "
          f"Clone actualiza cada {update_clone_every} eps")
    print(f"{'─'*58}")

    for ep in range(episodes):
        env.reset()
        use_random = random.random() < random_opponent_ratio
        opponent   = random_agent if use_random else clone

        agent.place_ships(env, 0)
        opponent.place_ships(env, 1)

        # Acumular colocación del agente TD
        own_grid = env.boards[0].own_grid
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if own_grid[r, c] == OWN_SHIP:
                    placement_heatmap[r, c] += 1

        if env.phase != "battle":
            continue

        winner         = -1
        episode_reward = 0.0

        while not env.done:
            cp = env.current_player

            if cp == 0:
                obs    = env.get_observation(0)
                action = agent.select_action(obs, obs["valid_actions"])
                _, _, reward, done, info = env.step(action)

                r_s, c_s = divmod(action, BOARD_SIZE)
                shot_heatmap[r_s, c_s] += 1

                next_obs = env.get_observation(0)
                agent.update(reward, next_obs, done)
                episode_reward += reward

                if done:
                    winner = info.get("winner", -1)
            else:
                obs    = env.get_observation(1)
                if use_random:
                    valid = env.get_valid_actions(1)
                    action = random.choice(valid) if valid else 0
                else:
                    action = clone.select_action(obs, obs["valid_actions"])
                _, _, _, done, info = env.step(action)
                if done:
                    winner = info.get("winner", -1)

        agent.decay_epsilon()

        if winner == 0:
            wins_window += 1
        rewards_window.append(episode_reward)

        if (ep + 1) % update_clone_every == 0:
            clone.v_table = agent.v_table.copy()

        if (ep + 1) % log_every == 0:
            win_rate   = wins_window / log_every
            avg_reward = float(np.mean(rewards_window)) if rewards_window else 0.0
            print(f"  Ep {ep+1:>5}/{episodes} │ WinRate: {win_rate:.1%} │ "
                  f"AvgReward: {avg_reward:+.3f} │ ε: {agent.epsilon:.4f}")
            history_log.append({
                "episode":    ep + 1,
                "win_rate":   win_rate,
                "avg_reward": avg_reward,
                "epsilon":    agent.epsilon,
            })
            wins_window    = 0
            rewards_window = []

    save_td_agent(agent, save_path)
    print(f"  Entrenamiento TD finalizado.\n")
    return agent, history_log, shot_heatmap, placement_heatmap


def train_rn_with_metrics(
    n_episodes:             int   = 800,
    update_opponent_every:  int   = 300,
    random_opponent_ratio:  float = 0.2,
    log_every:              int   = 200,
    save_prefix:            str   = RN_SAVE_PREFIX,
) -> tuple:
    """
    Llama a self_play_train() de RN.py (que ya devuelve history_log) y
    recopila heatmaps adicionales jugando 200 partidas de evaluación.

    Retorna
    -------
    (agent, history_log, shot_heatmap, placement_heatmap)
    """
    print(f"\n{'─'*58}")
    print(f"  Entrenando RN-Agent  ({n_episodes} episodios)")
    print(f"  Oponente aleatorio: {random_opponent_ratio:.0%}  │  "
          f"Clone actualiza cada {update_opponent_every} eps")
    print(f"{'─'*58}")

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

    # self_play_train ya devuelve (agent, history_log)
    agent, history_log = self_play_train(
        agent,
        n_episodes            = n_episodes,
        update_opponent_every = update_opponent_every,
        random_opponent_ratio = random_opponent_ratio,
        log_every             = log_every,
        save_best             = False,
        save_prefix           = save_prefix,
    )

    # Modo evaluación para los heatmaps
    agent.training = False
    agent.epsilon  = 0.0

    # Recopilamos heatmaps jugando partidas de evaluación
    env               = BattleshipEnv()
    random_agent      = RandomAgent("Random")
    shot_heatmap      = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    placement_heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)

    print("  Recopilando heatmaps RN (200 partidas de evaluación)...")
    for _ in range(200):
        env.reset()
        agent.place_ships(env, 0)
        random_agent.place_ships(env, 1)

        own_grid = env.boards[0].own_grid
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if own_grid[r, c] == OWN_SHIP:
                    placement_heatmap[r, c] += 1

        if env.phase != "battle":
            continue

        while not env.done:
            cp  = env.current_player
            obs = env.get_observation(cp)
            if cp == 0:
                action = agent.select_action(obs, obs["valid_actions"])
                r_s, c_s = divmod(action, BOARD_SIZE)
                shot_heatmap[r_s, c_s] += 1
            else:
                valid = obs["valid_actions"]
                action = random.choice(valid) if valid else 0
            env.step(action)

    save_rn_agent(agent, save_prefix)
    print(f"  Entrenamiento RN finalizado.\n")
    return agent, history_log, shot_heatmap, placement_heatmap


# EVALUACIÓN VS AGENTE ALEATORIO

def evaluate_vs_random(agent, n_games: int = 300, label: str = "Agent") -> float:
    """
    Enfrenta al agente contra RandomAgent n_games veces.
    El agente siempre es jugador 0.  Retorna su win rate.
    """
    env    = BattleshipEnv()
    rand   = RandomAgent("Random")
    wins   = 0
    for _ in range(n_games):
        w, _, _ = env.run_game(agent, rand, render=False)
        if w == 0:
            wins += 1
    wr = wins / n_games
    print(f"  {label:20s}  vs Random: {wins:>4}/{n_games}  ({wr:.1%})")
    return wr

# COMPETENCIA DIRECTA

def run_competition(
    agent_td: TDAgent,
    agent_rn: NeuralNetAgent,
    n_games:  int = 1000,
) -> dict:
    """
    Enfrenta TD vs RN en n_games partidas con turnos alternos (par → TD es j0;
    impar → RN es j0) para eliminar la ventaja del primer turno.

    Acumula heatmaps de disparos y colocación, rewards por episodio,
    duración en turnos y el historial de victorias.

    Retorna un dict con todas las métricas recopiladas.
    """
    env = BattleshipEnv()
    agent_td.epsilon  = 0.0
    agent_rn.training = False
    agent_rn.epsilon  = 0.0

    wins_td      = 0
    wins_rn      = 0
    steps_list   = []
    rewards_td   = []
    rewards_rn   = []
    win_history  = []   # 1 = TD ganó | 0 = RN ganó | -1 = empate/timeout

    shot_hm_td   = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    shot_hm_rn   = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    place_hm_td  = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    place_hm_rn  = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)

    print(f"\n{'═'*58}")
    print(f"  COMPETENCIA: TD-Learning vs Red Neuronal  ({n_games} partidas)")
    print(f"{'═'*58}")

    for game_idx in range(n_games):
        env.reset()

        # Alternamos el orden para eliminar ventaja de primer turno
        if game_idx % 2 == 0:
            p_td, p_rn = 0, 1
        else:
            p_rn, p_td = 0, 1

        agent_td.place_ships(env, p_td)
        agent_rn.place_ships(env, p_rn)

        if env.phase != "battle":
            win_history.append(-1)
            continue

        # Heatmaps de colocación para este juego
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if env.boards[p_td].own_grid[r, c] == OWN_SHIP:
                    place_hm_td[r, c] += 1
                if env.boards[p_rn].own_grid[r, c] == OWN_SHIP:
                    place_hm_rn[r, c] += 1

        ep_reward_td = 0.0
        ep_reward_rn = 0.0
        steps        = 0
        winner_env   = -1

        while not env.done and steps < 300:
            cp    = env.current_player
            obs   = env.get_observation(cp)
            valid = obs["valid_actions"]
            if not valid:
                break

            if cp == p_td:
                action = agent_td.select_action(obs, valid)
                r_s, c_s = divmod(action, BOARD_SIZE)
                shot_hm_td[r_s, c_s] += 1
            else:
                action = agent_rn.select_action(obs, valid)
                r_s, c_s = divmod(action, BOARD_SIZE)
                shot_hm_rn[r_s, c_s] += 1

            _, _, reward, done, info = env.step(action)

            if cp == p_td:
                ep_reward_td += reward
            else:
                ep_reward_rn += reward

            if done:
                winner_env = info.get("winner", -1)

            steps += 1

        # Traducir jugador-ID ganador a nombre de agente
        if winner_env == p_td:
            wins_td += 1
            win_history.append(1)
        elif winner_env == p_rn:
            wins_rn += 1
            win_history.append(0)
        else:
            win_history.append(-1)

        steps_list.append(steps)
        rewards_td.append(ep_reward_td)
        rewards_rn.append(ep_reward_rn)

        # Log cada 100 partidas
        if (game_idx + 1) % 100 == 0:
            recent    = win_history[-100:]
            rec_td    = sum(1 for x in recent if x == 1)
            rec_rn    = sum(1 for x in recent if x == 0)
            total_so_far = game_idx + 1
            print(f"  Partida {total_so_far:>5}/{n_games} │ "
                  f"TD: {wins_td} ({wins_td/total_so_far:.1%}) │ "
                  f"RN: {wins_rn} ({wins_rn/total_so_far:.1%}) │ "
                  f"Últimas 100 → TD:{rec_td}  RN:{rec_rn}")

    print(f"\n  ── Resultado final ──────────────────────────────────")
    print(f"  TD-Learning ganó: {wins_td:>5}/{n_games}  ({wins_td/n_games:.1%})")
    print(f"  Red Neuronal ganó:{wins_rn:>5}/{n_games}  ({wins_rn/n_games:.1%})")
    draws = n_games - wins_td - wins_rn
    print(f"  Empates/Timeout:  {draws:>5}/{n_games}  ({draws/n_games:.1%})\n")

    return {
        "wins_td":     wins_td,
        "wins_rn":     wins_rn,
        "draws":       draws,
        "n_games":     n_games,
        "steps_list":  steps_list,
        "rewards_td":  rewards_td,
        "rewards_rn":  rewards_rn,
        "win_history": win_history,
        "shot_hm_td":  shot_hm_td,
        "shot_hm_rn":  shot_hm_rn,
        "place_hm_td": place_hm_td,
        "place_hm_rn": place_hm_rn,
    }

# CAPTURAR REPLAY DE UNA PARTIDA

def capture_replay(agent_td: TDAgent, agent_rn: NeuralNetAgent) -> tuple:
    """
    Juega una partida completa TD (j0) vs RN (j1) y guarda un snapshot
    del estado del tablero en cada turno para renderizar el replay.

    Retorna (frames, winner_int).
    Cada frame es un dict con grids copiados y metadata del turno.
    """
    env = BattleshipEnv()
    agent_td.epsilon  = 0.0
    agent_rn.training = False
    agent_rn.epsilon  = 0.0

    env.reset()
    agent_td.place_ships(env, 0)
    agent_rn.place_ships(env, 1)

    frames = []
    step   = 0

    while not env.done and step < 300:
        cp    = env.current_player
        obs   = env.get_observation(cp)
        valid = obs["valid_actions"]
        if not valid:
            break

        action = agent_td.select_action(obs, valid) if cp == 0 \
                 else agent_rn.select_action(obs, valid)

        # Snapshot ANTES del disparo
        frame = {
            "step":     step,
            "shooter":  cp,
            "action":   action,
            "own_td":   env.boards[0].own_grid.copy(),
            "track_td": env.boards[0].tracking_grid.copy(),
            "own_rn":   env.boards[1].own_grid.copy(),
            "track_rn": env.boards[1].tracking_grid.copy(),
        }

        _, _, reward, done, info = env.step(action)
        frame.update({
            "reward": reward,
            "hit":    info["hit"],
            "sunk":   info["sunk"],
            "done":   done,
        })
        frames.append(frame)
        step += 1

    return frames, env.winner


# PALETAS Y HELPER DE HEATMAP
CMAP_SHOTS = LinearSegmentedColormap.from_list(
    "shots", ["#eef4ff", "#0050d9"], N=256)
CMAP_PLACE = LinearSegmentedColormap.from_list(
    "place", ["#fff0ee", "#cc1100"], N=256)
CMAP_VTABLE = "coolwarm"

COLOR_TD = "#4a90d9"
COLOR_RN = "#e05c5c"


def _draw_heatmap(ax, data: np.ndarray, title: str, cmap,
                  vmin: float = None, vmax: float = None,
                  annot_fmt: str = "{:.0f}") -> None:
    """Dibuja un heatmap con anotaciones en cada celda."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_xticklabels(range(BOARD_SIZE), fontsize=7)
    ax.set_yticklabels(range(BOARD_SIZE), fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    dmax = data.max() if data.max() != 0 else 1
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = data[r, c]
            txt_color = "white" if val / dmax > 0.55 else "black"
            ax.text(c, r, annot_fmt.format(val),
                    ha="center", va="center", fontsize=6, color=txt_color)
    return im

# MÉTRICAS INDIVIDUALES DE ENTRENAMIENTO
def plot_training_metrics(td_history: list, rn_history: list,
                          save_path: str = "graficas/plot_training_metrics.png") -> None:
    """
    2 filas × 3 columnas:
      Fila TD:  Win Rate │ Reward promedio  │ Epsilon
      Fila RN:  Win Rate │ Loss MSE         │ Epsilon
    """
    if not td_history or not rn_history:
        print("  (Sin historial de entrenamiento — gráfica omitida)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Métricas de Entrenamiento Individual",
                 fontsize=14, fontweight="bold")

    td_eps = [h["episode"]    for h in td_history]
    td_wr  = [h["win_rate"]   for h in td_history]
    td_rw  = [h["avg_reward"] for h in td_history]
    td_ep  = [h["epsilon"]    for h in td_history]

    rn_eps = [h["episode"]    for h in rn_history]
    rn_wr  = [h["win_rate"]   for h in rn_history]
    rn_ls  = [h.get("avg_loss", float("nan")) for h in rn_history]
    rn_ep  = [h["epsilon"]    for h in rn_history]

    # ── TD ──────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(td_eps, [w * 100 for w in td_wr], color=COLOR_TD, lw=2, label="Win Rate")
    ax.axhline(50, color="gray", ls="--", alpha=0.55, lw=1)
    ax.set_title("TD — Win Rate (%)", fontsize=10)
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(-2, 105)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(td_eps, td_rw, color="#2a9d8f", lw=2, label="Avg Reward")
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_title("TD — Reward Promedio", fontsize=10)
    ax.set_ylabel("Reward"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(td_eps, td_ep, color="#e76f51", lw=2, label="ε")
    ax.set_title("TD — Decaimiento ε", fontsize=10)
    ax.set_ylabel("ε"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ── RN ──────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(rn_eps, [w * 100 for w in rn_wr], color=COLOR_RN, lw=2, label="Win Rate")
    ax.axhline(50, color="gray", ls="--", alpha=0.55, lw=1)
    ax.set_title("RN — Win Rate (%)", fontsize=10)
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(-2, 105)
    ax.set_xlabel("Episodio"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    rn_ls_clean = [x if not np.isnan(x) else 0.0 for x in rn_ls]
    ax.plot(rn_eps, rn_ls_clean, color="#457b9d", lw=2, label="Avg Loss (MSE)")
    ax.set_title("RN — Loss Promedio (MSE)", fontsize=10)
    ax.set_ylabel("MSE"); ax.set_xlabel("Episodio")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(rn_eps, rn_ep, color="#e9c46a", lw=2, label="ε")
    ax.set_title("RN — Decaimiento ε", fontsize=10)
    ax.set_ylabel("ε"); ax.set_xlabel("Episodio")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Guardada → {save_path}")
    plt.close()


# PLOT 2 ─ HEATMAPS DE DISPAROS Y COLOCACIÓN

def plot_heatmaps(shot_td: np.ndarray, place_td: np.ndarray,
                  shot_rn: np.ndarray, place_rn: np.ndarray,
                  title: str = "Heatmaps",
                  save_path: str = "graficas/plot_heatmaps.png") -> None:
    """
    2×2 grid:
      [TD Disparos | RN Disparos ]
      [TD Colocac. | RN Colocac. ]
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    vmax_s = max(shot_td.max(), shot_rn.max(), 1)
    vmax_p = max(place_td.max(), place_rn.max(), 1)

    _draw_heatmap(axes[0, 0], shot_td,  "TD-Learning — Disparos",    CMAP_SHOTS, 0, vmax_s)
    _draw_heatmap(axes[0, 1], shot_rn,  "Red Neuronal — Disparos",   CMAP_SHOTS, 0, vmax_s)
    _draw_heatmap(axes[1, 0], place_td, "TD-Learning — Colocación",  CMAP_PLACE, 0, vmax_p)
    _draw_heatmap(axes[1, 1], place_rn, "Red Neuronal — Colocación", CMAP_PLACE, 0, vmax_p)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Guardada → {save_path}")
    plt.close()

# PLOT 3 ─ ESTRATEGIAS EMERGENTES

def plot_emergent_strategies(agent_td: TDAgent, agent_rn: NeuralNetAgent,
                             save_path: str = "graficas/plot_emergent_strategies.png") -> None:
    """
    Muestra qué aprendieron los modelos sobre el tablero vacío:
    · TD:  v_table (valor por celda)
    · RN:  Q-values sobre estado totalmente desconocido (preferencia de primer disparo)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Estrategias Emergentes — Preferencia de Disparo Aprendida",
                 fontsize=13, fontweight="bold")

    # ── TD: v-table ──────────────────────────────────────────────────────────
    vtable = agent_td.v_table
    vabs   = np.abs(vtable).max() if vtable.max() != 0 else 1
    _draw_heatmap(axes[0], vtable,
                  "TD-Learning — V-Table\n(valor aprendido por celda)",
                  CMAP_VTABLE, -vabs, vabs, annot_fmt="{:.2f}")
    im0 = axes[0].get_images()[0]
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label="V(s,a)")

    # ── RN: Q-values sobre estado vacío ──────────────────────────────────────
    empty_state = np.zeros(BOARD_SIZE ** 2 * 2, dtype=np.float32)  # tracking=0, own=0
    q_vals = agent_rn.battle_net.predict(empty_state).flatten()
    q_grid = q_vals.reshape(BOARD_SIZE, BOARD_SIZE)
    qabs   = np.abs(q_grid).max() if q_grid.max() != 0 else 1

    _draw_heatmap(axes[1], q_grid,
                  "Red Neuronal — Q-Values (tablero vacío)\n"
                  "(preferencia de primer disparo)",
                  CMAP_VTABLE, -qabs, qabs, annot_fmt="{:.2f}")
    im1 = axes[1].get_images()[0]
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label="Q(s,a)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Guardada → {save_path}")
    plt.close()


# PLOT 4 ─ DASHBOARD DE COMPETENCIA

def plot_competition_dashboard(results: dict,
                               wr_td_vs_rand: float,
                               wr_rn_vs_rand: float,
                               save_path: str = "graficas/plot_competition_dashboard.png") -> None:
    """
    Dashboard 2×3:
      [Win Rate acumulado (ancho 2)         │ Pie chart final    ]
      [Distribución de turnos  │ Reward/partida (ventana) │ Comparación ]
    """
    n       = results["n_games"]
    wh      = results["win_history"]
    steps   = results["steps_list"]
    r_td    = results["rewards_td"]
    r_rn    = results["rewards_rn"]
    w_td    = results["wins_td"]
    w_rn    = results["wins_rn"]
    draws   = results["draws"]

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Competencia Directa: TD-Learning vs Red Neuronal",
                 fontsize=14, fontweight="bold")

    # ── 1. Win Rate acumulado ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    cum_td, cum_rn = [], []
    cnt_td = cnt_rn = 0
    for i, w in enumerate(wh, 1):
        if w == 1:
            cnt_td += 1
        elif w == 0:
            cnt_rn += 1
        cum_td.append(cnt_td / i * 100)
        cum_rn.append(cnt_rn / i * 100)

    x = range(1, n + 1)
    ax1.plot(x, cum_td, color=COLOR_TD, lw=1.8, label="TD-Learning", alpha=0.9)
    ax1.plot(x, cum_rn, color=COLOR_RN, lw=1.8, label="Red Neuronal", alpha=0.9)
    ax1.axhline(50, color="gray", ls=":", lw=1.2, alpha=0.7, label="50% ref.")
    ax1.fill_between(x, cum_td, cum_rn, where=[t > r for t, r in zip(cum_td, cum_rn)],
                     alpha=0.08, color=COLOR_TD)
    ax1.fill_between(x, cum_td, cum_rn, where=[r >= t for t, r in zip(cum_td, cum_rn)],
                     alpha=0.08, color=COLOR_RN)
    ax1.set_title("Win Rate Acumulado (competencia)", fontsize=11)
    ax1.set_xlabel("Partida"); ax1.set_ylabel("Win Rate (%)")
    ax1.set_ylim(0, 100); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # ── 2. Pie chart resultado final ──────────────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 2])
    vals = [w_td, w_rn, draws]
    lbls = [f"TD  {w_td/n:.1%}", f"RN  {w_rn/n:.1%}", f"Empt {draws/n:.1%}"]
    clrs = [COLOR_TD, COLOR_RN, "#cccccc"]
    wedges, texts, autotexts = ax2.pie(
        vals, labels=lbls, colors=clrs, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 8.5}, pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(8)
    winner_lbl = ("TD-Learning" if w_td > w_rn
                  else "Red Neuronal" if w_rn > w_td else "Empate")
    ax2.set_title(f"Resultado Final\n🏆 {winner_lbl}", fontsize=11)

    # ── 3. Distribución de turnos ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(steps, bins=30, color="#457b9d", alpha=0.85, edgecolor="white")
    mu = np.mean(steps)
    ax3.axvline(mu, color="#e63946", ls="--", lw=1.8,
                label=f"μ = {mu:.1f} turnos")
    ax3.set_title("Duración de las Partidas", fontsize=10)
    ax3.set_xlabel("Turnos"); ax3.set_ylabel("Frecuencia")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ── 4. Reward promedio (ventana) ──────────────────────────────────────────
    ax4   = fig.add_subplot(gs[1, 1])
    W     = 50
    smooth_td = [np.mean(r_td[max(0, i-W): i+1]) for i in range(n)]
    smooth_rn = [np.mean(r_rn[max(0, i-W): i+1]) for i in range(n)]
    ax4.plot(x, smooth_td, color=COLOR_TD, lw=1.5, label="TD")
    ax4.plot(x, smooth_rn, color=COLOR_RN, lw=1.5, label="RN")
    ax4.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax4.set_title(f"Reward Promedio (ventana {W} partidas)", fontsize=10)
    ax4.set_xlabel("Partida"); ax4.set_ylabel("Reward")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ── 5. Comparación vs Random ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    cats = ["vs Agente\nAleatorio", "vs Oponente\n(competencia)"]
    td_v = [wr_td_vs_rand * 100, w_td / n * 100]
    rn_v = [wr_rn_vs_rand * 100, w_rn / n * 100]
    xp   = np.arange(len(cats))
    bw   = 0.3
    b1   = ax5.bar(xp - bw / 2, td_v, bw, label="TD", color=COLOR_TD, alpha=0.88)
    b2   = ax5.bar(xp + bw / 2, rn_v, bw, label="RN", color=COLOR_RN, alpha=0.88)
    ax5.axhline(50, color="gray", ls="--", lw=1, alpha=0.6)
    ax5.set_xticks(xp); ax5.set_xticklabels(cats, fontsize=9)
    ax5.set_ylim(0, 108); ax5.set_ylabel("Win Rate (%)")
    ax5.set_title("Comparación de Desempeño", fontsize=10)
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3, axis="y")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Guardada → {save_path}")
    plt.close()

# PLOT 5 ─ REPLAY VISUAL

def plot_replay_visual(frames: list, winner: int,
                       max_frames: int = 12,
                       save_path: str = "graficas/plot_replay_visual.png") -> None:
    """
    Selecciona max_frames snapshots distribuidos uniformemente en la partida
    y los muestra en una cuadrícula.  Cada snapshot muestra el tablero de
    rastreo del jugador que acaba de disparar, resaltando la celda elegida.
    Leyenda: 0=Desconocido  1=Agua  2=Hit  3=Hundido
    """
    if not frames:
        print("  (Replay vacío — gráfica omitida)")
        return

    indices  = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)),
                           dtype=int).tolist()
    n_frames = len(indices)
    ncols    = 4
    nrows    = (n_frames + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).flatten()

    # Colormap: desconocido=gris claro, agua=azul, hit=naranja, hundido=rojo
    cmap_track = LinearSegmentedColormap.from_list(
        "track", ["#dde8f5", "#5a9ac9", "#f4a261", "#e63946"], N=4)

    for plot_i, frame_i in enumerate(indices):
        ax    = axes_flat[plot_i]
        frame = frames[frame_i]
        s     = frame["shooter"]

        grid  = frame["track_td"] if s == 0 else frame["track_rn"]
        lbl   = "TD dispara" if s == 0 else "RN dispara"
        color = COLOR_TD if s == 0 else COLOR_RN

        im = ax.imshow(grid, cmap=cmap_track, vmin=0, vmax=3, aspect="equal")

        r_s, c_s = divmod(frame["action"], BOARD_SIZE)
        result   = ("HUNDIDO" if frame["sunk"]
                    else "HIT"   if frame["hit"]
                    else "agua")
        rw_str = f"r={frame['reward']:+.1f}"

        ax.set_title(f"T={frame['step']+1}  {lbl}\n"
                     f"({r_s},{c_s}) {result}  {rw_str}",
                     fontsize=8, color=color, fontweight="bold")
        ax.set_xticks(range(BOARD_SIZE))
        ax.set_yticks(range(BOARD_SIZE))
        ax.tick_params(labelsize=6)

        # Resaltar celda disparada
        rect = plt.Rectangle((c_s - 0.5, r_s - 0.5), 1, 1,
                               fill=False, edgecolor="yellow", lw=2.5)
        ax.add_patch(rect)

    # Ocultar ejes sobrantes
    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    winner_name = "TD-Learning" if winner == 0 else ("Red Neuronal" if winner == 1 else "?")
    fig.suptitle(f"Replay Visual — {len(frames)} turnos totales  │  Ganador: {winner_name}",
                 fontsize=13, fontweight="bold")

    # Leyenda compartida
    legend_items = [
        mpatches.Patch(facecolor="#dde8f5", label="Desconocido"),
        mpatches.Patch(facecolor="#5a9ac9", label="Agua"),
        mpatches.Patch(facecolor="#f4a261", label="Hit"),
        mpatches.Patch(facecolor="#e63946", label="Hundido"),
        mpatches.Patch(facecolor="none",    label="Celda disparada",
                       edgecolor="yellow", linewidth=2),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Guardada → {save_path}")
    plt.close()


# MAIN
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Competencia TD vs RN en Battleship")
    p.add_argument("--retrain",  action="store_true",
                   help="Forzar reentrenamiento aunque existan checkpoints")
    p.add_argument("--no-train", action="store_true",
                   help="Solo cargar modelos y ejecutar competencia (requiere checkpoints)")
    p.add_argument("--games",    type=int, default=1000,
                   help="Número de partidas en la competencia (default: 1000)")
    p.add_argument("--episodes", type=int, default=800,
                   help="Episodios de entrenamiento por modelo (default: 800)")
    return p.parse_args()


def main():
    args = _parse_args()
    t0   = time.time()

    print("\n" + "═" * 60)
    print("  BATTLESHIP AI COMPETITION")
    print("  TD-Learning vs Red Neuronal  (tablero 7×7)")
    print("═" * 60)

    force_retrain = args.retrain
    no_train      = args.no_train

    #  ENTRENAR O CARGAR MODELOS

    td_history = []
    rn_history = []
    shot_hm_td_tr  = np.zeros((BOARD_SIZE, BOARD_SIZE))
    place_hm_td_tr = np.zeros((BOARD_SIZE, BOARD_SIZE))
    shot_hm_rn_tr  = np.zeros((BOARD_SIZE, BOARD_SIZE))
    place_hm_rn_tr = np.zeros((BOARD_SIZE, BOARD_SIZE))

    TD_EPISODES = 800    # TD ya converge aquí
    RN_EPISODES = 800

    # ── TD-Learning ──────────────────────────────────────────────────────────
    if not no_train and (force_retrain or not os.path.exists(TD_SAVE_PATH)):
        print(f"\n[1/2] Entrenando TD-Agent ({args.episodes} episodios)...")
        (agent_td, td_history,
         shot_hm_td_tr, place_hm_td_tr) = train_td_with_metrics(
            episodes   = TD_EPISODES,
            save_path  = TD_SAVE_PATH,
        )
    elif os.path.exists(TD_SAVE_PATH):
        print(f"\n[1/2] TD: checkpoint encontrado → cargando desde {TD_SAVE_PATH}")
        agent_td = load_td_agent(TD_SAVE_PATH)
    else:
        raise FileNotFoundError(
            f"No se encontró '{TD_SAVE_PATH}'. "
            "Ejecuta sin --no-train para entrenar primero."
        )
    agent_td.epsilon = 0.0   # modo evaluación

    # ── Red Neuronal ──────────────────────────────────────────────────────────
    if not no_train and (force_retrain or not _rn_files_exist(RN_SAVE_PREFIX)):
        print(f"\n[2/2] Entrenando RN-Agent ({args.episodes} episodios)...")
        (agent_rn, rn_history,
         shot_hm_rn_tr, place_hm_rn_tr) = train_rn_with_metrics(
            n_episodes  = RN_EPISODES,
            save_prefix = RN_SAVE_PREFIX,
        )
    elif _rn_files_exist(RN_SAVE_PREFIX):
        print(f"\n[2/2] RN: checkpoints encontrados → cargando desde {RN_SAVE_PREFIX}_*.npz")
        agent_rn = load_rn_agent(RN_SAVE_PREFIX)
    else:
        raise FileNotFoundError(
            f"No se encontraron '{RN_SAVE_PREFIX}_battle.npz' / '_placement.npz'. "
            "Ejecuta sin --no-train para entrenar primero."
        )
    agent_rn.training = False
    agent_rn.epsilon  = 0.0

    # PASO 2 ─ MÉTRICAS DE ENTRENAMIENTO
    print("\n[Gráficas]  Métricas de entrenamiento individual...")
    plot_training_metrics(td_history, rn_history)

    if td_history or rn_history:
        # Heatmaps de la fase de entrenamiento (solo si entrenamos en esta sesión)
        print("[Gráficas]  Heatmaps del entrenamiento...")
        plot_heatmaps(
            shot_hm_td_tr, place_hm_td_tr,
            shot_hm_rn_tr, place_hm_rn_tr,
            title     = "Heatmaps — Fase de Entrenamiento",
            save_path = "graficas/plot_heatmaps_training.png",
        )

    # EVALUACIÓN vs AGENTE ALEATORIO (línea base)
    print("\n[Eval]  Evaluando ambos modelos vs Agente Aleatorio (300 partidas c/u)...")
    wr_td = evaluate_vs_random(agent_td, n_games=300, label="TD-Learning")
    wr_rn = evaluate_vs_random(agent_rn, n_games=300, label="Red Neuronal")

    #COMPETENCIA DIRECTA
    results = run_competition(agent_td, agent_rn, n_games=args.games)

    #  REPLAY VISUAL
    print("[Gráficas]  Capturando replay de una partida de ejemplo...")
    frames, replay_winner = capture_replay(agent_td, agent_rn)
    plot_replay_visual(frames, replay_winner, max_frames=12)

    #  HEATMAPS DE COMPETENCIA
    print("[Gráficas]  Heatmaps de disparos y colocación (competencia)...")
    plot_heatmaps(
        results["shot_hm_td"], results["place_hm_td"],
        results["shot_hm_rn"], results["place_hm_rn"],
        title     = "Heatmaps — Fase de Competencia  (TD vs RN)",
        save_path = "graficas/plot_heatmaps_competition.png",
    )

    # ESTRATEGIAS EMERGENTES
    print("[Gráficas]  Estrategias emergentes...")
    plot_emergent_strategies(agent_td, agent_rn)

   
    # DASHBOARD DE COMPETENCIA
    print("[Gráficas]  Dashboard de competencia...")
    plot_competition_dashboard(results, wr_td, wr_rn)

    # RESUMEN FINAL
    tf       = time.time()
    w_td     = results["wins_td"]
    w_rn     = results["wins_rn"]
    n_games  = results["n_games"]
    champ    = ("TD-Learning" if w_td > w_rn
                else "Red Neuronal" if w_rn > w_td
                else "EMPATE")

    print(f"\n{'═'*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*60}")
    print(f"  TD-Learning  ganó:  {w_td:>5} / {n_games}  ({w_td/n_games:.1%})")
    print(f"  Red Neuronal ganó:  {w_rn:>5} / {n_games}  ({w_rn/n_games:.1%})")
    print(f"  Empates:            {results['draws']:>5} / {n_games}  ({results['draws']/n_games:.1%})")
    print(f"")
    print(f"  TD vs Aleatorio:    {wr_td:.1%}")
    print(f"  RN vs Aleatorio:    {wr_rn:.1%}")
    print(f"")
    print(f"  Turnos promedio por partida: {np.mean(results['steps_list']):.1f}")
    print(f"  Tiempo total: {(tf-t0)/60:.1f} min")
    print(f"")
    print(f"  ⭐  GANADOR: {champ}")
    print(f"{'═'*60}\n")

    print("  Archivos generados:")
    for fname in [
        "plot_training_metrics.png",
        "plot_heatmaps_training.png",
        "plot_heatmaps_competition.png",
        "plot_emergent_strategies.png",
        "plot_competition_dashboard.png",
        "plot_replay_visual.png",
    ]:
        if os.path.exists(fname):
            print(f"    ✓  {fname}")

    return results


if __name__ == "__main__":
    main()