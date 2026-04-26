"""
battleship_env.py
=================
Entorno base para el proyecto final de IA: Competencia TD-Learning vs Red Neuronal.
Juego: Battleship en tablero 7x7.

Flujo de uso:
  1. env = BattleshipEnv()
  2. env.reset()
  3. Fase de colocación  → cada agente llama a env.place_ship() o env.place_ships_randomly()
  4. Fase de batalla     → turnos alternos con env.step(action)
  5. env.run_game(agent0, agent1) automatiza todo lo anterior

Interfaz que debe implementar cada agente:
  - place_ships(env, player_id)          → coloca los barcos en el entorno
  - select_action(obs, valid_actions)    → retorna una acción (int 0..48)

Autores: Renato R., Melisa M., Micaela Y.
Curso:   Inteligencia Artificial
"""

import numpy as np
import random
from copy import deepcopy


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

BOARD_SIZE = 7                  # Dimensión del tablero (7×7)
SHIP_SIZES = [4, 3, 2, 2]      # Tamaños de barcos, de mayor a menor
NUM_SHIPS  = len(SHIP_SIZES)    # 4 barcos en total

# ── Codificación: tablero propio ─────────────────────────────────────────────
OWN_EMPTY = 0   # celda vacía
OWN_SHIP  = 1   # celda con barco (intacto)
OWN_HIT   = 2   # celda de barco que fue impactada

# ── Codificación: tablero de rastreo (lo que sé del oponente) ────────────────
TRACK_UNKNOWN = 0   # no disparé aquí aún
TRACK_MISS    = 1   # disparé → agua
TRACK_HIT     = 2   # disparé → impacto (barco no hundido aún)
TRACK_SUNK    = 3   # disparé → impacto que completó el hundimiento

# ── Orientaciones de colocación ──────────────────────────────────────────────
HORIZONTAL = 0
VERTICAL   = 1

# ── Rewards ──────────────────────────────────────────────────────────────────
# Ajustar estos valores durante el entrenamiento si es necesario.
REWARD_MISS    = -0.1   # disparo que no impacta
REWARD_HIT     =  0.5   # impacto sin hundir
REWARD_SUNK    =  1.5   # barco hundido (reward adicional al HIT)
REWARD_WIN     =  5.0   # ganar la partida
REWARD_LOSE    = -3.0   # perder la partida (aplicado al perdedor al final)
REWARD_INVALID = -0.5   # disparar a una celda ya conocida (inválido)

# Nota sobre el reward de colocación:
#   La colocación NO tiene reward inmediato. Sin embargo, una buena estrategia
#   de colocación queda implícitamente recompensada a través de REWARD_LOSE
#   (barcos bien colocados son más difíciles de hundir, postergando la derrota).
#   Ambos modelos aprenden a colocar y a atacar en el mismo loop de entrenamiento.


# ══════════════════════════════════════════════════════════════════════════════
# CLASE: Ship
# ══════════════════════════════════════════════════════════════════════════════

class Ship:
    """
    Representa un barco individual dentro del tablero.
    Registra qué celdas ocupa y cuáles han sido impactadas.
    """

    def __init__(self, size: int, cells: list):
        """
        Parámetros
        ----------
        size  : longitud del barco (número de celdas)
        cells : lista de tuplas (row, col) que ocupa el barco
        """
        self.size  = size
        self.cells = set(cells)     # conjunto de todas las celdas del barco
        self.hits  = set()          # celdas impactadas

    def receive_hit(self, row: int, col: int) -> bool:
        """
        Registra un impacto en (row, col).
        Retorna True si la celda pertenece a este barco (y no estaba ya impactada).
        """
        if (row, col) in self.cells and (row, col) not in self.hits:
            self.hits.add((row, col))
            return True
        return False

    def is_sunk(self) -> bool:
        """True si todas las celdas del barco fueron impactadas."""
        return self.hits == self.cells

    def __repr__(self):
        status = "HUNDIDO" if self.is_sunk() else f"{len(self.hits)}/{self.size} hits"
        return f"Ship(size={self.size}, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# CLASE: Board
# ══════════════════════════════════════════════════════════════════════════════

class Board:
    """
    Tablero de un jugador. Contiene:
      - own_grid:      lo que el jugador tiene (sus barcos y los impactos recibidos)
      - tracking_grid: lo que el jugador sabe sobre el tablero del oponente
      - ships:         lista de objetos Ship propios
    """

    def __init__(self):
        self.own_grid      = np.full((BOARD_SIZE, BOARD_SIZE), OWN_EMPTY, dtype=np.int8)
        self.tracking_grid = np.full((BOARD_SIZE, BOARD_SIZE), TRACK_UNKNOWN, dtype=np.int8)
        self.ships: list[Ship] = []

    def reset(self):
        """Limpia el tablero por completo."""
        self.own_grid[:]      = OWN_EMPTY
        self.tracking_grid[:] = TRACK_UNKNOWN
        self.ships            = []

    # ── Colocación ───────────────────────────────────────────────────────────

    def _cells_for(self, size: int, row: int, col: int, orientation: int) -> list:
        """Genera la lista de celdas que ocuparía un barco según parámetros dados."""
        if orientation == HORIZONTAL:
            return [(row, col + i) for i in range(size)]
        else:
            return [(row + i, col) for i in range(size)]

    def _in_bounds(self, cells: list) -> bool:
        """True si todas las celdas están dentro del tablero."""
        return all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE for r, c in cells)

    def _cells_free(self, cells: list) -> bool:
        """
        True si las celdas están libres y no son adyacentes a otro barco.
        Se aplica la regla de no contacto: los barcos no pueden tocarse ni en diagonal.
        """
        occupied = set(cells)
        for (r, c) in cells:
            if self.own_grid[r, c] == OWN_SHIP:
                return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in occupied:
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if self.own_grid[nr, nc] == OWN_SHIP:
                                return False
        return True

    def valid_placement_positions(self, size: int, orientation: int) -> list:
        """
        Retorna lista de (row, col) válidos para colocar un barco de 'size'
        celdas con la orientación indicada, considerando bordes y barcos existentes.
        """
        positions = []
        row_limit = BOARD_SIZE if orientation == HORIZONTAL else BOARD_SIZE - size + 1
        col_limit = BOARD_SIZE - size + 1 if orientation == HORIZONTAL else BOARD_SIZE

        for r in range(row_limit):
            for c in range(col_limit):
                cells = self._cells_for(size, r, c, orientation)
                if self._in_bounds(cells) and self._cells_free(cells):
                    positions.append((r, c))
        return positions

    def place_ship(self, size: int, row: int, col: int, orientation: int) -> bool:
        """
        Intenta colocar un barco de 'size' celdas en (row, col) con la orientación dada.
        Retorna True si la colocación fue exitosa; False en caso contrario.
        """
        cells = self._cells_for(size, row, col, orientation)
        if not self._in_bounds(cells) or not self._cells_free(cells):
            return False

        ship = Ship(size, cells)
        self.ships.append(ship)
        for (r, c) in cells:
            self.own_grid[r, c] = OWN_SHIP
        return True

    # ── Recibir disparos ─────────────────────────────────────────────────────

    def receive_shot(self, row: int, col: int) -> tuple:
        """
        Procesa un disparo entrante en (row, col).

        Retorna
        -------
        (hit, sunk_ship)
          hit       : bool  → True si impactó algún barco
          sunk_ship : Ship | None → barco hundido, o None si no se hundió ninguno
        """
        for ship in self.ships:
            if (row, col) in ship.cells and (row, col) not in ship.hits:
                ship.receive_hit(row, col)
                self.own_grid[row, col] = OWN_HIT
                return True, (ship if ship.is_sunk() else None)
        return False, None

    def all_ships_sunk(self) -> bool:
        """True si todos los barcos propios han sido hundidos."""
        return all(ship.is_sunk() for ship in self.ships)

    # ── Actualizar tracking ──────────────────────────────────────────────────

    def update_tracking(self, row: int, col: int, hit: bool, sunk_ship) -> None:
        """
        Actualiza el tablero de rastreo tras haber disparado a (row, col).

        Parámetros
        ----------
        hit       : bool  → True si impactó
        sunk_ship : Ship | None → barco hundido (para marcar todas sus celdas)
        """
        if hit:
            if sunk_ship is not None:
                # Marcamos todas las celdas del barco hundido como SUNK
                for (r, c) in sunk_ship.cells:
                    self.tracking_grid[r, c] = TRACK_SUNK
            else:
                self.tracking_grid[row, col] = TRACK_HIT
        else:
            self.tracking_grid[row, col] = TRACK_MISS

    # ── Acciones válidas ─────────────────────────────────────────────────────

    def valid_shoot_actions(self) -> list:
        """
        Retorna lista de acciones válidas de disparo: celdas aún UNKNOWN en el
        tablero de rastreo, codificadas como entero (row * BOARD_SIZE + col).
        """
        return [
            r * BOARD_SIZE + c
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.tracking_grid[r, c] == TRACK_UNKNOWN
        ]

    # ── Info auxiliar ────────────────────────────────────────────────────────

    def ships_remaining(self) -> int:
        """Número de barcos propios que aún no han sido hundidos."""
        return sum(1 for s in self.ships if not s.is_sunk())

    def ships_status(self) -> list:
        """Retorna una lista de dicts con el estado de cada barco propio."""
        return [
            {"size": s.size, "sunk": s.is_sunk(), "hits": len(s.hits)}
            for s in self.ships
        ]


# ══════════════════════════════════════════════════════════════════════════════
# CLASE: BattleshipEnv
# ══════════════════════════════════════════════════════════════════════════════

class BattleshipEnv:
    """
    Entorno de Battleship para dos jugadores (jugador 0 y jugador 1).

    FASES DEL JUEGO
    ───────────────
    "placement"  → ambos jugadores colocan sus barcos (pueden hacerlo en paralelo).
                   La fase termina automáticamente cuando ambos completaron la colocación.
    "battle"     → turnos alternos de disparo. Comienza el jugador 0.
                   La fase termina cuando todos los barcos de un jugador son hundidos.

    REPRESENTACIÓN DE ACCIONES (fase de batalla)
    ─────────────────────────────────────────────
    Las acciones son enteros en [0, BOARD_SIZE^2 - 1].
    Conversión: action = row * BOARD_SIZE + col
    Helpers:    BattleshipEnv.action_to_coords(a) y .coords_to_action(r, c)

    OBSERVACIÓN (get_observation)
    ──────────────────────────────
    Retorna un dict con:
      "tracking"      → np.array (7,7) int8   — lo que sé del oponente
      "own"           → np.array (7,7) int8   — mi tablero propio
      "valid_actions" → list[int]             — acciones de disparo disponibles
      "flat"          → np.array (98,) float32 — concatenación normalizada (útil para RN)
      "phase"         → str                   — fase actual

    Para TD-Learning, usar get_state_tuple(player) → tupla hashable del tracking_grid.
    """

    def __init__(self):
        self.boards         = [Board(), Board()]
        self.current_player = 0
        self.phase          = "placement"
        self.done           = False
        self.winner         = None
        self._pending_ships = [list(SHIP_SIZES), list(SHIP_SIZES)]

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> tuple:
        """
        Reinicia el entorno completamente.
        Retorna las observaciones iniciales de ambos jugadores: (obs0, obs1).
        """
        for board in self.boards:
            board.reset()
        self.current_player = 0
        self.phase          = "placement"
        self.done           = False
        self.winner         = None
        self._pending_ships = [list(SHIP_SIZES), list(SHIP_SIZES)]
        return self.get_observation(0), self.get_observation(1)

    # ── Fase de colocación ───────────────────────────────────────────────────

    def next_ship_size(self, player: int):
        """Tamaño del próximo barco a colocar para 'player', o None si ya terminó."""
        pending = self._pending_ships[player]
        return pending[0] if pending else None

    def get_valid_placements(self, player: int, orientation: int = None) -> dict | list:
        """
        Posiciones válidas para el próximo barco del jugador.

        Si orientation es None → retorna dict {HORIZONTAL: [...], VERTICAL: [...]}
        Si se especifica orientation → retorna list de (row, col)
        """
        size = self.next_ship_size(player)
        if size is None:
            return {} if orientation is None else []
        board = self.boards[player]
        if orientation is not None:
            return board.valid_placement_positions(size, orientation)
        return {
            HORIZONTAL: board.valid_placement_positions(size, HORIZONTAL),
            VERTICAL:   board.valid_placement_positions(size, VERTICAL),
        }

    def place_ship(self, player: int, row: int, col: int, orientation: int) -> bool:
        """
        Coloca el siguiente barco pendiente del jugador en (row, col) con la orientación dada.
        Retorna True si fue exitoso.
        Avanza automáticamente a fase "battle" cuando ambos jugadores terminan de colocar.
        """
        if self.phase != "placement":
            raise RuntimeError("No estamos en fase de colocación.")
        size = self.next_ship_size(player)
        if size is None:
            raise RuntimeError(f"Jugador {player} ya colocó todos sus barcos.")

        success = self.boards[player].place_ship(size, row, col, orientation)
        if success:
            self._pending_ships[player].pop(0)
            if not self._pending_ships[0] and not self._pending_ships[1]:
                self.phase = "battle"
        return success

    def place_ships_randomly(self, player: int) -> None:
        """
        Coloca todos los barcos restantes del jugador en posiciones aleatorias válidas.
        Útil como fallback o para el agente aleatorio de referencia.
        """
        while self._pending_ships[player]:
            size    = self._pending_ships[player][0]
            placed  = False
            attempts = 0
            while not placed and attempts < 2000:
                orientation = random.choice([HORIZONTAL, VERTICAL])
                positions   = self.boards[player].valid_placement_positions(size, orientation)
                if positions:
                    row, col = random.choice(positions)
                    placed   = self.place_ship(player, row, col, orientation)
                attempts += 1
            if not placed:
                raise RuntimeError(
                    f"No se pudo colocar barco de tamaño {size} para jugador {player}."
                )

    def placement_done(self, player: int) -> bool:
        """True si el jugador ya terminó de colocar todos sus barcos."""
        return len(self._pending_ships[player]) == 0

    # ── Fase de batalla: step ────────────────────────────────────────────────

    def step(self, action: int) -> tuple:
        """
        El jugador activo (self.current_player) dispara a la celda codificada en 'action'.

        Parámetro
        ---------
        action : int en [0, BOARD_SIZE^2 - 1]
                 Equivale a (action // BOARD_SIZE, action % BOARD_SIZE)

        Retorna
        -------
        obs_shooter  : dict  — observación del jugador que disparó (post-disparo)
        obs_opponent : dict  — observación del oponente
        reward       : float — reward para el jugador activo
        done         : bool  — True si el juego terminó
        info         : dict  — metadata del turno (ver abajo)

        info contiene:
          "invalid"  : bool  — True si la acción era inválida
          "hit"      : bool  — True si el disparo impactó un barco
          "sunk"     : bool  — True si el disparo hundió un barco
          "shooter"  : int   — jugador que disparó
          "winner"   : int   — jugador ganador (solo si done=True)
        """
        if self.phase != "battle":
            raise RuntimeError("El juego no está en fase de batalla.")
        if self.done:
            raise RuntimeError("El juego ya terminó. Llama reset() para reiniciar.")

        shooter  = self.current_player
        opponent = 1 - shooter
        row, col = divmod(action, BOARD_SIZE)

        # ── Validar acción ──────────────────────────────────────────────────
        if self.boards[shooter].tracking_grid[row, col] != TRACK_UNKNOWN:
            obs_s = self.get_observation(shooter)
            obs_o = self.get_observation(opponent)
            return obs_s, obs_o, REWARD_INVALID, False, {
                "invalid": True, "hit": False, "sunk": False, "shooter": shooter
            }

        # ── Aplicar disparo ─────────────────────────────────────────────────
        hit, sunk_ship = self.boards[opponent].receive_shot(row, col)

        # ── Actualizar tracking del shooter ─────────────────────────────────
        self.boards[shooter].update_tracking(row, col, hit, sunk_ship)

        # ── Calcular reward y verificar fin ─────────────────────────────────
        game_over = self.boards[opponent].all_ships_sunk()
        reward    = self._compute_reward(hit, sunk_ship, game_over)

        info = {
            "invalid": False,
            "hit":     hit,
            "sunk":    sunk_ship is not None,
            "shooter": shooter,
        }

        if game_over:
            self.done   = True
            self.winner = shooter
            info["winner"] = shooter
        else:
            self.current_player = opponent   # turno al oponente

        obs_s = self.get_observation(shooter)
        obs_o = self.get_observation(opponent)
        return obs_s, obs_o, reward, self.done, info

    def _compute_reward(self, hit: bool, sunk_ship, game_over: bool) -> float:
        """
        Calcula el reward para el jugador que acaba de disparar.

        Jerarquía:
          WIN  → SUNK  → HIT  → MISS
        (gana solo si hundió el último barco, por eso se verifica primero)
        """
        if game_over:
            return REWARD_WIN
        if sunk_ship is not None:
            return REWARD_SUNK      # hundió un barco (pero el juego continúa)
        if hit:
            return REWARD_HIT
        return REWARD_MISS

    # ── Observaciones ────────────────────────────────────────────────────────

    def get_observation(self, player: int) -> dict:
        """
        Estado observable del juego desde la perspectiva de 'player'.

        Retorna dict con:
          "tracking"      → np.array (7,7) int8    valores: 0=UNKNOWN 1=MISS 2=HIT 3=SUNK
          "own"           → np.array (7,7) int8    valores: 0=EMPTY 1=SHIP 2=HIT
          "valid_actions" → list[int]              acciones de disparo disponibles
          "flat"          → np.array (98,) float32 concatenación normalizada
                            [tracking/3 | own/2]  (útil para entrada de Red Neuronal)
          "phase"         → str                    "placement" | "battle"

        Notas para los modelos:
          - La Red Neuronal puede usar "flat" directamente como input.
          - TD-Learning puede usar get_state_tuple() para indexar su tabla Q.
        """
        board    = self.boards[player]
        tracking = board.tracking_grid.copy()
        own      = board.own_grid.copy()
        flat     = np.concatenate([
            tracking.flatten() / 3.0,   # normalizado a [0, 1]
            own.flatten()      / 2.0,
        ]).astype(np.float32)

        return {
            "tracking":      tracking,
            "own":           own,
            "valid_actions": board.valid_shoot_actions(),
            "flat":          flat,
            "phase":         self.phase,
        }

    def get_state_tuple(self, player: int) -> tuple:
        """
        Retorna el tracking_grid del jugador como tupla hashable.
        Útil para TD-Learning con tabla Q (diccionario de estados).

        Nota: el espacio de estados (4^49 ≈ 10^29) es enorme para una tabla
        exacta. Usar aproximación de función en el modelo TD.
        """
        return tuple(self.boards[player].tracking_grid.flatten())

    # ── Información del juego ────────────────────────────────────────────────

    def ships_status(self, player: int) -> list:
        """Lista de dicts con el estado de cada barco del jugador."""
        return self.boards[player].ships_status()

    def ships_remaining(self, player: int) -> int:
        """Número de barcos del jugador que aún no han sido hundidos."""
        return self.boards[player].ships_remaining()

    def get_valid_actions(self, player: int) -> list:
        """Acciones de disparo válidas para el jugador (celdas aún UNKNOWN)."""
        return self.boards[player].valid_shoot_actions()

    # ── Render ───────────────────────────────────────────────────────────────

    _OWN_SYMBOLS   = {OWN_EMPTY: "·", OWN_SHIP: "O", OWN_HIT: "X"}
    _TRACK_SYMBOLS = {
        TRACK_UNKNOWN: "~",
        TRACK_MISS:    "•",
        TRACK_HIT:     "H",
        TRACK_SUNK:    "S",
    }

    def render(self, player: int = None) -> None:
        """
        Imprime el estado del juego en consola.

        Si player=None, muestra ambos tableros.
        Leyenda:
          Tablero propio:    · vacío  O barco  X impactado
          Tablero rastreo:   ~ desconocido  • agua  H hit  S hundido
        """
        targets   = [0, 1] if player is None else [player]
        header    = "  " + " ".join(str(i) for i in range(BOARD_SIZE))

        for p in targets:
            board = self.boards[p]
            print(f"\n{'═'*52}")
            print(f"  JUGADOR {p}   |   Fase: {self.phase}"
                  f"   |   Turno: {'→ TÚ ←' if self.current_player == p else '  oponente'}")
            print(f"{'═'*52}")
            print(f"  Tablero propio:           Tablero de rastreo:")
            print(f"  {header}                 {header}")

            for r in range(BOARD_SIZE):
                own_row   = " ".join(
                    self._OWN_SYMBOLS[board.own_grid[r, c]]
                    for c in range(BOARD_SIZE)
                )
                track_row = " ".join(
                    self._TRACK_SYMBOLS[board.tracking_grid[r, c]]
                    for c in range(BOARD_SIZE)
                )
                print(f"{r} {own_row}               {r} {track_row}")

            print(f"  Barcos activos: {board.ships_remaining()}/{NUM_SHIPS}")

        if self.done:
            print(f"\n{'═'*52}")
            print(f" Fin del juego. ¡Ganó el Jugador {self.winner}!")
            print(f"{'═'*52}")

    # ── Correr una partida completa ──────────────────────────────────────────

    def run_game(self, agent0, agent1, render: bool = False, max_steps: int = 300) -> tuple:
        """
        Ejecuta una partida completa entre agent0 y agent1.

        Los agentes deben implementar:
          place_ships(env, player_id)         → colocar barcos en el entorno
          select_action(obs, valid_actions)   → retornar acción (int)

        Parámetros
        ----------
        render    : mostrar el tablero en cada turno
        max_steps : límite de disparos totales (evita loops infinitos)

        Retorna
        -------
        winner  : int (0 o 1) o -1 si se llegó al límite sin ganador
        steps   : int — número de disparos totales
        history : list[dict] — registro de cada turno
        """
        self.reset()
        agents = [agent0, agent1]

        # ─ Fase de colocación ──────────────────────────────────────────────
        for p, agent in enumerate(agents):
            agent.place_ships(self, p)

        if self.phase != "battle":
            raise RuntimeError(
                "Ambos agentes deben colocar todos sus barcos antes de iniciar la batalla."
            )

        if render:
            print("\n" + "═"*52)
            print("  INICIO DE PARTIDA")
            self.render()

        # ─ Fase de batalla ─────────────────────────────────────────────────
        history = []
        steps   = 0

        while not self.done and steps < max_steps:
            current = self.current_player
            obs     = self.get_observation(current)
            action  = agents[current].select_action(obs, obs["valid_actions"])

            obs_s, obs_o, reward, done, info = self.step(action)

            history.append({
                "step":   steps,
                "player": current,
                "action": action,
                "reward": reward,
                **info,
            })
            steps += 1

            if render:
                r, c = divmod(action, BOARD_SIZE)
                resultado = (
                    "HUNDIDO!!!" if info["sunk"]
                    else "IMPACTO!" if info["hit"]
                    else "agua..."
                )
                print(f"\n  Jugador {current} → dispara ({r},{c}): {resultado}  "
                      f"[reward={reward:+.2f}]")
                self.render()

        winner = self.winner if self.done else -1

        if render:
            if winner == -1:
                print(f"\nEmpate por límite de pasos ({max_steps}).")
            else:
                print(f"\nJugador {winner} ganó en {steps} disparos.")

        return winner, steps, history

    # ── Utilidades estáticas ─────────────────────────────────────────────────

    @staticmethod
    def action_to_coords(action: int) -> tuple:
        """Convierte una acción (int) a coordenadas (row, col)."""
        return divmod(action, BOARD_SIZE)

    @staticmethod
    def coords_to_action(row: int, col: int) -> int:
        """Convierte coordenadas (row, col) a acción (int)."""
        return row * BOARD_SIZE + col


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE ALEATORIO — baseline de referencia
# ══════════════════════════════════════════════════════════════════════════════

class RandomAgent: #     Usando solo este, veces falla, porque ya que es random, no decide tan bien y puede caer en una excepción (hacer acciones no permitidas). Si se planea usar solo este, reducir la cantidad de N para tener menos oportunidades de error, o asumir tener suerte xd
    """
    Agente que juega completamente al azar.
    Sirve como baseline para verificar que el entorno funciona,
    y como oponente inicial durante las primeras iteraciones de entrenamiento.

    Implementa la interfaz mínima requerida por run_game():
      - place_ships(env, player_id)
      - select_action(obs, valid_actions)
    """

    def __init__(self, name: str = "Random"):
        self.name = name

    def place_ships(self, env: BattleshipEnv, player_id: int) -> None:
        """Coloca todos los barcos en posiciones aleatorias válidas."""
        env.place_ships_randomly(player_id)

    def select_action(self, obs: dict, valid_actions: list) -> int:
        """Elige un disparo aleatorio entre las celdas aún no exploradas."""
        return random.choice(valid_actions)

    def __repr__(self):
        return f"RandomAgent('{self.name}')"


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA 
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: Battleship 7x7 — dos agentes aleatorios")
    print("=" * 60)

    env = BattleshipEnv()
    a0  = RandomAgent("Agente-0")
    a1  = RandomAgent("Agente-1")

    # Partida con render activado
    winner, steps, history = env.run_game(a0, a1, render=True)
    print(f"\nResumen: Ganador={winner}, Disparos totales={steps}")

    # Estadísticas sobre N partidas
    N = 500
    print(f"\nSimulando {N} partidas sin render...")
    wins        = [0, 0, 0]   # [j0_wins, j1_wins, empates]
    total_steps = []

    for _ in range(N):
        w, s, _ = env.run_game(a0, a1, render=False)
        idx = w if w != -1 else 2
        wins[idx] += 1
        total_steps.append(s)

    avg  = sum(total_steps) / len(total_steps)
    minn = min(total_steps)
    maxx = max(total_steps)

    print(f"  Agente-0 ganó:  {wins[0]:>4}  ({wins[0]/N*100:.1f}%)")
    print(f"  Agente-1 ganó:  {wins[1]:>4}  ({wins[1]/N*100:.1f}%)")
    print(f"  Empates:        {wins[2]:>4}  ({wins[2]/N*100:.1f}%)")
    print(f"  Disparos — promedio: {avg:.1f}  mín: {minn}  máx: {maxx}")
    print("\nEntorno validado correctamente.")