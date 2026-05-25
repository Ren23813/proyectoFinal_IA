# Battleship AI — TD-Learning vs Red Neuronal
### Proyecto Final — Inteligencia Artificial

---

## Descripción

Implementación y competencia de dos modelos de Inteligencia Artificial en el juego **Battleship** (Batalla Naval), sobre un tablero reducido de **7×7**. Ambos modelos fueron desarrollados completamente desde cero usando únicamente NumPy, sin librerías especializadas de IA.

Los modelos enfrentados son:
- **TD-Learning** — tabla de valores por celda con política ε-greedy
- **Red Neuronal** — DQN para disparo + REINFORCE para colocación de naves

---

## Estructura del proyecto

```
├── battleshipENV.py      # Entorno del juego: tablero, reglas, agente aleatorio
├── TDLearning.py         # Modelo TD-Learning y su entrenamiento
├── RN.py                 # Modelo Red Neuronal y su entrenamiento
├── competition.py        # Archivo de unión: entrena, compite y genera gráficas
└── README.md
```

---

## Requisitos

```
Python 3.10+
numpy
matplotlib
```

Instalación de dependencias:
```bash
pip install numpy matplotlib
```

---

## Uso

### Correr la competencia completa

```bash
python competition.py
```

Esto entrena ambos modelos (o carga checkpoints si ya existen), los evalúa contra un agente aleatorio, ejecuta 1000 partidas directas entre ellos y genera todas las gráficas.

### Opciones disponibles

```bash
python competition.py --retrain       # fuerza reentrenamiento aunque existan checkpoints
python competition.py --no-train      # solo carga modelos y compite (requiere checkpoints)
python competition.py --games 500     # cambia el número de partidas de la competencia
python competition.py --episodes 800  # cambia los episodios de entrenamiento
```

### Probar solo el entorno

```bash
python battleshipENV.py   # enfrenta dos agentes aleatorios con render activado
```

### Probar solo el TD-Learning

```bash
python TDLearning.py      # entrena y evalúa el TD-Learning por separado
```

### Probar solo la Red Neuronal

```bash
python RN.py              # entrena y evalúa la Red Neuronal por separado
```

---

## Archivos generados

### Checkpoints (modelos entrenados)
| Archivo | Contenido |
|---|---|
| `td_agent_vtable.pkl` | V-table del TD-Learning |
| `rn_agent_battle.npz` | Pesos de BattleNet (Red Neuronal) |
| `rn_agent_placement.npz` | Pesos de PlacementNet (Red Neuronal) |

### Gráficas
| Archivo | Descripción |
|---|---|
| `plot_training_metrics.png` | Win rate, reward y epsilon durante el entrenamiento |
| `plot_heatmaps_training.png` | Heatmaps de disparos y colocación (entrenamiento) |
| `plot_heatmaps_competition.png` | Heatmaps de disparos y colocación (competencia) |
| `plot_emergent_strategies.png` | V-table del TD y Q-values iniciales de la RN |
| `plot_competition_dashboard.png` | Dashboard completo de la competencia |
| `plot_replay_visual.png` | Replay visual de una partida de ejemplo |

---

## Detalles técnicos

### Entorno (battleshipENV.py)
- Tablero 7×7 (49 celdas)
- 4 naves de tamaños: 4, 3, 2, 2
- Regla de no contacto: las naves no pueden tocarse ni en diagonal
- Codificación de rewards: Miss = −0.1 · Hit = +0.5 · Hundido = +1.5 · Victoria = +5.0

### TD-Learning (TDLearning.py)
- Representación: matriz 7×7 de valores V(s,a) por celda
- Política: ε-greedy con ε-decay = 0.997
- Entrenamiento: 800 episodios · 80% self-play · 20% vs agente aleatorio
- Clone congelado actualizado cada 300 episodios

### Red Neuronal (RN.py)
- **BattleNet** `[98 → 128 → 64 → 49]` — Q-Learning con replay buffer (DQN)
- **PlacementNet** `[51 → 64 → 32 → 49]` — REINFORCE (policy gradient)
- Activación: ReLU en capas ocultas · salida lineal para BattleNet · softmax para PlacementNet
- Entrenamiento: 800 episodios · 80% self-play · 20% vs agente aleatorio
- Target network sincronizada cada 200 pasos de gradiente

---

## Resultados

| Métrica | TD-Learning | Red Neuronal |
|---|---|---|
| Win rate vs aleatorio | ~57% | ~60% |
| Win rate en competencia directa | ~12% | ~88% |
| Episodios hasta convergencia | ~500 | ~800 |

La Red Neuronal supera al TD-Learning porque aprende comportamientos contextuales (disparar cerca de un impacto previo) que la v_table no puede representar. Se verificó que aumentar los episodios de entrenamiento más allá de 800 no produce mejoras significativas en ninguno de los dos modelos, confirmando convergencia temprana en este entorno reducido.

---

## Autores

- Renato R.
- Melisa M.
- Micaela Y.

Curso: Inteligencia Artificial — Universidad del Valle