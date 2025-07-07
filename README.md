# Obligatorio_2_Sistemas_Multiagentes

**Juegos alternados de información perfecta e imperfecta**

## Objetivo

El objetivo principal de este trabajo es implementar y validar diferentes estrategias de decisión y adaptación para agentes autónomos que interactúan en escenarios de **juegos alternados**, tanto de **información perfecta** como **imperfecta**.  
Se analiza el desempeño de algoritmos representativos del aprendizaje multiagente en entornos competitivos, con foco en:

- **Monte Carlo Tree Search (MCTS)**
- **Minimax (MM)**
- **Counterfactual Regret Minimization (CRM)**

Cada algoritmo aborda la exploración, la actualización de políticas y la adaptación desde diferentes perspectivas, en función del tipo de juego y la información disponible.

---

## Estructura del proyecto

```bash
.
├── agents/                    # Implementación de agentes inteligentes
│   ├── agent_random.py
│   ├── counterfactualregret_t.py
│   ├── mcts_t_V2.py
│   └── minimax.py
│
├── base/                      # Clases base comunes para los entornos y agentes
│   ├── agent.py
│   ├── game.py
│   └── utils.py
│
├── games/                     # Lógica de los entornos de juegos
│   ├── kuhn_poker/
│   │   ├── kuhn_2.py
│   │   └── kuhn3.py
│   ├── nocca_nocca/
│   │   ├── board.py
│   │   ├── nocca_nocca.py
│   │   └── nocca_nocca_original.py
│   └── tictactoe/
│       ├── tictactoe.py
│       ├── tictactoe_v3.py
│       └── tictactoe_env.py
│
├── obligatorio/               # Experimentos y análisis por juego
│   ├── Khun_Poker/
│   │   ├── KuhnPoker_2player.ipynb
│   │   ├── KuhnPoker_3player.ipynb
│   │   └── torneo_khun_poker_3.py
│   ├── Nocca_Nocca/
│   │   ├── Nocca_Nocca.ipynb
│   │   └── torneo_nocca_nocca.py
│   └── TicTacToe/
│       ├── TicTacToe.ipynb
│       └── torneo_TicTacToe.py
│
├── informe.pdf                # Informe final del proyecto
```

## Autor
Valeria Eskenazi

