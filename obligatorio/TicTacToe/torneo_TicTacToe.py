import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import time

class AgentEvaluator:
    def __init__(self, game_class):
        self.game_class = game_class
        self.results = defaultdict(list)
        self.detailed_stats = defaultdict(dict)
        # Para tracking de evolución durante el torneo
        self.evolution_data = defaultdict(list)
        self.game_counter = 0
    
    def run_tournament(self, agents_dict, num_games=500):
        """
        Ejecuta torneo round-robin: todos contra todos
        
        Args:
            agents_dict: {'nombre': AgentObject}
            num_games: partidas por enfrentamiento
        """
        agent_names = list(agents_dict.keys())
        
        # Todos los emparejamientos posibles
        matchups = list(combinations(agent_names, 2))
        
        print(f"Ejecutando torneo con {len(matchups)} enfrentamientos")
        print(f"{num_games} partidas por enfrentamiento")
        
        for i, (agent1_name, agent2_name) in enumerate(matchups):
            print(f"\nEnfrentamiento {i+1}/{len(matchups)}: {agent1_name} vs {agent2_name}")
            
            # Jugar como X y como O
            stats1 = self._play_matches(
                agents_dict[agent1_name], agents_dict[agent2_name], 
                agent1_name, agent2_name, num_games//2
            )
            
            stats2 = self._play_matches(
                agents_dict[agent2_name], agents_dict[agent1_name],
                agent2_name, agent1_name, num_games//2
            )
            
            # Combinar estadísticas
            self._combine_stats(agent1_name, agent2_name, stats1, stats2)
    
    def _play_matches(self, agent1, agent2, name1, name2, num_games):
        """Juega num_games entre dos agentes específicos"""
        stats = {
            'wins_1': 0, 'wins_2': 0, 'draws': 0,
            'game_lengths': [], 'times_1': [], 'times_2': []
        }
        
        for game_num in range(num_games):
            # Crear nueva instancia del juego
            game = self.game_class(render_mode='')
            game.reset()
            
            # CLAVE: Actualizar la referencia del juego en los agentes
            agent1.game = game
            agent2.game = game
            
            game_length = 0
            
            while not game.terminated():
                # Determinar el agente actual basado en agent_selection
                if game.agent_selection == game.agents[0]:  # Primer agente (X)
                    current_agent = agent1
                    is_agent1 = True
                else:  # Segundo agente (O)
                    current_agent = agent2
                    is_agent1 = False
                
                # Medir tiempo de decisión
                start_time = time.time()
                action = current_agent.action()
                decision_time = time.time() - start_time
                
                # Registrar tiempo
                if is_agent1:
                    stats['times_1'].append(decision_time)
                else:
                    stats['times_2'].append(decision_time)
                
                # Ejecutar acción
                game.step(action)
                game_length += 1
                
                # Prevenir bucles infinitos
                if game_length > 100:
                    print(f"Juego demasiado largo, terminando...")
                    break
            
            # Registrar resultado
            stats['game_lengths'].append(game_length)
            
            # Obtener resultado del juego
            # En tu TicTacToe, game.rewards[agent] da 1 para ganador, -1 para perdedor, 0 para empate
            reward_agent1 = game.rewards[game.agents[0]]  # Recompensa del primer agente (X)
            reward_agent2 = game.rewards[game.agents[1]]  # Recompensa del segundo agente (O)
            
            if reward_agent1 == 1:  # agent1 ganó
                stats['wins_1'] += 1
                result_1, result_2 = 1, -1
            elif reward_agent1 == -1:  # agent2 ganó
                stats['wins_2'] += 1
                result_1, result_2 = -1, 1
            else:  # empate
                stats['draws'] += 1
                result_1, result_2 = 0, 0
            
            # Guardar evolución del torneo
            self.game_counter += 1
            self.evolution_data[name1].append({
                'game_number': self.game_counter,
                'opponent': name2,
                'result': result_1,
                'game_length': game_length,
                'decision_time': np.mean(stats['times_1'][-1:]) if stats['times_1'] else 0
            })
            self.evolution_data[name2].append({
                'game_number': self.game_counter,
                'opponent': name1,
                'result': result_2,
                'game_length': game_length,
                'decision_time': np.mean(stats['times_2'][-1:]) if stats['times_2'] else 0
            })
                
            # Progress bar simple
            if (game_num + 1) % 50 == 0:
                print(f"  Progreso: {game_num + 1}/{num_games}")
        
        return stats
    
    def _combine_stats(self, name1, name2, stats1, stats2):
        """Combina estadísticas de ambas direcciones del enfrentamiento"""
        total_games = len(stats1['game_lengths']) + len(stats2['game_lengths'])
        
        # Para name1
        wins_1 = stats1['wins_1'] + stats2['wins_2']  # victorias cuando jugó como X y como O
        losses_1 = stats1['wins_2'] + stats2['wins_1']
        draws_1 = stats1['draws'] + stats2['draws']
        
        self.results[f"{name1}_vs_{name2}"] = {
            'agent': name1,
            'opponent': name2,
            'wins': wins_1,
            'losses': losses_1,
            'draws': draws_1,
            'total_games': total_games,
            'win_rate': wins_1 / total_games if total_games > 0 else 0,
            'avg_game_length': np.mean(stats1['game_lengths'] + stats2['game_lengths']),
            'avg_decision_time': np.mean(stats1['times_1'] + stats2['times_2'])
        }
        
        # Para name2 (simétrico)
        self.results[f"{name2}_vs_{name1}"] = {
            'agent': name2,
            'opponent': name1,
            'wins': losses_1,
            'losses': wins_1,
            'draws': draws_1,
            'total_games': total_games,
            'win_rate': losses_1 / total_games if total_games > 0 else 0,
            'avg_game_length': np.mean(stats1['game_lengths'] + stats2['game_lengths']),
            'avg_decision_time': np.mean(stats1['times_2'] + stats2['times_1'])
        }
    
    def generate_report(self):
        """Genera reporte completo de resultados"""
        
        # Convertir a DataFrame para análisis
        df = pd.DataFrame([v for v in self.results.values()])
        
        print("=== REPORTE DE EVALUACIÓN DE AGENTES ===\n")
        
        # 1. Ranking general por win rate
        agent_performance = df.groupby('agent').agg({
            'win_rate': 'mean',
            'total_games': 'sum',
            'avg_decision_time': 'mean',
            'avg_game_length': 'mean'
        }).round(3)
        
        agent_performance = agent_performance.sort_values('win_rate', ascending=False)
        
        print("1. RANKING GENERAL (por win rate promedio):")
        print(agent_performance)
        print()
        
        # 2. Matriz de enfrentamientos
        agents = df['agent'].unique()
        win_matrix = np.zeros((len(agents), len(agents)))
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    key = f"{agent1}_vs_{agent2}"
                    if key in self.results:
                        win_matrix[i][j] = self.results[key]['win_rate']
        
        print("2. MATRIZ DE WIN RATES:")
        matrix_df = pd.DataFrame(win_matrix, index=agents, columns=agents)
        print(matrix_df.round(3))
        print()
        
        # 3. Estadísticas por agente
        print("3. ESTADÍSTICAS DETALLADAS:")
        for agent in agents:
            agent_data = df[df['agent'] == agent]
            print(f"\n{agent}:")
            print(f"  Win rate promedio: {agent_data['win_rate'].mean():.3f}")
            print(f"  Mejor enfrentamiento: vs {agent_data.loc[agent_data['win_rate'].idxmax(), 'opponent']} ({agent_data['win_rate'].max():.3f})")
            print(f"  Peor enfrentamiento: vs {agent_data.loc[agent_data['win_rate'].idxmin(), 'opponent']} ({agent_data['win_rate'].min():.3f})")
            print(f"  Tiempo promedio por movimiento: {agent_data['avg_decision_time'].mean():.4f}s")
        
        return df, agent_performance, matrix_df
    
    def plot_results(self, df):
        """Genera visualizaciones de los resultados"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        
        # 1. Win rates por agente
        agent_winrates = df.groupby('agent')['win_rate'].mean().sort_values(ascending=False)
        axes[0].bar(range(len(agent_winrates)), agent_winrates.values)
        axes[0].set_xticks(range(len(agent_winrates)))
        axes[0].set_xticklabels(agent_winrates.index, rotation=45)
        axes[0].set_title('Win Rate Promedio por Agente')
        axes[0].set_ylabel('Win Rate')
        
        # 2. Tiempo de decisión por agente
        agent_times = df.groupby('agent')['avg_decision_time'].mean()
        axes[1].bar(range(len(agent_times)), agent_times.values)
        axes[1].set_xticks(range(len(agent_times)))
        axes[1].set_xticklabels(agent_times.index, rotation=45)
        axes[1].set_title('Tiempo Promedio de Decisión')
        axes[1].set_ylabel('Segundos')
        
    def plot_detailed_evolution(self, agents_to_show=None):
        """Gráfico más detallado de la evolución durante el torneo"""
        
        if agents_to_show is None:
            agents_to_show = list(self.evolution_data.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        
        # 1. Win rate acumulativo (más estable que ventana móvil)
        for agent_name in agents_to_show:
            games = self.evolution_data[agent_name]
            if len(games) < 10:
                continue
                
            cumulative_wins = 0
            cumulative_winrates = []
            game_numbers = []
            
            for i, game in enumerate(games, 1):
                if game['result'] == 1:
                    cumulative_wins += 1
                cumulative_winrates.append(cumulative_wins / i)
                game_numbers.append(i)
            
            axes[0].plot(game_numbers, cumulative_winrates, label=agent_name, linewidth=2)
        
        axes[0].set_xlabel('Número de Partida')
        axes[0].set_ylabel('Win Rate Acumulativo')
        axes[0].set_title('Evolución del Win Rate Acumulativo')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # 2. Distribución de resultados por agente
        results_data = []
        for agent_name in agents_to_show:
            games = self.evolution_data[agent_name]
            wins = sum(1 for g in games if g['result'] == 1)
            draws = sum(1 for g in games if g['result'] == 0)
            losses = sum(1 for g in games if g['result'] == -1)
            
            results_data.append([wins, draws, losses])
        
        results_array = np.array(results_data)
        width = 0.6
        
        axes[1].bar(agents_to_show, results_array[:, 0], width, label='Victorias', color='green', alpha=0.7)
        axes[1].bar(agents_to_show, results_array[:, 1], width, bottom=results_array[:, 0], 
                     label='Empates', color='yellow', alpha=0.7)
        axes[1].bar(agents_to_show, results_array[:, 2], width, 
                     bottom=results_array[:, 0] + results_array[:, 1], 
                     label='Derrotas', color='red', alpha=0.7)
        
        axes[1].set_ylabel('Número de Partidas')
        axes[1].set_title('Distribución de Resultados por Agente')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
