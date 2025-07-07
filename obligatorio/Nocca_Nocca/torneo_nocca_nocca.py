import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from collections import defaultdict
from itertools import combinations
import pandas as pd
from games.nocca_nocca.nocca_nocca import NoccaNocca

def play_single_match(agent1, agent2, agent1_name, agent2_name, game_config, first_player_idx=0):
    """
    Juega una partida entre dos agentes.
    
    Args:
        agent1, agent2: Instancias de los agentes
        agent1_name, agent2_name: Nombres de los agentes
        game_config: Configuración del juego (alfa_D, alfa_B, dis_type)
        first_player_idx: 0 si agent1 va primero, 1 si agent2 va primero
    
    Returns:
        tuple: (ganador_name, tiempo_agent1, tiempo_agent2, error_msg)
        ganador_name: nombre del agente ganador, None si empate
    """
    try:
        # Crear el juego con la configuración especificada
        game = NoccaNocca(
            initial_player=first_player_idx,
            max_steps=150,  # Límite de pasos para evitar juegos infinitos
            alfa_D=game_config['alfa_D'],
            alfa_B=game_config['alfa_B'],
            dis_type=game_config['dis_type']
        )
        
        # Resetear el juego
        game.reset()
        
        # Configurar agentes con nombres del juego
        agents = [agent1, agent2]
        agent_names = ["Black", "White"]
        real_agent_names = [agent1_name, agent2_name]
        
        # Asignar agentes a los juegos
        for i, agent in enumerate(agents):
            agent.game = game
            agent.agent = agent_names[i]
        
        # Tiempos de decisión
        decision_times = [0.0, 0.0]
        
        # Jugar la partida
        while not game.terminated():
            current_game_agent = game.agent_selection
            current_agent_idx = 0 if current_game_agent == "Black" else 1
            current_agent = agents[current_agent_idx]
            
            # Medir tiempo de decisión
            start_time = time.time()
            action = current_agent.action()
            end_time = time.time()
            
            decision_times[current_agent_idx] += (end_time - start_time)
            
            # Ejecutar acción
            game.step(action)
        
        # Determinar ganador usando nombres reales
        winner = game.check_for_winner()
        if winner == "Black":
            winner_name = real_agent_names[0]  # agent1_name
        elif winner == "White":
            winner_name = real_agent_names[1]  # agent2_name
        else:
            winner_name = None  # Empate
        
        return winner_name, decision_times[0], decision_times[1], None
        
    except Exception as e:
        return None, 0.0, 0.0, str(e)

def evaluate_agents_single_config(agents_dict, game_config, n_matches=10):
    """
    Evalúa todos los agentes para una configuración específica del juego.
    
    Args:
        agents_dict: Diccionario {nombre_agente: instancia_agente}
        game_config: Configuración del juego
        n_matches: Número de partidas por par de agentes
    
    Returns:
        dict: Resultados de la evaluación
    """
    agent_names = list(agents_dict.keys())
    n_agents = len(agent_names)
    
    # Inicializar métricas
    wins = defaultdict(int)
    total_games = defaultdict(int)
    decision_times = defaultdict(list)
    win_rates_evolution = defaultdict(list)
    match_numbers = defaultdict(list)
    
    print(f"Evaluando configuración: alfa_D={game_config['alfa_D']}, alfa_B={game_config['alfa_B']}, dis_type={game_config['dis_type']}")
    
    # Jugar entre todos los pares de agentes
    for i, agent1_name in enumerate(agent_names):
        for j, agent2_name in enumerate(agent_names):
            if i >= j:  # Evitar duplicados y auto-juegos
                continue
                
            print(f"  Jugando: {agent1_name} vs {agent2_name}")
            
            # Jugar n_matches partidas
            for match in range(n_matches):
                # Alternar quién va primero
                first_player = match % 2
                
                # Jugar la partida
                winner_name, time1, time2, error = play_single_match(
                    agents_dict[agent1_name], 
                    agents_dict[agent2_name],
                    agent1_name,
                    agent2_name,
                    game_config, 
                    first_player
                )
                
                if error:
                    print(f"    ERROR: Partida {match+1} falló - {error}")
                    continue
                
                # Actualizar métricas
                total_games[agent1_name] += 1
                total_games[agent2_name] += 1
                
                if winner_name == agent1_name:
                    wins[agent1_name] += 1
                elif winner_name == agent2_name:
                    wins[agent2_name] += 1
                # Si winner_name es None, es empate (no se suma a ninguno)
                
                # Tiempos de decisión
                decision_times[agent1_name].append(time1)
                decision_times[agent2_name].append(time2)
                
                # Evolución del win rate
                for agent_name in [agent1_name, agent2_name]:
                    current_win_rate = wins[agent_name] / total_games[agent_name] if total_games[agent_name] > 0 else 0
                    win_rates_evolution[agent_name].append(current_win_rate)
                    match_numbers[agent_name].append(total_games[agent_name])
    
    # Calcular estadísticas finales
    final_win_rates = {}
    avg_decision_times = {}
    
    for agent_name in agent_names:
        final_win_rates[agent_name] = wins[agent_name] / total_games[agent_name] if total_games[agent_name] > 0 else 0
        avg_decision_times[agent_name] = np.mean(decision_times[agent_name]) if decision_times[agent_name] else 0
    
    return {
        'win_rates': final_win_rates,
        'avg_decision_times': avg_decision_times,
        'win_rates_evolution': dict(win_rates_evolution),
        'match_numbers': dict(match_numbers),
        'total_games': dict(total_games)
    }

def plot_level1_results(results, config_name):
    """
    Genera gráficos para el Nivel 1 (competencias individuales).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Nivel 1 - Competencia Individual: {config_name}', fontsize=16)
    
    agent_names = list(results['win_rates'].keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    
    # 1. Evolución del win rate
    ax1 = axes[0]
    for i, agent_name in enumerate(agent_names):
        evolution = results['win_rates_evolution'][agent_name]
        matches = results['match_numbers'][agent_name]
        if evolution and matches:
            ax1.plot(matches, evolution, marker='o', label=agent_name, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Número de partidas')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Evolución del Win Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Win rates promedio (barras)
    ax2 = axes[1]
    win_rates = [results['win_rates'][agent] for agent in agent_names]
    bars = ax2.bar(agent_names, win_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Win Rate Promedio')
    ax2.set_title('Win Rates Promedio por Agente')
    ax2.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Rotar nombres de agentes si son largos
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Tiempos de decisión promedio (barras)
    ax3 = axes[2]
    decision_times = [results['avg_decision_times'][agent] for agent in agent_names]
    bars = ax3.bar(agent_names, decision_times, color=colors, alpha=0.8)
    ax3.set_ylabel('Tiempo Promedio (segundos)')
    ax3.set_title('Tiempos de Decisión Promedio')
    
    # Añadir valores en las barras
    for bar, time_val in zip(bars, decision_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Rotar nombres de agentes si son largos
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_level2_results(all_results):
    """
    Genera gráficos para el Nivel 2 (torneo general).
    """
    config_names = list(all_results.keys())
    
    if not config_names:
        print("No hay resultados para mostrar en el Nivel 2")
        return
    
    # Obtener nombres de agentes
    agent_names = list(all_results[config_names[0]]['win_rates'].keys())
    
    # Preparar datos para gráficos
    win_rates_data = []
    decision_times_data = []
    
    for config_name in config_names:
        results = all_results[config_name]
        for agent_name in agent_names:
            win_rates_data.append({
                'Configuración': config_name,
                'Agente': agent_name,
                'Win Rate': results['win_rates'][agent_name]
            })
            decision_times_data.append({
                'Configuración': config_name,
                'Agente': agent_name,
                'Tiempo': results['avg_decision_times'][agent_name]
            })
    
    # Crear DataFrames
    df_win_rates = pd.DataFrame(win_rates_data)
    df_decision_times = pd.DataFrame(decision_times_data)
    
    # Crear gráficos
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Nivel 2 - Torneo General: Comparación entre Funciones de Evaluación', fontsize=16)
    
    # 1. Win rates acumulados
    ax1 = axes[0]
    sns.barplot(data=df_win_rates, x='Configuración', y='Win Rate', hue='Agente', ax=ax1)
    ax1.set_title('Win Rates Acumulados por Configuración')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Tiempos de decisión promedio
    ax2 = axes[1]
    sns.barplot(data=df_decision_times, x='Configuración', y='Tiempo', hue='Agente', ax=ax2)
    ax2.set_title('Tiempos de Decisión Promedio por Configuración')
    ax2.set_ylabel('Tiempo Promedio (segundos)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def evaluate_agents_tournament(agents_dict, evaluation_configs=None, n_matches=10):
    """
    Función principal para evaluar agentes en un torneo completo.
    
    Args:
        agents_dict: Diccionario {nombre_agente: instancia_agente}
        evaluation_configs: Lista de configuraciones del juego. Si None, usa configuraciones por defecto.
        n_matches: Número de partidas por par de agentes en cada configuración
    
    Returns:
        dict: Resultados completos del torneo
    """
    # Configuraciones por defecto si no se proporcionan
    if evaluation_configs is None:
        evaluation_configs = [
            {'alfa_D': 10, 'alfa_B': 5, 'dis_type': 'Vertical', 'name': 'Vertical_10_5'},
            {'alfa_D': 10, 'alfa_B': 5, 'dis_type': 'Manhattan', 'name': 'Manhattan_10_5'},
            {'alfa_D': 5, 'alfa_B': 10, 'dis_type': 'Vertical', 'name': 'Vertical_5_10'},
            {'alfa_D': 20, 'alfa_B': 2, 'dis_type': 'Vertical', 'name': 'Vertical_20_2'},
        ]
    
    print(f"=== INICIO DEL TORNEO ===")
    print(f"Agentes participantes: {list(agents_dict.keys())}")
    print(f"Configuraciones de evaluación: {len(evaluation_configs)}")
    print(f"Partidas por par de agentes: {n_matches}")
    print("=" * 50)
    
    all_results = {}
    
    # Evaluar cada configuración
    for config in evaluation_configs:
        config_name = config['name']
        print(f"\n--- EVALUANDO CONFIGURACIÓN: {config_name} ---")
        
        # Ejecutar evaluación para esta configuración
        results = evaluate_agents_single_config(agents_dict, config, n_matches)
        all_results[config_name] = results
        
        # Mostrar gráficos de Nivel 1
        plot_level1_results(results, config_name)
        
        # Mostrar resumen textual
        print(f"\nResumen de {config_name}:")
        for agent_name in results['win_rates']:
            print(f"  {agent_name}: Win Rate = {results['win_rates'][agent_name]:.3f}, "
                  f"Tiempo promedio = {results['avg_decision_times'][agent_name]:.4f}s")
    
    # Mostrar gráficos de Nivel 2
    print(f"\n--- GRÁFICOS DE NIVEL 2: TORNEO GENERAL ---")
    plot_level2_results(all_results)
    
    # Resumen final
    print(f"\n=== RESUMEN FINAL DEL TORNEO ===")
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        sorted_agents = sorted(results['win_rates'].items(), key=lambda x: x[1], reverse=True)
        for i, (agent_name, win_rate) in enumerate(sorted_agents, 1):
            print(f"  {i}. {agent_name}: {win_rate:.3f} ({results['total_games'][agent_name]} partidas)")
    
    return all_results

# Función de conveniencia para uso rápido
def quick_evaluation(agents_dict, n_matches=10):
    """
    Evaluación rápida con configuraciones predeterminadas.
    
    Args:
        agents_dict: Diccionario {nombre_agente: instancia_agente}
        n_matches: Número de partidas por par de agentes
    
    Returns:
        dict: Resultados del torneo
    """
    return evaluate_agents_tournament(agents_dict, n_matches=n_matches)