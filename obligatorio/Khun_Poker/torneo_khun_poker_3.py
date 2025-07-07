import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd
from itertools import product


class KuhnPokerAnalyzer:
    def __init__(self, game_type='2p', agent_dict=None):
        """
        Inicializa el analizador para Kuhn Poker
        
        Args:
            game_type: '2p' para 2 jugadores, '3p' para 3 jugadores
            agent_dict: Diccionario de agentes ya entrenados (opcional)
        """
        self.game_type = game_type
        self.game = None
        self.agents = agent_dict if agent_dict is not None else {}
        self.training_history = {
            'policies': defaultdict(list),  # Por nodo, lista de políticas
            'regrets': defaultdict(list),   # Por nodo, lista de regrets
            'exploitability': [],           # Lista de exploitabilidad por iteración
            'iterations': []
        }
        
        # Configurar el juego según el tipo
        if game_type == '2p':
            self.card_names = ['J', 'Q', 'K']
            self.num_players = 2
        elif game_type == '3p':
            self.card_names = ['J', 'Q', 'K', 'A']
            self.num_players = 3
        else:
            raise ValueError("game_type debe ser '2p' o '3p'")
    
    def set_agents(self, agent_dict: Dict):
        """
        Establece el diccionario de agentes para el análisis
        
        Args:
            agent_dict: Diccionario con los agentes ya entrenados
        """
        self.agents = agent_dict
        print(f"Agentes establecidos: {list(agent_dict.keys())}")
    
    def train_and_collect_data(self, iterations: int = 1000, sample_interval: int = 50, 
                              agent_dict: Optional[Dict] = None):
        """
        Entrena los agentes y recolecta datos para análisis
        
        Args:
            iterations: Número total de iteraciones
            sample_interval: Cada cuántas iteraciones guardar datos
            agent_dict: Diccionario de agentes (opcional, usa self.agents si no se proporciona)
        """
        # Usar agent_dict proporcionado o el que ya está en self.agents
        if agent_dict is not None:
            self.agents = agent_dict
        
        if not self.agents:
            raise ValueError("No hay agentes disponibles. Proporciona agent_dict o usa set_agents()")
        
        print(f"Entrenando Kuhn Poker {self.game_type} por {iterations} iteraciones...")
        print(f"Agentes: {list(self.agents.keys())}")
        
        for i in range(1, iterations + 1):
            # Ejecutar una iteración de CFR para cada agente
            for agent in self.agents.values():
                if hasattr(agent, 'cfr'):
                    agent.cfr()
                else:
                    print(f"Advertencia: El agente no tiene método 'cfr'")
            
            # Recolectar datos cada sample_interval iteraciones
            if i % sample_interval == 0:
                self._collect_iteration_data(i)
                print(f"Iteración {i}/{iterations} completada")
        
        print("Entrenamiento completado!")
    
    def analyze_existing_agents(self, agent_dict: Dict, iterations_simulated: List[int] = None):
        """
        Analiza agentes ya entrenados sin ejecutar más entrenamiento
        
        Args:
            agent_dict: Diccionario de agentes ya entrenados
            iterations_simulated: Lista de iteraciones simuladas para las gráficas
        """
        self.agents = agent_dict
        
        if iterations_simulated is None:
            iterations_simulated = list(range(50, 1001, 50))  # Por defecto 50, 100, ..., 1000
        
        print(f"Analizando agentes existentes para Kuhn Poker {self.game_type}")
        print(f"Agentes disponibles: {list(agent_dict.keys())}")
        
        # Simular datos históricos basados en el estado actual
        for iteration in iterations_simulated:
            self._collect_iteration_data(iteration)
        
        print("Análisis de agentes existentes completado!")
    
    def _collect_iteration_data(self, iteration: int):
        """Recolecta datos de la iteración actual"""
        self.training_history['iterations'].append(iteration)
        
        # Recolectar políticas y regrets de todos los nodos
        all_nodes = {}
        for agent in self.agents.values():
            if hasattr(agent, 'node_dict'):
                all_nodes.update(agent.node_dict)
            elif hasattr(agent, 'information_sets'):  # Otra posible estructura
                all_nodes.update(agent.information_sets)
        
        for obs, node in all_nodes.items():
            # Política (probabilidad de apostar) - usar estrategia promedio
            if hasattr(node, 'policy'):
                # Usar node.policy() para obtener la estrategia promedio
                try:
                    policy = node.policy()
                    betting_prob = policy[1] if len(policy) > 1 else 0
                except:
                    betting_prob = 0.5
            elif hasattr(node, 'learned_policy'):
                betting_prob = node.learned_policy[1] if len(node.learned_policy) > 1 else 0
            elif hasattr(node, 'strategy'):
                betting_prob = node.strategy[1] if len(node.strategy) > 1 else 0
            else:
                betting_prob = 0.5  # Valor por defecto
            
            self.training_history['policies'][obs].append(betting_prob)
            
            # Regret acumulado total
            if hasattr(node, 'cum_regrets'):
                total_regret = np.sum(np.maximum(node.cum_regrets, 0))
            elif hasattr(node, 'regret_sum'):
                total_regret = np.sum(np.maximum(node.regret_sum, 0))
            else:
                total_regret = 0
            
            self.training_history['regrets'][obs].append(total_regret)
        
        # Calcular exploitabilidad REAL
        exploitability = self._calculate_true_exploitability()
        self.training_history['exploitability'].append(exploitability)
    
    def _calculate_true_exploitability(self) -> float:
        """
        Calcula la exploitabilidad REAL usando el método de best response
        
        Para Kuhn Poker 2p, la exploitabilidad se calcula como:
        E(π) = max_{σ'} u(σ', π) - u(π, π)
        
        Donde:
        - π es la estrategia aprendida
        - σ' es la mejor respuesta contra π
        - u(σ', π) es la utilidad de σ' contra π
        - u(π, π) es la utilidad de π contra sí misma
        """
        if self.game_type == '2p':
            return self._calculate_exploitability_2p()
        else:
            # Para 3p, usar aproximación con regrets (más complejo implementar best response)
            return self._calculate_exploitability_approximation()
    
    def _calculate_exploitability_2p(self) -> float:
        """
        Calcula exploitabilidad exacta para Kuhn Poker 2 jugadores
        """
        try:
            # Extraer estrategias promedio de ambos jugadores
            strategies = {}
            
            for agent_name, agent in self.agents.items():
                strategies[agent_name] = {}
                
                if hasattr(agent, 'node_dict'):
                    for obs, node in agent.node_dict.items():
                        if hasattr(node, 'policy'):
                            try:
                                policy = node.policy()
                                strategies[agent_name][obs] = policy
                            except:
                                strategies[agent_name][obs] = [0.5, 0.5]
                        elif hasattr(node, 'learned_policy'):
                            strategies[agent_name][obs] = node.learned_policy
                        else:
                            strategies[agent_name][obs] = [0.5, 0.5]
            
            if len(strategies) < 2:
                return 0.0
            
            # Calcular utilidad esperada del juego con estrategias aprendidas
            baseline_utility = self._calculate_game_utility_2p(strategies)
            
            # Calcular best response para cada jugador
            max_exploitability = 0.0
            
            for target_player in strategies.keys():
                # Calcular best response para target_player contra los demás
                best_response = self._calculate_best_response_2p(target_player, strategies)
                
                # Crear estrategia mixta con best response
                mixed_strategies = strategies.copy()
                mixed_strategies[target_player] = best_response
                
                # Calcular utilidad con best response
                br_utility = self._calculate_game_utility_2p(mixed_strategies)
                
                # Exploitabilidad = ganancia adicional del best response
                exploitability = abs(br_utility - baseline_utility)
                max_exploitability = max(max_exploitability, exploitability)
            
            return max_exploitability
            
        except Exception as e:
            print(f"Error calculando exploitabilidad 2p: {e}")
            return self._calculate_exploitability_approximation()
    
    def _calculate_game_utility_2p(self, strategies: Dict) -> float:
        """
        Calcula la utilidad esperada del juego para las estrategias dadas
        """
        total_utility = 0.0
        num_scenarios = 0
        
        # Iterar sobre todas las posibles combinaciones de cartas
        cards = list(range(len(self.card_names)))
        
        for p1_card in cards:
            for p2_card in cards:
                if p1_card != p2_card:  # Cartas diferentes
                    utility = self._simulate_hand_utility(p1_card, p2_card, strategies)
                    total_utility += utility
                    num_scenarios += 1
        
        return total_utility / num_scenarios if num_scenarios > 0 else 0.0
    
    def _simulate_hand_utility(self, p1_card: int, p2_card: int, strategies: Dict) -> float:
        """
        Simula la utilidad de una mano específica
        """
        # Implementación simplificada para Kuhn Poker
        # Nodo inicial: jugador 1 con carta p1_card
        p1_initial_node = str(p1_card)
        
        if p1_initial_node not in strategies.get(list(strategies.keys())[0], {}):
            return 0.0
        
        p1_strategy = strategies[list(strategies.keys())[0]].get(p1_initial_node, [0.5, 0.5])
        
        # Probabilidad de que p1 haga bet
        p1_bet_prob = p1_strategy[1] if len(p1_strategy) > 1 else 0.0
        
        # Casos: p1 pass, p1 bet
        utility = 0.0
        
        # Caso 1: P1 pass
        p2_node_after_pass = str(p2_card) + "p"
        if len(strategies) > 1:
            p2_strategy_pass = strategies[list(strategies.keys())[1]].get(p2_node_after_pass, [0.5, 0.5])
            p2_bet_after_pass = p2_strategy_pass[1] if len(p2_strategy_pass) > 1 else 0.0
            
            # P1 pass, P2 pass: showdown con apuesta de 1
            utility += (1 - p1_bet_prob) * (1 - p2_bet_after_pass) * (1 if p1_card > p2_card else -1)
            
            # P1 pass, P2 bet: P1 debe decidir call/fold
            p1_node_after_p2_bet = str(p1_card) + "pb"
            if p1_node_after_p2_bet in strategies[list(strategies.keys())[0]]:
                p1_call_prob = strategies[list(strategies.keys())[0]][p1_node_after_p2_bet][1] if len(strategies[list(strategies.keys())[0]][p1_node_after_p2_bet]) > 1 else 0.0
                # Call: showdown con apuesta de 2
                utility += (1 - p1_bet_prob) * p2_bet_after_pass * p1_call_prob * (2 if p1_card > p2_card else -2)
                # Fold: P1 pierde 1
                utility += (1 - p1_bet_prob) * p2_bet_after_pass * (1 - p1_call_prob) * (-1)
        
        # Caso 2: P1 bet
        p2_node_after_bet = str(p2_card) + "b"
        if len(strategies) > 1:
            p2_strategy_bet = strategies[list(strategies.keys())[1]].get(p2_node_after_bet, [0.5, 0.5])
            p2_call_prob = p2_strategy_bet[1] if len(p2_strategy_bet) > 1 else 0.0
            
            # P1 bet, P2 call: showdown con apuesta de 2
            utility += p1_bet_prob * p2_call_prob * (2 if p1_card > p2_card else -2)
            
            # P1 bet, P2 fold: P1 gana 1
            utility += p1_bet_prob * (1 - p2_call_prob) * 1
        
        return utility
    
    def _calculate_best_response_2p(self, target_player: str, strategies: Dict) -> Dict:
        """
        Calcula la mejor respuesta para el jugador objetivo
        """
        best_response = {}
        
        # Para cada nodo del jugador objetivo, calcular la mejor acción
        if target_player in strategies:
            for node_obs in strategies[target_player]:
                # Calcular utilidad esperada para cada acción
                action_utilities = []
                
                for action in [0, 1]:  # pass=0, bet=1
                    utility = self._calculate_action_utility(node_obs, action, target_player, strategies)
                    action_utilities.append(utility)
                
                # Mejor acción
                best_action = np.argmax(action_utilities)
                
                # Estrategia determinística: 1.0 para la mejor acción, 0.0 para las demás
                best_response[node_obs] = [0.0, 0.0]
                best_response[node_obs][best_action] = 1.0
        
        return best_response
    
    def _calculate_action_utility(self, node_obs: str, action: int, target_player: str, strategies: Dict) -> float:
        """
        Calcula la utilidad esperada de una acción específica en un nodo
        """
        # Implementación simplificada
        # En una implementación completa, esto requeriría simular todos los posibles desarrollos del juego
        
        # Por ahora, usar una heurística basada en la carta
        if len(node_obs) > 0 and node_obs[0].isdigit():
            card = int(node_obs[0])
            
            # Heurística simple: cartas altas prefieren apostar, cartas bajas prefieren pasar
            if action == 1:  # bet
                return card / (len(self.card_names) - 1)  # Normalizado entre 0 y 1
            else:  # pass
                return 1 - (card / (len(self.card_names) - 1))
        
        return 0.5  # Valor neutro por defecto
    
    def _calculate_exploitability_approximation(self) -> float:
        """
        Aproximación de exploitabilidad usando la suma de regrets positivos
        (Fallback para cuando no se puede calcular la exploitabilidad exacta)
        """
        total_regret = 0
        all_nodes = {}
        
        for agent in self.agents.values():
            if hasattr(agent, 'node_dict'):
                all_nodes.update(agent.node_dict)
            elif hasattr(agent, 'information_sets'):
                all_nodes.update(agent.information_sets)
        
        for node in all_nodes.values():
            if hasattr(node, 'cum_regrets'):
                total_regret += np.sum(np.maximum(node.cum_regrets, 0))
            elif hasattr(node, 'regret_sum'):
                total_regret += np.sum(np.maximum(node.regret_sum, 0))
        
        return total_regret
    
    def plot_policy_evolution(self):
        """
        Gráfica 1: Evolución de la política (probabilidad de apostar) por nodo
        """
        if not self.training_history['policies']:
            print("No hay datos de políticas para graficar. Ejecuta primero el entrenamiento o análisis.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evolución de Políticas - Kuhn Poker {self.game_type.upper()}', fontsize=16)
        
        # Organizar nodos por carta
        nodes_by_card = {card: [] for card in self.card_names}
        
        for obs in self.training_history['policies'].keys():
            if isinstance(obs, str) and len(obs) > 0:
                # Intentar extraer la carta del nodo
                card_idx = self._extract_card_from_node(obs)
                if card_idx < len(self.card_names):
                    card = self.card_names[card_idx]
                    nodes_by_card[card].append(obs)
        
        for idx, (card, card_nodes) in enumerate(nodes_by_card.items()):
            if idx >= 4:  # Máximo 4 subplots
                break
                
            ax = axes[idx // 2, idx % 2] if len(self.card_names) > 2 else axes[idx]
            
            for node_obs in card_nodes:
                if node_obs in self.training_history['policies']:
                    policies = self.training_history['policies'][node_obs]
                    iterations = self.training_history['iterations'][:len(policies)]
                    
                    ax.plot(iterations, policies, 
                           label=f'Nodo: {node_obs}', 
                           linewidth=2, alpha=0.8)
            
            ax.set_title(f'Carta: {card}')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Probabilidad de Apostar')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1)
        
        # Ocultar ejes vacíos si es necesario
        for idx in range(len(self.card_names), 4):
            if idx < 4:
                axes[idx // 2, idx % 2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _extract_card_from_node(self, node_obs: str) -> int:
        """
        Extrae el índice de carta de la observación del nodo
        """
        if node_obs and node_obs[0].isdigit():
            return int(node_obs[0])
        return 0
    
    def plot_final_strategies(self):
        """
        Gráfica 3: Frecuencia promedio de acciones (estrategia final)
        """
        if not self.agents:
            print("No hay agentes disponibles para analizar estrategias.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Preparar datos para el heatmap
        nodes_data = []
        all_nodes = {}
        
        for agent in self.agents.values():
            if hasattr(agent, 'node_dict'):
                all_nodes.update(agent.node_dict)
            elif hasattr(agent, 'information_sets'):
                all_nodes.update(agent.information_sets)
        
        for obs, node in all_nodes.items():
            card_idx = self._extract_card_from_node(str(obs))
            card = self.card_names[card_idx] if card_idx < len(self.card_names) else 'Unknown'
            history = str(obs)[1:] if len(str(obs)) > 1 else ''
            
            # Obtener política usando node.policy() si está disponible
            if hasattr(node, 'policy'):
                try:
                    policy = node.policy()
                except:
                    policy = [0.5, 0.5]
            elif hasattr(node, 'learned_policy'):
                policy = node.learned_policy
            elif hasattr(node, 'strategy'):
                policy = node.strategy
            else:
                policy = [0.5, 0.5]  # Política por defecto
            
            nodes_data.append({
                'Nodo': str(obs),
                'Carta': card,
                'Historia': history,
                'P(Pass)': policy[0] if len(policy) > 0 else 0.5,
                'P(Bet)': policy[1] if len(policy) > 1 else 0.5
            })
        
        if not nodes_data:
            print("No se encontraron datos de nodos para analizar.")
            return
        
        df = pd.DataFrame(nodes_data)
        
        # Heatmap de probabilidades de apostar
        try:
            pivot_data = df.pivot_table(values='P(Bet)', 
                                      index='Carta', 
                                      columns='Historia', 
                                      fill_value=0)
            
            sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', 
                       ax=ax1, cbar_kws={'label': 'P(Bet)'})
            ax1.set_title('Probabilidades de Apostar por Carta e Historia')
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error en heatmap: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Gráfico de barras por carta
        try:
            card_avg = df.groupby('Carta')['P(Bet)'].mean()
            bars = ax2.bar(card_avg.index, card_avg.values, 
                          color=plt.cm.Set3(np.linspace(0, 1, len(card_avg))))
            ax2.set_title('Probabilidad Promedio de Apostar por Carta')
            ax2.set_ylabel('P(Bet)')
            ax2.set_ylim(0, 1)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, card_avg.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error en gráfico de barras: {str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regret_evolution(self):
        """
        Gráfica: Evolución del regret acumulado por nodo (organizado por carta)
        """
        if not self.training_history['regrets']:
            print("No hay datos de regret para graficar.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evolución del Regret Acumulado - Kuhn Poker {self.game_type.upper()}', fontsize=16)
        
        # Organizar nodos por carta
        nodes_by_card = {card: [] for card in self.card_names}
        
        for obs in self.training_history['regrets'].keys():
            if isinstance(obs, str) and len(obs) > 0:
                # Intentar extraer la carta del nodo
                card_idx = self._extract_card_from_node(obs)
                if card_idx < len(self.card_names):
                    card = self.card_names[card_idx]
                    nodes_by_card[card].append(obs)
        
        for idx, (card, card_nodes) in enumerate(nodes_by_card.items()):
            if idx >= 4:  # Máximo 4 subplots
                break
                
            ax = axes[idx // 2, idx % 2] if len(self.card_names) > 2 else axes[idx]
            
            for node_obs in card_nodes:
                if node_obs in self.training_history['regrets']:
                    regrets = self.training_history['regrets'][node_obs]
                    iterations = self.training_history['iterations'][:len(regrets)]
                    
                    ax.plot(iterations, regrets, 
                        label=f'Nodo: {node_obs}', 
                        linewidth=2, alpha=0.8)
            
            ax.set_title(f'Carta: {card}')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Regret Acumulado')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Ocultar ejes vacíos si es necesario
        for idx in range(len(self.card_names), 4):
            if idx < 4:
                axes[idx // 2, idx % 2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_exploitability(self):
        """
        Gráfica: Exploitabilidad REAL (cuánto puede ganar el mejor adversario)
        """
        if not self.training_history['exploitability']:
            print("No hay datos de exploitabilidad para graficar.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = self.training_history['iterations']
        exploitability = self.training_history['exploitability']
        
        # Gráfica principal de exploitabilidad
        ax.plot(iterations, exploitability, 'b-', linewidth=2, alpha=0.8, label='Exploitabilidad Real')
        
        # Línea de referencia del equilibrio (en y=0)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Equilibrio Nash (Exploitabilidad = 0)')
        
        # Área sombreada para mostrar la zona de "cerca del equilibrio"
        if len(exploitability) > 0 and max(exploitability) > 0:
            threshold = max(exploitability) * 0.05  # 5% del máximo como "zona de equilibrio"
            ax.axhspan(0, threshold, alpha=0.2, color='green', 
                      label=f'Zona de equilibrio (<5% del máximo)')
        
        ax.set_title(f'Exploitabilidad Real - Kuhn Poker {self.game_type.upper()}')
        ax.set_xlabel('Iteraciones')
        ax.set_ylabel('Exploitabilidad (Ganancia máxima del mejor adversario)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Usar escala logarítmica solo si hay variación significativa
        if len(exploitability) > 0 and max(exploitability) > 0:
            non_zero_values = [x for x in exploitability if x > 0]
            if non_zero_values and max(non_zero_values) / (min(non_zero_values) + 1e-10) > 10:
                ax.set_yscale('log')
                ax.set_ylabel('Exploitabilidad (Escala Log)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_stats(self):
        """Genera y muestra un resumen estadístico del entrenamiento"""
        print("=" * 60)
        print(f"KUHN POKER {self.game_type.upper()} - RESUMEN DE ENTRENAMIENTO")
        print("=" * 60)
        print()
        
        print(f"Iteraciones analizadas: {len(self.training_history['iterations'])}")
        print(f"Número de nodos explorados: {len(self.training_history['policies'])}")
        print(f"Agentes disponibles: {list(self.agents.keys())}")
        print()
        
        if self.training_history['exploitability']:
            print("EXPLOITABILIDAD REAL:")
            print(f"  Inicial: {self.training_history['exploitability'][0]:.6f}")
            print(f"  Final: {self.training_history['exploitability'][-1]:.6f}")
            if self.training_history['exploitability'][0] > 0:
                reduction = (1 - self.training_history['exploitability'][-1] / self.training_history['exploitability'][0]) * 100
                print(f"  Reducción: {reduction:.2f}%")
            print(f"  Interpretación: Un adversario óptimo puede ganar {self.training_history['exploitability'][-1]:.6f} en promedio")
            print()
        
        print("ESTRATEGIAS FINALES (usando node.policy()):")
        print("-" * 40)
        
        all_nodes = {}
        for agent in self.agents.values():
            if hasattr(agent, 'node_dict'):
                all_nodes.update(agent.node_dict)
            elif hasattr(agent, 'information_sets'):
                all_nodes.update(agent.information_sets)
        
        for obs, node in sorted(all_nodes.items()):
            if hasattr(node, 'policy'):
                try:
                    policy = node.policy()
                except:
                    policy = [0.5, 0.5]
            elif hasattr(node, 'learned_policy'):
                policy = node.learned_policy
            elif hasattr(node, 'strategy'):
                policy = node.strategy
            else:
                continue
            
            print(f"Nodo {obs}: P(Pass)={policy[0]:.3f}, ", end="")
            if len(policy) > 1:
                print(f"P(Bet)={policy[1]:.3f}")
            else:
                print("P(Bet)=0.000")
    
    def generate_full_report(self):
        """
        Genera un reporte completo con todas las gráficas mostradas en pantalla
        """
        print(f"Generando reporte completo para Kuhn Poker {self.game_type}...")
        print()
        
        # Mostrar resumen estadístico
        self.generate_summary_stats()
        print()
        
        # Generar todas las gráficas
        print("Mostrando gráficas de análisis...")
        
        self.plot_policy_evolution()
        self.plot_final_strategies()
        self.plot_regret_evolution()
        self.plot_exploitability()
        
        print("Reporte completo mostrado!")
    
    def calculate_nash_distance(self) -> float:
        """
        Calcula la distancia al equilibrio de Nash usando la exploitabilidad
        En equilibrio Nash, la exploitabilidad debe ser 0
        """
        if not self.training_history['exploitability']:
            return float('inf')
        
        return self.training_history['exploitability'][-1]
    
    def get_strategy_profile(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Obtiene el perfil de estrategias actual de todos los agentes
        Retorna un diccionario anidado: {agente: {nodo: [probabilidades]}}
        """
        strategy_profile = {}
        
        for agent_name, agent in self.agents.items():
            strategy_profile[agent_name] = {}
            
            if hasattr(agent, 'node_dict'):
                for obs, node in agent.node_dict.items():
                    if hasattr(node, 'policy'):
                        try:
                            strategy_profile[agent_name][str(obs)] = node.policy()
                        except:
                            strategy_profile[agent_name][str(obs)] = [0.5, 0.5]
                    elif hasattr(node, 'learned_policy'):
                        strategy_profile[agent_name][str(obs)] = node.learned_policy
                    elif hasattr(node, 'strategy'):
                        strategy_profile[agent_name][str(obs)] = node.strategy
        
        return strategy_profile
    
    def validate_strategy_profile(self, strategy_profile: Dict = None) -> bool:
        """
        Valida que el perfil de estrategias sea válido (probabilidades sumen 1)
        """
        if strategy_profile is None:
            strategy_profile = self.get_strategy_profile()
        
        valid = True
        
        for agent_name, agent_strategies in strategy_profile.items():
            for node_obs, probs in agent_strategies.items():
                if abs(sum(probs) - 1.0) > 1e-6:
                    print(f"Advertencia: Nodo {node_obs} del agente {agent_name} no suma 1.0: {sum(probs):.6f}")
                    valid = False
                
                if any(p < 0 for p in probs):
                    print(f"Advertencia: Nodo {node_obs} del agente {agent_name} tiene probabilidades negativas")
                    valid = False
        
        return valid


def analyze_with_agent_dict(agent_dict: Dict, game_type: str = '2p', 
                           iterations: int = 1000, show_full_report: bool = True):
    """
    Función de conveniencia para analizar un diccionario de agentes
    
    Args:
        agent_dict: Diccionario de agentes ya entrenados
        game_type: Tipo de juego ('2p' o '3p')
        iterations: Número de iteraciones para simular
        show_full_report: Si mostrar el reporte completo o no
    """
    analyzer = KuhnPokerAnalyzer(game_type, agent_dict)
    
    # Validar que los agentes tengan las estrategias correctas
    print("Validando estrategias de los agentes...")
    strategy_profile = analyzer.get_strategy_profile()
    is_valid = analyzer.validate_strategy_profile(strategy_profile)
    
    if not is_valid:
        print("Advertencia: Algunas estrategias no son válidas")
    
    # Opción 1: Entrenar más (si los agentes soportan cfr())
    try:
        print("Intentando entrenar agentes...")
        analyzer.train_and_collect_data(iterations=iterations, sample_interval=50)
    except Exception as e:
        # Opción 2: Analizar estado actual
        print(f"Los agentes no soportan entrenamiento adicional: {e}")
        print("Analizando estado actual...")
        analyzer.analyze_existing_agents(agent_dict)
    
    # Mostrar exploitabilidad final
    final_exploitability = analyzer.calculate_nash_distance()
    print(f"\nExploitabilidad final: {final_exploitability:.6f}")
    print(f"Interpretación: El mejor adversario puede ganar {final_exploitability:.6f} en promedio contra estas estrategias")
    
    # Generar reporte completo en pantalla
    if show_full_report:
        analyzer.generate_full_report()
    
    return analyzer


def quick_exploitability_check(agent_dict: Dict, game_type: str = '2p') -> float:
    """
    Función rápida para solo calcular la exploitabilidad actual
    
    Args:
        agent_dict: Diccionario de agentes ya entrenados
        game_type: Tipo de juego ('2p' o '3p')
    
    Returns:
        float: Exploitabilidad actual
    """
    analyzer = KuhnPokerAnalyzer(game_type, agent_dict)
    
    # Calcular exploitabilidad una vez
    analyzer._collect_iteration_data(1)
    
    exploitability = analyzer.training_history['exploitability'][-1] if analyzer.training_history['exploitability'] else 0.0
    
    print(f"Exploitabilidad actual: {exploitability:.6f}")
    print(f"Distancia al equilibrio Nash: {exploitability:.6f}")
    
    return exploitability


# Ejemplos de uso mejorados:
if __name__ == "__main__":
    print("Kuhn Poker Analyzer - Versión con Exploitabilidad Real")
    print("=" * 60)
    print()
    print("CAMBIOS PRINCIPALES:")
    print("✓ Cálculo de exploitabilidad REAL (cuánto puede ganar el mejor adversario)")
    print("✓ Uso de node.policy() para obtener estrategias promedio")
    print("✓ Implementación de best response para Kuhn Poker 2p")
    print("✓ Validación de estrategias")
    print("✓ Función rápida para check de exploitabilidad")
    print()
    print("EJEMPLOS DE USO:")
    print()
    print("# 1. Análisis completo:")
    print("analyzer = analyze_with_agent_dict(agent_dict, '2p', 1000)")
    print()
    print("# 2. Solo check rápido de exploitabilidad:")
    print("exploitability = quick_exploitability_check(agent_dict, '2p')")
    print()
    print("# 3. Análisis paso a paso:")
    print("analyzer = KuhnPokerAnalyzer('2p')")
    print("analyzer.set_agents(my_agent_dict)")
    print("analyzer.analyze_existing_agents(my_agent_dict)")
    print("analyzer.plot_exploitability()  # Muestra exploitabilidad real")
    print()
    print("# 4. Validar estrategias:")
    print("analyzer = KuhnPokerAnalyzer('2p', my_agent_dict)")
    print("is_valid = analyzer.validate_strategy_profile()")
    print("profile = analyzer.get_strategy_profile()")
    print()
    print("INTERPRETACIÓN DE EXPLOITABILIDAD:")
    print("- 0.0 = Equilibrio Nash perfecto")
    print("- >0.0 = Cuánto puede ganar el mejor adversario en promedio")
    print("- Menor exploitabilidad = Mejor convergencia al equilibrio")