import numpy as np
from typing import TypeAlias, Self

BLACK = 0
WHITE = 1
EMPTY = -1
BLACK_START = 1
WHITE_START = 6
BLACK_GOAL = 7
WHITE_GOAL = 0
ROWS = 8
COLS = 5
MAX_STACK = 3
MOVES = ["N", "S", "W", "E", "SW", "NW", "SE", "NE"]

Player: TypeAlias = int
Action: TypeAlias = tuple[int, int, str]
Coords: TypeAlias = tuple[int, int]

class Board:

    def __init__(self):
        self.squares = np.ndarray((ROWS, COLS, MAX_STACK))
        self.squares.fill(EMPTY)
        for y in range(COLS):
            self.squares[BLACK_START][y][0] = BLACK
            self.squares[WHITE_START][y][0] = WHITE

    @staticmethod
    def _opponent(player: Player) -> Player:
        if player == BLACK:
            return WHITE
        return BLACK

    def _check_player_blocked(self, player: Player) -> bool:
        player_squares = np.argwhere(self.squares == player)
        for x, y, k in player_squares:
            stack = self.squares[x][y]
            opponent_pieces = np.argwhere(stack == Board._opponent(player)).tolist()
            player_blocked = len(opponent_pieces) != 0 and any(
                h > k for h in opponent_pieces
            )
            if not player_blocked:
                return False
        return True

    @staticmethod
    def _map_action_to_new_pos(action: Action) -> Coords:
        (x, y, move) = action
        match move:
            case "N":
                return x - 1, y
            case "S":
                return x + 1, y
            case "E":
                return x, y + 1
            case "W":
                return x, y - 1
            case "NE":
                return x - 1, y + 1
            case "NW":
                return x - 1, y - 1
            case "SE":
                return x + 1, y + 1
            case "SW":
                return x + 1, y - 1

    def check_game_over(self) -> bool:
        return self.check_for_winner() is not None

    def check_for_winner(self) -> Player:
        # check if a white piece reached the goal or if all black pieces are blocked
        if any(
            x == WHITE for x in self.squares[WHITE_GOAL][:].T[0]
        ) or self._check_player_blocked(BLACK):
            return WHITE
        # check if a black piece reached the goal or if all white pieces are blocked
        elif any(
            x == BLACK for x in self.squares[BLACK_GOAL][:].T[0]
        ) or self._check_player_blocked(WHITE):
            return BLACK
        else:
            return None

    def play_turn(self, player: Player, action: Action) -> None:
        (x, y, _) = action
        # take the highest player piece off the tower
        if self.squares[x][y][2] == player:
            self.squares[x][y][2] = EMPTY
        elif self.squares[x][y][1] == player:
            self.squares[x][y][1] = EMPTY
        else:
            self.squares[x][y][0] = EMPTY

        # put the piece in the correct square
        new_x, new_y = Board._map_action_to_new_pos(action)
        if self.squares[new_x][new_y][0] == EMPTY:
            self.squares[new_x][new_y][0] = player
        elif self.squares[new_x][new_y][1] == EMPTY:
            self.squares[new_x][new_y][1] = player
        else:
            self.squares[new_x][new_y][2] = player

    def legal_moves(self, player: Player) -> list[Action]:
        legal_moves = []
        player_squares = np.argwhere(self.squares == player)
        for square in player_squares:
            for move in MOVES:
                action = (square[0], square[1], move)
                is_legal_move, _ = self.is_legal_move(player, action)
                if is_legal_move:
                    legal_moves.append(action)
        return legal_moves

    def is_legal_move(self, player: Player, action: Action) -> tuple[bool, str]:
        (x, y, move) = action
        # check if there is a piece in position x, y
        stack = self.squares[x][y]
        if all(x != player for x in stack):
            return (False, f"There are no player pieces in position ({x},{y})")
        # check if the piece is blocked
        player_squares = np.argwhere(stack == player).tolist()
        opponent_squares = np.argwhere(stack == Board._opponent(player)).tolist()
        if player_squares != []:
            max_pos = max(player_squares)
            if any(h > max_pos for h in opponent_squares):
                return (
                    False,
                    f"Player pieces in position ({x}, {y}) are blocked by an opponent piece",
                )
        # check if move is legal
        if x in [WHITE_GOAL, BLACK_GOAL]:
            return (False, "Game already over")
        if (x == BLACK_START and player == BLACK and move in ["N", "NW", "NE"]) or (
            x == WHITE_START and player == WHITE and move in ["S", "SW", "SE"]
        ):
            return (False, "Cannot move into your own goal")
        if (y == 0 and move in ["W", "NW", "SW"]) or (
            y == 4 and move in ["E", "NE", "SE"]
        ):
            return (False, "Cannot move out of bounds")
        # missing check for destination tower height
        return (True, "Legal move")

    def set_board(self, board: Self) -> None:
        self.squares = np.copy(board.squares)
    
    def render(self):
        # rendering a stack of pieces
        def stack_to_str(pieces) -> str:
            s = []
            for h in range(MAX_STACK):
                s += '_' if pieces[h] == -1 else str(int(pieces[h]))
            return ''.join(s) + ' '
        # rendering the whole board
        for x in range(ROWS):
            print(f"{x}: ", end="")
            for y in range(COLS):
                print(stack_to_str(self.squares[x, y, :]), end="")
            print()


    #Agrego funcion para el calculo de la distancia a la meta a ser utilizada por la funcion Eval
    def dist_to_opponent_home(self, player: Player, dis_type='Vertical') -> int:
        """
        Calcula la menor distancia de cualquier ficha NO BLOQUEADA del jugador a la meta rival.
        
        Args:
            player: El jugador (BLACK o WHITE)
            dis_type: 'Vertical' para distancia vertical directa, 'Manhattan' para distancia Manhattan
        
        Returns:
            int: La menor distancia encontrada, o 100 si no hay fichas disponibles
        """
        player_squares = np.argwhere(self.squares == player)
        
        # Determinar la fila objetivo
        if player == BLACK:
            goal_row = WHITE_GOAL  # 0
        else:
            goal_row = BLACK_GOAL  # 7
        
        min_dist = float('inf')
        
        for x, y, h in player_squares:
            # Verificar si esta ficha específica está bloqueada
            stack = self.squares[x][y]
            
            # Buscar fichas oponentes en posiciones SUPERIORES a esta ficha
            is_blocked = False
            for stack_level in range(h + 1, MAX_STACK):  # Solo niveles superiores
                if stack[stack_level] == self._opponent(player):
                    is_blocked = True
                    break
            
            if not is_blocked:
                if dis_type == 'Vertical':
                    # Distancia vertical directa (solo filas)
                    dist = abs(x - goal_row)
                else:  # Manhattan simplificado
                    # Distancia vertical + penalidad si la columna tiene obstáculos
                    row_dist = abs(x - goal_row)
                    col_penalty = 0
                    
                    # Verificar si hay obstáculos (stacks llenos) en la columna hacia la meta
                    if x != goal_row:
                        # Determinar el rango de filas en el camino hacia la meta
                        if player == BLACK:  # BLACK va hacia fila 0
                            path_rows = range(x - 1, goal_row - 1, -1)  # De x-1 hasta 0
                        else:  # WHITE va hacia fila 7
                            path_rows = range(x + 1, goal_row + 1)  # De x+1 hasta 7
                        
                        # Verificar si hay al menos un obstáculo en la columna
                        for check_row in path_rows:
                            stack_at_pos = self.squares[check_row][y]
                            stack_count = sum(1 for piece in stack_at_pos if piece != EMPTY)
                            if stack_count >= MAX_STACK:
                                col_penalty = 1  # Penalidad fija por columna bloqueada
                                break
                    
                    dist = row_dist + col_penalty
                
                min_dist = min(min_dist, dist)
        
        return int(min_dist)
  

    #Agrego funcion para contar fichas bloqueadas al oponente
    def count_blocked_pieces(self, player: Player) -> int:
        """
        Cuenta cuántas fichas del jugador están completamente bloqueadas.
        Una ficha está bloqueada si hay AL MENOS una ficha oponente 
        en cualquier posición superior del stack.
        """
        player_squares = np.argwhere(self.squares == player)
        blocked_count = 0
        
        for x, y, h in player_squares:
            stack = self.squares[x][y]
            
            # Verificar si hay AL MENOS una ficha oponente en posiciones superiores
            opponent_above = False
            for level in range(h + 1, MAX_STACK):
                if stack[level] == self._opponent(player):
                    opponent_above = True
                    break
            
            if opponent_above:
                blocked_count += 1
        
        return blocked_count
  


        