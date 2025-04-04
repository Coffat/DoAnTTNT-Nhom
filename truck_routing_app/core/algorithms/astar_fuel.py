"""
A* search algorithm implementation with fuel constraints.
"""

from typing import List, Tuple, Dict, Set
import numpy as np
from .base_search import BaseSearch, SearchState
from queue import PriorityQueue

class AStarFuel(BaseSearch):
    """A* search algorithm implementation with fuel constraints."""
    
    def __init__(self, grid: np.ndarray):
        """Initialize A* with fuel constraints."""
        super().__init__(grid)
        self.open_set = PriorityQueue()
        self.closed_set = set()
        self.g_score = {}  # Chi phí thực tế từ điểm bắt đầu đến điểm hiện tại
        self.f_score = {}  # Ước tính tổng chi phí (g_score + heuristic)
        self.parent = {}
        self.start = None
        self.goal = None
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], fuel: float) -> float:
        """Tính hàm heuristic (Manhattan distance + ước tính chi phí nhiên liệu)."""
        x1, y1 = pos
        x2, y2 = goal
        distance = float(abs(x1 - x2) + abs(y1 - y2))
        
        # Ước tính chi phí nhiên liệu cần thiết
        fuel_cost = max(0, distance * self.FUEL_PER_MOVE - fuel)
        
        return distance + fuel_cost
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute A* with fuel constraints from start to goal."""
        self.start = start
        self.goal = goal
        self.open_set = PriorityQueue()
        self.closed_set.clear()
        self.g_score.clear()
        self.f_score.clear()
        self.parent.clear()
        self.visited.clear()
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Tạo trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Khởi tạo các giá trị ban đầu
        self.g_score[initial_state.get_state_key()] = 0
        self.f_score[initial_state.get_state_key()] = self.heuristic(start, goal, self.MAX_FUEL)
        self.open_set.put((self.f_score[initial_state.get_state_key()], initial_state))
        self.visited.append(start)
        self.current_position = start
        self.current_path = [start]
        
        while not self.open_set.empty():
            self.steps += 1
            current_state = self.open_set.get()[1]
            self.current_position = current_state.position
            
            if current_state.position == goal:
                # Kiểm tra tính khả thi của đường đi
                is_feasible, reason = self.is_path_feasible(current_state.path, self.MAX_FUEL)
                if is_feasible:
                    self.current_path = current_state.path
                    self.path_length = len(self.current_path) - 1
                    self.cost = current_state.total_cost
                    self.current_fuel = current_state.fuel
                    self.current_total_cost = current_state.total_cost
                    self.current_fuel_cost = current_state.total_cost - self.current_toll_cost
                    return self.current_path
                else:
                    continue
            
            # Thêm trạng thái hiện tại vào tập đóng
            self.closed_set.add(current_state.get_state_key())
            
            # Xử lý các trạng thái lân cận
            for next_pos in self.get_neighbors(current_state.position):
                # Tính toán chi phí và nhiên liệu cho bước tiếp theo
                new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                
                # Tạo trạng thái mới
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=current_state.total_cost + move_cost,
                    path=current_state.path + [next_pos],
                    visited_gas_stations=current_state.visited_gas_stations.copy(),
                    toll_stations_visited=current_state.toll_stations_visited.copy()
                )
                
                # Cập nhật các tập đã thăm
                if next_pos in self.grid and self.grid[next_pos[1]][next_pos[0]] == 'G':
                    new_state.visited_gas_stations.add(next_pos)
                elif next_pos in self.grid and self.grid[next_pos[1]][next_pos[0]] == 'T':
                    new_state.toll_stations_visited.add(next_pos)
                
                # Kiểm tra tính khả thi của trạng thái mới
                if new_fuel < 0 or new_state.total_cost > self.MAX_TOTAL_COST:
                    continue
                
                # Kiểm tra xem trạng thái đã được thăm chưa
                state_key = new_state.get_state_key()
                if state_key in self.closed_set:
                    continue
                
                # Cập nhật chi phí nếu tìm thấy đường đi tốt hơn
                tentative_g_score = current_state.total_cost + move_cost
                if state_key not in self.g_score or tentative_g_score < self.g_score[state_key]:
                    self.g_score[state_key] = tentative_g_score
                    self.f_score[state_key] = tentative_g_score + self.heuristic(next_pos, goal, new_fuel)
                    self.open_set.put((self.f_score[state_key], new_state))
                    self.visited.append(next_pos)
        
        return []  # No path found
    
    def step(self) -> bool:
        """Execute one step of A* with fuel constraints."""
        if not self.open_set:
            self.current_position = None
            return True
        
        self.steps += 1
        
        # Tìm điểm có f_score nhỏ nhất trong open_set
        current = min(self.open_set, key=lambda x: self.f_score[x])
        
        if current == self.goal:
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(self._reconstruct_path(current), self.MAX_FUEL)
            if is_feasible:
                self.current_path = self._reconstruct_path(current)
                self.path_length = len(self.current_path) - 1
                self.cost = self.g_score[current]
                self.fuel = self.fuel[current]
                self.total_cost = self.total_cost[current]
                self.fuel_cost = self.total_cost[current] - self.toll_cost[current]
                self.current_position = None
                return True
            else:
                self.open_set.remove(current)
                self.closed_set.add(current)
                return False
        
        self.open_set.remove(current)
        self.closed_set.add(current)
        self.visited.append(current)
        self.current_position = current
        
        # Xử lý các điểm lân cận
        for next_pos in self.get_neighbor_states(current):
            if next_pos in self.closed_set:
                continue
            
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            distance_cost, fuel_cost, toll_cost = self.calculate_cost(current, next_pos)
            new_g_score = float(self.g_score[current]) + float(distance_cost) + float(fuel_cost) + float(toll_cost)
            
            # Nếu điểm lân cận chưa trong open_set hoặc có g_score tốt hơn
            if next_pos not in self.open_set or new_g_score < self.g_score.get(next_pos, float('inf')):
                self.parent[next_pos] = current
                self.g_score[next_pos] = new_g_score
                self.f_score[next_pos] = new_g_score + self.heuristic(next_pos, self.goal, self.fuel[current])
                
                # Cập nhật nhiên liệu và chi phí
                self.fuel[next_pos] = float(self.fuel[current]) - float(self.FUEL_PER_MOVE)
                self.total_cost[next_pos] = new_g_score
                self.fuel_cost[next_pos] = self.fuel_cost[current] + fuel_cost
                self.toll_cost[next_pos] = self.toll_cost[current] + toll_cost
                
                # Cập nhật các trạm đã thăm
                self.visited_gas_stations[next_pos] = self.visited_gas_stations[current].copy()
                self.toll_stations_visited[next_pos] = self.toll_stations_visited[current].copy()
                
                # Nếu đến trạm xăng
                if self.grid[next_pos[0], next_pos[1]] == 3:
                    self.fuel[next_pos] = float(self.MAX_FUEL)
                    self.visited_gas_stations[next_pos].add(next_pos)
                
                # Nếu đến trạm thu phí
                if self.grid[next_pos[0], next_pos[1]] == 2:
                    self.toll_stations_visited[next_pos].add(next_pos)
                
                self.open_set.add(next_pos)
        
        return False
    
    def _reconstruct_path(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Tái tạo đường đi từ điểm hiện tại về điểm bắt đầu."""
        path = [current]
        while self.parent[current] is not None:
            current = self.parent[current]
            path.append(current)
        return list(reversed(path)) 