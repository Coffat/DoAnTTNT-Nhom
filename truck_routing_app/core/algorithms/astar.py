"""
A* search algorithm implementation.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from queue import PriorityQueue
from .base_search import BaseSearch, SearchState

class AStar(BaseSearch):
    """A* search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, use_fuel_priority: bool = True):
        """Initialize A* with a grid.
        
        Args:
            grid: Bản đồ
            use_fuel_priority: Có ưu tiên tìm trạm xăng khi nhiên liệu thấp hay không (mặc định: True)
        """
        super().__init__(grid)
        self.open_set = PriorityQueue()
        self.closed_set = set()
        self.g_score = {}  # Chi phí thực tế từ điểm bắt đầu đến điểm hiện tại
        self.f_score = {}  # Ước tính tổng chi phí (g_score + heuristic)
        self.parent = {}
        self.start = None
        self.goal = None
        self.use_fuel_priority = use_fuel_priority
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Tính hàm heuristic (Manhattan distance)."""
        x1, y1 = pos
        x2, y2 = goal
        return float(abs(x1 - x2) + abs(y1 - y2))
    
    def heuristic_with_fuel_priority(self, pos: Tuple[int, int], goal: Tuple[int, int], 
                                   fuel: float) -> float:
        """Tính hàm heuristic với ưu tiên trạm xăng khi nhiên liệu thấp.
        
        Args:
            pos: Vị trí hiện tại
            goal: Vị trí đích
            fuel: Lượng nhiên liệu hiện tại
            
        Returns:
            float: Giá trị heuristic (càng nhỏ càng tốt)
        """
        # Nếu không sử dụng ưu tiên nhiên liệu hoặc nhiên liệu đủ, hướng về đích
        if not self.use_fuel_priority or fuel >= self.LOW_FUEL_THRESHOLD:
            return self.heuristic(pos, goal)
        
        # Tạo trạng thái tạm để tìm trạm xăng gần nhất
        temp_state = SearchState(
            position=pos,
            fuel=fuel,
            total_cost=0,
            path=[],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Tìm trạm xăng gần nhất
        nearest_gas = self.find_nearest_reachable_gas_station(temp_state)
        
        # Nếu không tìm thấy trạm xăng trong tầm với, vẫn hướng về đích
        if nearest_gas is None:
            return self.heuristic(pos, goal)
        
        # Tính chi phí đến trạm xăng
        cost_to_gas = self.heuristic(pos, nearest_gas)
        
        # Tính chi phí từ trạm xăng đến đích
        cost_from_gas_to_goal = self.heuristic(nearest_gas, goal)
        
        # Ưu tiên đường đi qua trạm xăng bằng cách giảm chi phí
        # Nhân hệ số 0.8 để ưu tiên đường đi qua trạm xăng hơn đường đi trực tiếp
        return cost_to_gas + 0.8 * cost_from_gas_to_goal
    
    def calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], fuel: float) -> float:
        """Tính heuristic dựa vào cài đặt"""
        if self.use_fuel_priority:
            return self.heuristic_with_fuel_priority(pos, goal, fuel)
        else:
            return self.heuristic(pos, goal)
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute A* from start to goal."""
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
        initial_h = self.calculate_heuristic(start, goal, self.MAX_FUEL)
        self.f_score[initial_state.get_state_key()] = initial_h
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
                # Kiểm tra vật cản
                if self.grid[next_pos[1], next_pos[0]] == 3:  # Nếu là vật cản (loại 3)
                    continue
                    
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
                if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng
                    new_state.visited_gas_stations.add(next_pos)
                elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí
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
                    h_score = self.calculate_heuristic(next_pos, goal, new_fuel)
                    self.f_score[state_key] = tentative_g_score + h_score
                    self.open_set.put((self.f_score[state_key], new_state))
                    self.visited.append(next_pos)
        
        return []  # No path found
    
    def step(self) -> bool:
        """Execute one step of A*."""
        if self.open_set.empty():
            self.current_position = None
            return True
        
        self.steps += 1
        current_state = self.open_set.get()[1]
        self.current_position = current_state.position
        
        if current_state.position == self.goal:
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(current_state.path, self.MAX_FUEL)
            if is_feasible:
                self.current_path = current_state.path
                self.path_length = len(self.current_path) - 1
                self.cost = current_state.total_cost
                self.current_fuel = current_state.fuel
                self.current_total_cost = current_state.total_cost
                self.current_fuel_cost = current_state.total_cost - self.current_toll_cost
                self.current_position = None
                return True
            else:
                return False
        
        # Thêm trạng thái hiện tại vào tập đóng
        self.closed_set.add(current_state.get_state_key())
        
        for next_pos in self.get_neighbors(current_state.position):
            # Kiểm tra vật cản
            if self.grid[next_pos[1], next_pos[0]] == 3:  # Nếu là vật cản (loại 3)
                continue
            
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
            if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng
                new_state.visited_gas_stations.add(next_pos)
            elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí
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
                h_score = self.calculate_heuristic(next_pos, self.goal, new_fuel)
                self.f_score[state_key] = tentative_g_score + h_score
                self.open_set.put((self.f_score[state_key], new_state))
                self.visited.append(next_pos)
        
        return False
    
    def _reconstruct_path(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Tái tạo đường đi từ điểm hiện tại về điểm bắt đầu."""
        path = [current]
        while current in self.parent:
            current = self.parent[current]
            path.append(current)
        return list(reversed(path)) 