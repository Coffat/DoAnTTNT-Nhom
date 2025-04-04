"""
Greedy Search algorithm implementation.
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import deque
import math
import numpy as np
from .base_search import BaseSearch, SearchState
from queue import PriorityQueue

class GreedySearch(BaseSearch):
    """Greedy Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray):
        """Initialize Greedy Search with a grid."""
        super().__init__(grid)
        self.open_set = PriorityQueue()
        self.closed_set = set()
        self.parent = {}   # Lưu trữ đường đi
        self.fuel = {}     # Lưu trữ nhiên liệu còn lại tại mỗi điểm
        self.total_cost = {}  # Lưu trữ tổng chi phí tại mỗi điểm
        self.fuel_cost = {}   # Lưu trữ chi phí nhiên liệu tại mỗi điểm
        self.toll_cost = {}   # Lưu trữ chi phí thu phí tại mỗi điểm
        self.visited_gas_stations = set()
        self.toll_stations_visited = set()
        self.start = None
        self.goal = None
        self.current_position = None
        self.current_fuel = 0.0  # Thêm biến để lưu nhiên liệu hiện tại
        self.current_total_cost = 0.0  # Thêm biến để lưu tổng chi phí hiện tại
        self.current_fuel_cost = 0.0  # Thêm biến để lưu chi phí nhiên liệu hiện tại
        self.current_toll_cost = 0.0  # Thêm biến để lưu chi phí thu phí hiện tại
        self.distance_from_gas_station = {}  # Lưu khoảng cách từ mỗi ô đến trạm xăng gần nhất
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Tính hàm heuristic (Manhattan distance)."""
        x1, y1 = pos
        x2, y2 = goal
        return float(abs(x1 - x2) + abs(y1 - y2))
    
    def is_gas_station_neighbor(self, pos: Tuple[int, int]) -> bool:
        """Kiểm tra xem xung quanh có trạm xăng không."""
        for neighbor in self.get_neighbors(pos):
            if self.grid[neighbor[1], neighbor[0]] == 2:  # Trạm xăng
                return True
        return False
    
    def heuristic_with_fuel(self, state: SearchState, goal: Tuple[int, int]) -> float:
        """Tính hàm heuristic có xét đến nhiên liệu (đã sửa đổi).
        
        Args:
            state: Trạng thái hiện tại
            goal: Vị trí đích
            
        Returns:
            float: Giá trị heuristic (càng nhỏ càng tốt)
        """
        pos = state.position
        fuel = state.fuel
        
        # Kiểm tra vật cản ngay đầu tiên
        cell_type = self.grid[pos[1], pos[0]]
        if cell_type == 3:  # Vật cản
            return float('inf')  # Không bao giờ đi qua vật cản
        
        # Kiểm tra nếu đang ở cạnh trạm xăng và nhiên liệu không đầy
        if cell_type != 2 and self.is_gas_station_neighbor(pos) and fuel < self.MAX_FUEL:
            # Tìm trạm xăng lân cận
            for neighbor in self.get_neighbors(pos):
                if self.grid[neighbor[1], neighbor[0]] == 2:  # Trạm xăng
                    # Ưu tiên cao nhất để đi vào trạm xăng lân cận
                    return -10000.0
            
        # Trạm xăng - Ưu tiên khi nhiên liệu thấp
        if cell_type == 2:  # Nếu đang ở trạm xăng
            if fuel < self.MAX_FUEL:  # Nếu nhiên liệu chưa đầy
                return -10000.0  # Ưu tiên tuyệt đối để đổ xăng
            else:
                # Tính khoảng cách đến đích
                return self.heuristic(pos, goal)  # Nhiên liệu đã đầy, tiếp tục đi
        
        # Tính khoảng cách đến đích
        h_goal = self.heuristic(pos, goal)
            
        # Xét nhiên liệu
        if fuel < self.LOW_FUEL_THRESHOLD:
            # Khi nhiên liệu thấp
            nearest_gas = self.find_nearest_reachable_gas_station(state)
            if nearest_gas is not None:
                # Tính khoảng cách đến trạm xăng
                cost_to_gas = self.heuristic(pos, nearest_gas)
                
                # Mức độ ưu tiên dựa trên lượng nhiên liệu
                # Càng ít nhiên liệu, ưu tiên càng cao
                fuel_factor = 1.0 - (fuel / self.LOW_FUEL_THRESHOLD)
                
                # Nếu nhiên liệu rất thấp (dưới 30% ngưỡng xăng thấp), ưu tiên tuyệt đối cho trạm xăng
                if fuel < 0.3 * self.LOW_FUEL_THRESHOLD:
                    return cost_to_gas - 10000.0  # Ưu tiên tuyệt đối cho trạm xăng
                else:
                    # Ưu tiên cao cho việc tìm trạm xăng
                    return cost_to_gas - 5000.0 * fuel_factor
            else:
                # Không tìm thấy trạm xăng nào trong tầm với
                # Trả về giá trị lớn để giảm ưu tiên cho trạng thái này
                return h_goal + 1000.0
        
        # Nhiên liệu đủ, hướng về đích
        # Thêm hình phạt nhẹ nếu ô hiện tại là trạm thu phí chưa đi qua
        if cell_type == 1 and pos not in state.toll_stations_visited:
            h_goal += self.TOLL_PENALTY * 0.1  # Hình phạt nhỏ
                
        return h_goal
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Tính toán chi phí và nhiên liệu khi di chuyển đến ô tiếp theo.
        
        Args:
            current: Trạng thái hiện tại
            next_pos: Vị trí tiếp theo
            
        Returns: 
            Tuple[float, float]: (new_fuel, move_cost) - nhiên liệu mới và chi phí di chuyển
        """
        # Kiểm tra chướng ngại vật
        if self.grid[next_pos[1], next_pos[0]] == 3:
            return -1.0, float('inf')  # Không thể đi qua chướng ngại vật
            
        # Nhiên liệu giảm khi di chuyển
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = 0.0
        
        # Xét loại ô tiếp theo
        cell_type = self.grid[next_pos[1], next_pos[0]]
        
        # Lưu ý: new_fuel tại đây là nhiên liệu *sau khi* tiêu thụ, *trước khi* đổ
        fuel_after_consumption = new_fuel
        
        if cell_type == 2:  # Trạm xăng
            # Luôn nạp đầy nhiên liệu khi đến trạm xăng
            fuel_needed = self.MAX_FUEL - fuel_after_consumption
            if fuel_needed > 0:
                # Tính chi phí đổ xăng dựa trên lượng cần đổ
                move_cost = self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL)
                new_fuel = self.MAX_FUEL  # Cập nhật lại new_fuel sau khi đổ
            else:
                new_fuel = self.MAX_FUEL  # Vẫn đảm bảo là MAX_FUEL dù không cần đổ
                
        elif cell_type == 1:  # Trạm thu phí
            # Chỉ tính phí nếu chưa đi qua trạm này
            if next_pos not in current.toll_stations_visited:
                move_cost = self.TOLL_COST + self.TOLL_PENALTY
        
        return new_fuel, move_cost
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Greedy Search from start to goal."""
        self.start = start
        self.goal = goal
        self.open_set = PriorityQueue()
        self.closed_set.clear()
        self.visited.clear()
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Kiểm tra vị trí bắt đầu và đích
        if self.grid[start[1], start[0]] == 3 or self.grid[goal[1], goal[0]] == 3:
            return []  # Không thể bắt đầu từ vật cản hoặc đi đến vật cản
        
        # Tạo trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Thêm trạng thái ban đầu vào hàng đợi ưu tiên
        initial_priority = self.heuristic_with_fuel(initial_state, goal)
        self.open_set.put((initial_priority, 0, initial_state))
        self.visited.append(start)
        self.current_position = start
        self.current_path = [start]
        
        # Biến đếm để đảm bảo thứ tự ổn định trong priority queue
        counter = 1
        
        while not self.open_set.empty():
            self.steps += 1
            
            # Lấy trạng thái có ưu tiên cao nhất
            _, _, current_state = self.open_set.get()
            
            # Hiển thị vị trí hiện tại cho quá trình mô phỏng
            self.current_position = current_state.position
            
            # Kiểm tra xem đã thăm trạng thái này chưa
            current_key = current_state.get_state_key()
            if current_key in self.closed_set:
                continue
                
            # Thêm vào tập đóng để không xét lại
            self.closed_set.add(current_key)
            
            # Đạt đến đích
            if current_state.position == goal:
                # Kiểm tra tính khả thi của đường đi
                is_feasible, reason = self.is_path_feasible(current_state.path, self.MAX_FUEL)
                if is_feasible:
                    self.current_path = current_state.path
                    self.path_length = len(self.current_path) - 1
                    self.cost = current_state.total_cost
                    self.current_fuel = current_state.fuel
                    self.current_total_cost = current_state.total_cost
                    # Tính toán chi phí nhiên liệu và chi phí thu phí
                    total_toll_cost = 0
                    for pos in current_state.toll_stations_visited:
                        total_toll_cost += self.TOLL_COST
                    self.current_toll_cost = total_toll_cost
                    self.current_fuel_cost = current_state.total_cost - self.current_toll_cost
                    return self.current_path
                else:
                    continue
            
            # Nếu nhiên liệu thấp và đang ở cạnh trạm xăng, ưu tiên đi vào trạm xăng trước
            if current_state.fuel < self.LOW_FUEL_THRESHOLD:
                gas_station_found = False
                for next_pos in self.get_neighbors(current_state.position):
                    if self.grid[next_pos[1], next_pos[0]] == 2:  # Nếu là trạm xăng
                        # Tính toán chi phí và nhiên liệu khi đi vào trạm xăng
                        new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                        
                        # Nếu đủ nhiên liệu để đi đến trạm xăng
                        if new_fuel >= 0:
                            # Tạo trạng thái mới
                            new_total_cost = current_state.total_cost + move_cost
                            new_visited_gas = current_state.visited_gas_stations.copy()
                            new_visited_gas.add(next_pos)
                            
                            new_state = SearchState(
                                position=next_pos,
                                fuel=self.MAX_FUEL,  # Đổ đầy nhiên liệu
                                total_cost=new_total_cost,
                                path=current_state.path + [next_pos],
                                visited_gas_stations=new_visited_gas,
                                toll_stations_visited=current_state.toll_stations_visited.copy()
                            )
                            
                            # Kiểm tra closed set cho trạng thái mới
                            new_key = new_state.get_state_key()
                            if new_key not in self.closed_set:
                                # Thêm vào hàng đợi với ưu tiên rất cao
                                counter += 1
                                self.open_set.put((-10000.0, counter, new_state))
                                if next_pos not in self.visited:
                                    self.visited.append(next_pos)
                                gas_station_found = True
                
                # Nếu đã tìm thấy và ưu tiên trạm xăng, bỏ qua các lân cận khác
                if gas_station_found:
                    continue
            
            # Xử lý các trạng thái lân cận
            for next_pos in self.get_neighbors(current_state.position):
                # Kiểm tra vật cản TRƯỚC TIÊN
                if self.grid[next_pos[1], next_pos[0]] == 3:  # Nếu là vật cản (loại 3)
                    continue
                
                # Tính toán chi phí và nhiên liệu cho bước tiếp theo
                new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                
                # Kiểm tra chi phí và nhiên liệu
                if new_fuel < 0 or move_cost == float('inf'):
                    continue
                
                # Tạo trạng thái mới
                new_total_cost = current_state.total_cost + move_cost
                
                # Kiểm tra chi phí tối đa
                if new_total_cost > self.MAX_TOTAL_COST:
                    continue
                
                # Tạo bản sao các tập hợp để tránh thay đổi trạng thái cha
                new_visited_gas = current_state.visited_gas_stations.copy()
                new_visited_toll = current_state.toll_stations_visited.copy()
                
                # Cập nhật tập đã thăm dựa trên ô *next_pos*
                next_cell_type = self.grid[next_pos[1], next_pos[0]]
                if next_cell_type == 2:  # Trạm xăng
                    new_visited_gas.add(next_pos)
                elif next_cell_type == 1:  # Trạm thu phí
                    new_visited_toll.add(next_pos)
                
                # Tạo trạng thái mới
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=new_total_cost,
                    path=current_state.path + [next_pos],
                    visited_gas_stations=new_visited_gas,
                    toll_stations_visited=new_visited_toll
                )
                
                # Kiểm tra closed set cho trạng thái mới
                new_key = new_state.get_state_key()
                if new_key in self.closed_set:
                    continue
                
                # Tính toán ưu tiên cho trạng thái mới
                priority = self.heuristic_with_fuel(new_state, goal)
                
                # Chỉ thêm vào hàng đợi nếu heuristic hợp lệ
                if priority != float('inf'):
                    counter += 1
                    self.open_set.put((priority, counter, new_state))
                    # Thêm vào danh sách đã thăm nếu chưa có
                    if next_pos not in self.visited:
                        self.visited.append(next_pos)
        
        return []  # No path found
        
    def can_reach_gas_station(self, pos: Tuple[int, int], fuel: float) -> bool:
        """Kiểm tra từ vị trí hiện tại với lượng nhiên liệu hiện có có thể đến được trạm xăng không.
        
        Args:
            pos: Vị trí hiện tại
            fuel: Lượng nhiên liệu hiện có
            
        Returns:
            bool: True nếu có thể đến được trạm xăng, False nếu không
        """
        # Tạo trạng thái tạm thời để kiểm tra
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
        
        # Có thể đến được trạm xăng nếu tìm thấy một trạm
        return nearest_gas is not None
    
    def step(self) -> bool:
        """Execute one step of Greedy Search."""
        if self.open_set.empty():
            self.current_position = None
            return True
        
        self.steps += 1
        _, _, current_state = self.open_set.get()
        self.current_position = current_state.position
        
        # Kiểm tra xem đã thăm trạng thái này chưa
        current_key = current_state.get_state_key()
        if current_key in self.closed_set:
            return False
            
        # Thêm vào tập đóng để không xét lại
        self.closed_set.add(current_key)
        
        if current_state.position == self.goal:
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(current_state.path, self.MAX_FUEL)
            if is_feasible:
                self.current_path = current_state.path
                self.path_length = len(self.current_path) - 1
                self.cost = current_state.total_cost
                self.current_fuel = current_state.fuel
                self.current_total_cost = current_state.total_cost
                # Tính toán chi phí nhiên liệu và chi phí thu phí
                total_toll_cost = 0
                for pos in current_state.toll_stations_visited:
                    total_toll_cost += self.TOLL_COST
                self.current_toll_cost = total_toll_cost
                self.current_fuel_cost = current_state.total_cost - self.current_toll_cost
                self.current_position = None
                return True
            else:
                return False
        
        # Nếu nhiên liệu thấp và đang ở cạnh trạm xăng, ưu tiên đi vào trạm xăng trước
        if current_state.fuel < self.LOW_FUEL_THRESHOLD:
            gas_station_found = False
            for next_pos in self.get_neighbors(current_state.position):
                if self.grid[next_pos[1], next_pos[0]] == 2:  # Nếu là trạm xăng
                    # Tính toán chi phí và nhiên liệu khi đi vào trạm xăng
                    new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                    
                    # Nếu đủ nhiên liệu để đi đến trạm xăng
                    if new_fuel >= 0:
                        # Tạo trạng thái mới
                        new_total_cost = current_state.total_cost + move_cost
                        new_visited_gas = current_state.visited_gas_stations.copy()
                        new_visited_gas.add(next_pos)
                        
                        new_state = SearchState(
                            position=next_pos,
                            fuel=self.MAX_FUEL,  # Đổ đầy nhiên liệu
                            total_cost=new_total_cost,
                            path=current_state.path + [next_pos],
                            visited_gas_stations=new_visited_gas,
                            toll_stations_visited=current_state.toll_stations_visited.copy()
                        )
                        
                        # Kiểm tra closed set cho trạng thái mới
                        new_key = new_state.get_state_key()
                        if new_key not in self.closed_set:
                            # Thêm vào hàng đợi với ưu tiên rất cao
                            self.open_set.put((-10000.0, self.steps, new_state))
                            if next_pos not in self.visited:
                                self.visited.append(next_pos)
                            gas_station_found = True
            
            # Nếu đã tìm thấy và ưu tiên trạm xăng, bỏ qua các lân cận khác
            if gas_station_found:
                return False
        
        # Xử lý các trạng thái lân cận
        for next_pos in self.get_neighbors(current_state.position):
            # Kiểm tra vật cản TRƯỚC TIÊN
            if self.grid[next_pos[1], next_pos[0]] == 3:  # Nếu là vật cản (loại 3)
                continue
                
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
            
            # Kiểm tra chi phí và nhiên liệu
            if new_fuel < 0 or move_cost == float('inf'):
                continue
            
            # Tạo trạng thái mới
            new_total_cost = current_state.total_cost + move_cost
            
            # Kiểm tra chi phí tối đa
            if new_total_cost > self.MAX_TOTAL_COST:
                continue
                
            # Tạo bản sao các tập hợp để tránh thay đổi trạng thái cha
            new_visited_gas = current_state.visited_gas_stations.copy()
            new_visited_toll = current_state.toll_stations_visited.copy()
            
            # Cập nhật tập đã thăm dựa trên ô *next_pos*
            next_cell_type = self.grid[next_pos[1], next_pos[0]]
            if next_cell_type == 2:  # Trạm xăng
                new_visited_gas.add(next_pos)
            elif next_cell_type == 1:  # Trạm thu phí
                new_visited_toll.add(next_pos)
            
            # Tạo trạng thái mới
            new_state = SearchState(
                position=next_pos,
                fuel=new_fuel,
                total_cost=new_total_cost,
                path=current_state.path + [next_pos],
                visited_gas_stations=new_visited_gas,
                toll_stations_visited=new_visited_toll
            )
            
            # Kiểm tra closed set cho trạng thái mới
            new_key = new_state.get_state_key()
            if new_key in self.closed_set:
                continue
            
            # Tính toán ưu tiên cho trạng thái mới
            priority = self.heuristic_with_fuel(new_state, self.goal)
            
            # Chỉ thêm vào hàng đợi nếu heuristic hợp lệ
            if priority != float('inf'):
                self.open_set.put((priority, self.steps, new_state))
                # Thêm vào danh sách đã thăm nếu chưa có
                if next_pos not in self.visited:
                    self.visited.append(next_pos)
        
        return False 