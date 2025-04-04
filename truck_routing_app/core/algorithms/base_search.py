"""
Base search algorithm module.
Provides abstract base classes for implementing various search algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from queue import PriorityQueue
import numpy as np
import math

@dataclass
class SearchState:
    """Trạng thái tìm kiếm bao gồm vị trí, nhiên liệu và chi phí"""
    position: Tuple[int, int]  # (x, y)
    fuel: float               # Lượng nhiên liệu còn lại (L)
    total_cost: float        # Tổng chi phí (đ)
    path: List[Tuple[int, int]]  # Đường đi đã qua
    visited_gas_stations: Set[Tuple[int, int]]  # Các trạm xăng đã thăm
    toll_stations_visited: Set[Tuple[int, int]]  # Các trạm thu phí đã qua
    
    def __lt__(self, other):
        # So sánh để sử dụng trong PriorityQueue
        # Ưu tiên theo tổng chi phí thấp hơn
        return self.total_cost < other.total_cost
    
    def __eq__(self, other):
        # So sánh bằng dựa trên vị trí và lượng nhiên liệu
        if not isinstance(other, SearchState):
            return False
        return (self.position == other.position and 
                abs(self.fuel - other.fuel) < 0.01)  # So sánh float với độ chính xác
    
    def __hash__(self):
        # Hash dựa trên vị trí và lượng nhiên liệu (làm tròn để tránh vấn đề với float)
        return hash((self.position, round(self.fuel, 2)))
    
    def get_state_key(self) -> Tuple[Tuple[int, int], float]:
        """Tạo khóa duy nhất cho trạng thái dựa trên vị trí và nhiên liệu"""
        return (self.position, round(self.fuel, 2))

class BaseSearch(ABC):
    """Abstract base class for search algorithms."""
    
    # Các hằng số
    FUEL_PER_MOVE = 0.4      # Nhiên liệu tiêu thụ mỗi bước (L)
    MAX_FUEL = 3.0           # Dung tích bình xăng tối đa (L)
    GAS_STATION_COST = 30.0  # Chi phí đổ xăng (đ)
    TOLL_COST = 5.0          # Chi phí qua trạm thu phí (đ)
    TOLL_PENALTY = 1000.0    # Phạt cho việc đi qua trạm thu phí
    MAX_TOTAL_COST = 5000.0  # Chi phí tối đa cho phép (đ)
    LOW_FUEL_THRESHOLD = 1.0  # Ngưỡng xăng thấp để ưu tiên tìm trạm xăng (L)
    
    def __init__(self, grid: np.ndarray):
        """Initialize the search algorithm with a grid."""
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.visited = []
        self.current_path = []
        self.current_position = None
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL  # Ban đầu có 3l xăng
        self.current_total_cost = 0  # Tổng chi phí (bao gồm cả trạm thu phí)
        self.current_fuel_cost = 0  # Chi phí xăng
        self.current_toll_cost = 0  # Chi phí trạm thu phí
        self.toll_stations_visited = set()  # Các trạm thu phí đã qua
    
    @abstractmethod
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute search algorithm from start to goal."""
        pass
    
    @abstractmethod
    def step(self) -> bool:
        """Execute one step of the algorithm. Returns True if finished."""
        pass
    
    def get_current_path(self) -> List[Tuple[int, int]]:
        """Get the current path found by the algorithm."""
        return self.current_path
    
    def get_visited(self) -> List[Tuple[int, int]]:
        """Get the list of visited positions."""
        return self.visited
    
    def get_current_position(self) -> Optional[Tuple[int, int]]:
        """Get the current position of the algorithm."""
        return self.current_position
    
    def get_statistics(self) -> Dict:
        """Get statistics about the algorithm's execution."""
        return {
            "steps": self.steps,
            "visited": len(self.visited),
            "path_length": self.path_length,
            "cost": self.cost,
            "fuel": self.current_fuel,
            "total_cost": self.current_total_cost,
            "fuel_cost": self.current_fuel_cost,
            "toll_cost": self.current_toll_cost
        }
    
    def is_finished(self) -> bool:
        """Check if the algorithm has finished."""
        return self.current_position is None
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid on the grid."""
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Tính toán chi phí và nhiên liệu khi di chuyển đến ô tiếp theo
        Returns: (new_fuel, move_cost)"""
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = 0.0
        
        next_cell_type = self.grid[next_pos[1], next_pos[0]]  # Lấy giá trị số nguyên
        
        if next_cell_type == 2:  # Trạm xăng (thay vì 'G')
            # Chỉ tính chi phí đổ xăng nếu nhiên liệu chưa đầy
            if new_fuel < self.MAX_FUEL:
                # Tính chính xác chi phí đổ xăng dựa trên lượng nhiên liệu cần thêm
                fuel_needed = self.MAX_FUEL - new_fuel
                move_cost = self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL)
                new_fuel = self.MAX_FUEL
        elif next_cell_type == 1:  # Trạm thu phí (thay vì 'T')
            # Chỉ tính phí trạm thu phí nếu chưa đi qua trạm này
            if next_pos not in current.toll_stations_visited:
                # Tính chi phí dựa trên mức phí và hệ số phạt
                toll_base_cost = self.TOLL_COST
                
                # Áp dụng phạt với mức giảm dần theo số trạm thu phí đã đi qua
                # (càng nhiều trạm đã qua, phạt càng giảm)
                toll_penalty = self.TOLL_PENALTY * (1.0 - min(0.5, len(current.toll_stations_visited) * 0.1))
                
                move_cost = toll_base_cost + toll_penalty
            
        return new_fuel, move_cost
    
    def is_path_feasible(self, path: List[Tuple[int, int]], start_fuel: float) -> Tuple[bool, str]:
        """Kiểm tra xem đường đi có khả thi không với ràng buộc nhiên liệu và chi phí.
        Returns: (is_feasible, reason)"""
        current_fuel = start_fuel
        total_cost = 0
        visited_tolls = set()
        
        for i in range(len(path) - 1):
            pos1, pos2 = path[i], path[i + 1]
            new_fuel, move_cost = self.calculate_cost(SearchState(
                position=pos1,
                fuel=current_fuel,
                total_cost=total_cost,
                path=path[:i+1],
                visited_gas_stations=set(),
                toll_stations_visited=visited_tolls
            ), pos2)
            
            # Kiểm tra nhiên liệu
            current_fuel = new_fuel
            if current_fuel < 0:
                return False, "Không đủ nhiên liệu để hoàn thành đường đi"
            
            # Cập nhật tổng chi phí
            total_cost += move_cost
            
            # Kiểm tra tổng chi phí
            if total_cost > self.MAX_TOTAL_COST:
                return False, "Tổng chi phí vượt quá giới hạn cho phép"
        
        return True, "Đường đi khả thi"
    
    def update_fuel_and_cost(self, pos: Tuple[int, int]):
        """Cập nhật nhiên liệu và chi phí khi di chuyển đến một vị trí."""
        # Giảm xăng khi di chuyển
        self.current_fuel -= self.FUEL_PER_MOVE
        
        # Nếu đến trạm xăng, đổ đầy xăng
        if self.grid[pos[1], pos[0]] == 2:  # Trạm xăng
            self.current_fuel = self.MAX_FUEL
            self.current_fuel_cost += self.GAS_STATION_COST
        
        # Nếu đến trạm thu phí, tăng chi phí
        if self.grid[pos[1], pos[0]] == 1:  # Trạm thu phí
            if pos not in self.toll_stations_visited:
                self.current_toll_cost += self.TOLL_COST + self.TOLL_PENALTY
                self.toll_stations_visited.add(pos)
        
        # Cập nhật tổng chi phí
        self.current_total_cost = self.current_fuel_cost + self.current_toll_cost
    
    def has_enough_fuel(self, pos: Tuple[int, int]) -> bool:
        """Kiểm tra xem có đủ xăng để di chuyển đến vị trí không."""
        return self.current_fuel >= self.FUEL_PER_MOVE
    
    def find_nearest_reachable_gas_station(self, current_state: SearchState) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất có thể đến được từ trạng thái hiện tại
        
        Args:
            current_state: Trạng thái hiện tại
            
        Returns:
            Optional[Tuple[int, int]]: Tọa độ trạm xăng gần nhất hoặc None nếu không tìm thấy
        """
        from collections import deque
        
        # Khởi tạo hàng đợi BFS và tập đã thăm
        queue = deque([current_state.position])
        visited = {current_state.position}
        parent = {current_state.position: None}
        distance = {current_state.position: 0}
        max_distance = current_state.fuel / self.FUEL_PER_MOVE  # Số bước tối đa có thể đi
        
        # BFS để tìm trạm xăng gần nhất
        while queue:
            current_pos = queue.popleft()
            
            # Nếu đã đi quá xa so với lượng xăng hiện tại, bỏ qua
            if distance[current_pos] > max_distance:
                continue
            
            # Nếu vị trí hiện tại là trạm xăng, trả về vị trí này
            if self.grid[current_pos[1], current_pos[0]] == 2:  # Trạm xăng (loại 2)
                return current_pos
            
            # Kiểm tra các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Bỏ qua ô vật cản
                if self.grid[next_pos[1], next_pos[0]] == 3:  # Vật cản (loại 3)
                    continue
                
                # Nếu ô chưa được thăm, thêm vào hàng đợi
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    parent[next_pos] = current_pos
                    distance[next_pos] = distance[current_pos] + 1
        
        # Không tìm thấy trạm xăng trong phạm vi có thể đi được
        return None

class BasePathFinder:
    """Lớp cơ sở cho thuật toán tìm đường với ràng buộc nhiên liệu"""
    
    # Các hằng số
    FUEL_PER_MOVE = 0.5      # Nhiên liệu tiêu thụ mỗi bước (L)
    MAX_FUEL = 3.0           # Dung tích bình xăng tối đa (L)
    GAS_STATION_COST = 30.0  # Chi phí đổ xăng (đ)
    TOLL_COST = 5.0          # Chi phí qua trạm thu phí (đ)
    TOLL_PENALTY = 1000.0    # Phạt cho việc đi qua trạm thu phí
    
    def __init__(self, grid: List[List[str]]):
        """
        Khởi tạo thuật toán với bản đồ
        grid: List[List[str]] - Bản đồ với các ký tự:
            '.' - Ô trống
            'G' - Trạm xăng
            'T' - Trạm thu phí
            'S' - Điểm bắt đầu
            'E' - Điểm kết thúc
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        
        # Tìm tất cả các trạm xăng trên bản đồ
        self.gas_stations = set()
        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] == 'G':
                    self.gas_stations.add((x, y))
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các ô lân cận có thể di chuyển tới"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4 hướng
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                neighbors.append((new_x, new_y))
        return neighbors
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Tính toán chi phí và nhiên liệu khi di chuyển đến ô tiếp theo
        Returns: (new_fuel, move_cost)"""
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = 0.0
        
        next_cell_type = self.grid[next_pos[1], next_pos[0]]  # Lấy giá trị số nguyên
        
        if next_cell_type == 2:  # Trạm xăng (thay vì 'G')
            # Chỉ đổ xăng nếu chưa đổ tại trạm này và nhiên liệu chưa đầy
            if next_pos not in current.visited_gas_stations and new_fuel < self.MAX_FUEL:
                new_fuel = self.MAX_FUEL
                move_cost = self.GAS_STATION_COST
        elif next_cell_type == 1:  # Trạm thu phí (thay vì 'T')
            # Chỉ tính phí trạm thu phí nếu chưa đi qua trạm này
            if next_pos not in current.toll_stations_visited:
                move_cost = self.TOLL_COST + self.TOLL_PENALTY
            
        return new_fuel, move_cost
    
    def is_valid_state(self, state: SearchState) -> bool:
        """Kiểm tra trạng thái có hợp lệ không (đủ nhiên liệu)"""
        return state.fuel >= self.FUEL_PER_MOVE
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Tìm đường đi từ điểm bắt đầu đến điểm kết thúc
        Phương thức này sẽ được triển khai bởi các lớp con cụ thể
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")