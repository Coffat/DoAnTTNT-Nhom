"""
Module for routing visualization and algorithm comparison.
This module provides the UI for visualizing different routing algorithms.
"""

import streamlit as st
import numpy as np
import time
from typing import List, Tuple, Dict
from core.algorithms.base_search import BaseSearch
from core.algorithms.bfs import BFS
from core.algorithms.dfs import DFS
from core.algorithms.astar import AStar
from core.algorithms.greedy import GreedySearch
from core.algorithms.hill_climbing import HillClimbing
from core.algorithms.local_beam import LocalBeamSearch
from ui.map_display import draw_map, draw_route

def draw_visualization_step(map_data, visited, current_pos, path=None, current_neighbors=None):
    """Vẽ một bước của thuật toán với các hiệu ứng trực quan."""
    try:
        # Vẽ bản đồ với các hiệu ứng trực quan
        draw_map(
            map_data=map_data,
            visited=visited,
            current_neighbors=current_neighbors,
            current_pos=current_pos,
            path=path
        )
    except Exception as e:
        st.error(f"Lỗi khi vẽ bước trực quan: {str(e)}")

def run_algorithm(algorithm_name: str, map_data: np.ndarray, start: Tuple[int, int], 
                 goal: Tuple[int, int]) -> Dict:
    """Chạy một thuật toán và trả về kết quả."""
    # Lấy grid từ map_data
    grid = map_data.grid if hasattr(map_data, 'grid') else map_data
    
    # Khởi tạo thuật toán
    if algorithm_name == "BFS":
        algorithm = BFS(grid)
    elif algorithm_name == "DFS":
        algorithm = DFS(grid)
    elif algorithm_name == "A*":
        algorithm = AStar(grid)
    elif algorithm_name == "Greedy":
        algorithm = GreedySearch(grid)
    elif algorithm_name == "Hill Climbing":
        algorithm = HillClimbing(grid)
    elif algorithm_name == "Local Beam Search":
        algorithm = LocalBeamSearch(grid)
    else:
        st.error(f"Thuật toán {algorithm_name} không được hỗ trợ!")
        return None
    
    # Chạy thuật toán
    path = algorithm.search(start, goal)
    
    # Lấy thống kê
    stats = algorithm.get_statistics()
    
    # Xử lý thêm cho BFS và DFS để hiển thị thông tin về tính khả thi
    if algorithm_name in ["BFS", "DFS"] and path:
        # Kiểm tra nếu nhiên liệu về 0, có thể đường đi không khả thi
        if stats["fuel"] <= 0:
            stats["is_feasible"] = False
            stats["reason"] = "Hết nhiên liệu trên đường đi"
        else:
            stats["is_feasible"] = True
            stats["reason"] = "Đường đi khả thi"
    else:
        # Các thuật toán khác đã xét ràng buộc trong quá trình tìm kiếm
        stats["is_feasible"] = True if path else False
        stats["reason"] = "Đường đi khả thi" if path else "Không tìm thấy đường đi khả thi"
    
    return {
        "path": path,
        "visited": algorithm.get_visited(),
        "stats": stats
    }

def render_routing_visualization():
    """Render tab định tuyến và tối ưu hệ thống."""
    st.markdown("## 🗺️ Định Tuyến & Tối Ưu Hệ Thống")
    
    # Kiểm tra xem đã có bản đồ chưa
    if "map" not in st.session_state:
        st.warning("⚠️ Vui lòng tạo bản đồ trước khi sử dụng tính năng này!")
        return
    
    # Kiểm tra vị trí bắt đầu và điểm đích
    if "start_pos" not in st.session_state or st.session_state.start_pos is None:
        st.warning("⚠️ Vui lòng thiết lập vị trí bắt đầu của xe!")
        return
    
    if "end_pos" not in st.session_state or st.session_state.end_pos is None:
        st.warning("⚠️ Vui lòng thiết lập điểm đích!")
        return
    
    # Tạo layout chính
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Container cho bản đồ và trực quan
        map_container = st.empty()
        
        # Hiển thị bản đồ ban đầu
        with map_container:
            draw_map(st.session_state.map)
    
    with col2:
        # Chọn thuật toán
        st.markdown("### 🔍 Chọn thuật toán")
        algorithm = st.selectbox(
            "Thuật toán tìm đường:",
            ["BFS", "DFS", "A*", "Greedy", "Hill Climbing", "Local Beam Search"],
            help="Chọn thuật toán để tìm đường đi tối ưu"
        )
        
        # Lưu thuật toán đã chọn vào session state
        st.session_state.algorithm = algorithm
        
        # Hiển thị mô tả thuật toán
        algorithm_descriptions = {
            "BFS": "Tìm kiếm theo chiều rộng, đảm bảo tìm được đường đi ngắn nhất.",
            "DFS": "Tìm kiếm theo chiều sâu, thích hợp cho không gian tìm kiếm lớn.",
            "A*": "Kết hợp tìm kiếm tốt nhất và heuristic, tối ưu và hiệu quả. Tự động ưu tiên tìm trạm xăng khi sắp hết nhiên liệu.",
            "Greedy": "Luôn chọn bước đi có vẻ tốt nhất tại thời điểm hiện tại.",
            "Hill Climbing": "Tìm kiếm cục bộ, luôn di chuyển theo hướng tốt hơn.",
            "Local Beam Search": "Duy trì nhiều trạng thái cùng lúc, tăng khả năng tìm kiếm."
        }
        st.info(algorithm_descriptions[algorithm])
        
        # Nút tìm đường
        if st.button("🔍 Tìm đường", use_container_width=True):
            with st.spinner("🔄 Đang tìm đường..."):
                try:
                    result = run_algorithm(
                        algorithm,
                        st.session_state.map,
                        st.session_state.start_pos,
                        st.session_state.end_pos
                    )
                    if result:
                        st.session_state.current_result = result
                        st.session_state.visualization_step = 0
                        st.session_state.is_playing = False
                        st.success("✅ Đã tìm thấy đường đi!")
                    else:
                        st.error("❌ Không thể tìm được đường đi!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi thực thi thuật toán: {str(e)}")
                    return
    
    # Hiển thị kết quả nếu có
    if "current_result" in st.session_state:
        st.markdown("### 📊 Kết quả tìm đường")
        
        # Hiển thị thống kê
        stats = st.session_state.current_result["stats"]
        algorithm = st.session_state.algorithm if "algorithm" in st.session_state else ""
        
        # Hiển thị tính khả thi đối với BFS và DFS
        if algorithm in ["BFS", "DFS"]:
            is_feasible = stats.get("is_feasible", False)
            reason = stats.get("reason", "")
            
            if is_feasible:
                st.success(f"✅ Đường đi khả thi")
            else:
                st.error(f"❌ Đường đi không khả thi: {reason}")
        
        # Tạo layout cho thống kê
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số bước thực hiện", stats["steps"])
            st.metric("Số ô đã thăm", stats["visited"])
        with col2:
            st.metric("Độ dài đường đi", stats["path_length"])
            st.metric("Xăng còn lại", f"{stats['fuel']:.1f}l")
        with col3:
            st.metric("Chi phí trạm thu phí", f"{stats['toll_cost']}đ")
            st.metric("Tổng chi phí", f"{stats['total_cost']}đ")
        
        # Điều khiển trực quan
        st.markdown("### 🎬 Trực quan hóa thuật toán")
        
        # Điều khiển tốc độ
        speed = st.slider(
            "Tốc độ hiển thị:",
            min_value=1,
            max_value=10,
            value=5,
            help="Điều chỉnh tốc độ hiển thị (1: chậm nhất, 10: nhanh nhất)"
        )
        
        # Nút điều khiển
        control_cols = st.columns(4)
        with control_cols[0]:
            if st.button("⏮️ Về đầu", use_container_width=True):
                st.session_state.visualization_step = 0
                st.session_state.is_playing = False
        with control_cols[1]:
            if st.button("▶️ Chạy/Tạm dừng", use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
        with control_cols[2]:
            if st.button("⏭️ Kết thúc", use_container_width=True):
                st.session_state.visualization_step = len(st.session_state.current_result["visited"])
                st.session_state.is_playing = False
        with control_cols[3]:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.session_state.visualization_step = 0
                st.session_state.is_playing = False
        
        # Thanh tiến trình
        if "visualization_step" not in st.session_state:
            st.session_state.visualization_step = 0
        
        total_steps = len(st.session_state.current_result["visited"])
        current_step = st.session_state.visualization_step
        progress = float(current_step) / total_steps if total_steps > 0 else 0
        st.progress(progress, text=f"Bước {current_step}/{total_steps}")
        
        # Xử lý animation
        visited = st.session_state.current_result["visited"]
        path = st.session_state.current_result["path"]
        
        if st.session_state.is_playing and current_step < total_steps:
            # Hiển thị bản đồ với các ô đã thăm
            current_visited = visited[:current_step + 1]
            current_pos = visited[current_step]
            
            # Lấy các ô hàng xóm của vị trí hiện tại
            current_neighbors = []
            if hasattr(st.session_state.map, 'get_neighbors'):
                current_neighbors = st.session_state.map.get_neighbors(current_pos)
            
            # Vẽ bước hiện tại
            with map_container:
                draw_visualization_step(
                    st.session_state.map,
                    current_visited,
                    current_pos,
                    path if current_step == total_steps - 1 else None,
                    current_neighbors
                )
            
            # Tăng bước và đợi
            time.sleep(1.0 / speed)
            st.session_state.visualization_step += 1
            st.rerun()
        else:
            # Hiển thị trạng thái hiện tại
            if current_step < total_steps:
                current_visited = visited[:current_step + 1]
                current_pos = visited[current_step]
                current_neighbors = []
                if hasattr(st.session_state.map, 'get_neighbors'):
                    current_neighbors = st.session_state.map.get_neighbors(current_pos)
                
                with map_container:
                    draw_visualization_step(
                        st.session_state.map,
                        current_visited,
                        current_pos,
                        path if current_step == total_steps - 1 else None,
                        current_neighbors
                    )
            else:
                # Hiển thị kết quả cuối cùng với đường đi
                with map_container:
                    draw_visualization_step(
                        st.session_state.map,
                        visited,
                        None,
                        path
                    ) 