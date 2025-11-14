################################################################################
# ------------------------------------------------------------------------------
# PHẦN 1: IMPORT THƯ VIỆN
# (SỬA: Dùng Folium, xóa PyDeck và Leaflet)
# ------------------------------------------------------------------------------
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
import time
import itertools
from tqdm.notebook import tqdm # Tqdm sẽ không dùng, nhưng giữ lại cho logic
import warnings

# === SỬA: DÙNG FOLIUM ===
import folium
from streamlit_folium import st_folium
# === HẾT SỬA ===

# Thư viện gốc của bạn
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained

# Tắt các cảnh báo
warnings.filterwarnings('ignore', category=UserWarning)

# Cấu hình layout cho Streamlit
st.set_page_config(layout="wide", page_title="RTM Visit Planner")


################################################################################
# ------------------------------------------------------------------------------
# PHẦN 2: "BỘ NÃO" LOGIC CỐT LÕI (GIỮ NGUYÊN 100%)
# ------------------------------------------------------------------------------
################################################################################

# --- 2a. Các Tham số Business (Đọc từ đây) ---
BUSINESS_PARAMS = {
    'VISIT_TIME_MAP': {
        'MT': 19.5, 'Cooler': 18.0, 'Gold': 9.0,
        'Silver': 7.8, 'Bronze': 6.8, 'default': 10.0
    },
    'WEEKDAY_CAPACITY': 480,
    'SATURDAY_CAPACITY': 240,
    'DAY_CAPACITY_MAP': {
        'Mon': 480, 'Tue': 480, 'Wed': 480,
        'Thu': 480, 'Fri': 480, 'Sat': 240
    },
    'DAYS_OF_WEEK': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
    'N_CLUSTERS_PER_WEEK': 6,
    'AVG_SPEED_KMH': 40.0,
    'CIRCUTIY_FACTOR': 1.4,
    'WEEKS': ['W1', 'W2', 'W3', 'W4'],
    'FREQ_TO_WEEKLY_VISITS': {
        16: 4, 12: 3, 8: 2, 4: 1, 2: 0, 1: 0
    },
    'CONSECUTIVE_PAIRS': {
        ('Mon', 'Tue'), ('Tue', 'Wed'), ('Wed', 'Thu'),
        ('Thu', 'Fri'), ('Fri', 'Sat')
    },
    'BALANCE_ITERATION_LIMIT': 60,
    'BALANCE_TOLERANCE_MIN': 30,
    'COUNT_TOLERANCE_PERCENT': 0.20
}

# --- 2b. Helper Functions (Step 3 - Giữ nguyên logic) ---

def calculate_travel_time(coords_1, coords_2, params):
    """Tính thời gian di chuyển (sử dụng params)."""
    if pd.isna(coords_1[0]) or pd.isna(coords_1[1]) or \
       pd.isna(coords_2[0]) or pd.isna(coords_2[1]):
        return 0
    straight_dist_km = geodesic(coords_1, coords_2).kilometers
    estimated_road_dist_km = straight_dist_km * params['CIRCUTIY_FACTOR']
    travel_time_min = (estimated_road_dist_km / params['AVG_SPEED_KMH']) * 60
    return travel_time_min

def get_optimal_route_workload(customer_data_list, depot_coords, params):
    """Tính workload tối ưu (sử dụng OR-Tools)."""
    total_visit_time = sum(cust['Visit Time (min)'] for cust in customer_data_list)
    if not customer_data_list: return 0, 0
    if len(customer_data_list) == 1:
        cust_coords = customer_data_list[0]['coords']
        time_to_cust = calculate_travel_time(depot_coords, cust_coords, params)
        time_to_depot = calculate_travel_time(cust_coords, depot_coords, params)
        total_travel = time_to_cust + time_to_depot
        return total_visit_time + total_travel, total_travel
    
    locations = [depot_coords] + [cust['coords'] for cust in customer_data_list]
    num_locations = len(locations)
    time_matrix = np.zeros((num_locations, num_locations), dtype=int)
    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            travel_val = int(calculate_travel_time(locations[i], locations[j], params))
            time_matrix[i, j] = travel_val
            time_matrix[j, i] = travel_val
    
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node, to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def time_plus_service_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        visit_time = 0
        if to_node > 0:
            visit_time = int(customer_data_list[to_node - 1]['Visit Time (min)'])
        return time_matrix[from_node, to_node] + visit_time
    
    transit_plus_service_index = routing.RegisterTransitCallback(time_plus_service_callback)
    routing.AddDimension(transit_plus_service_index, 0, 99999, True, 'TimeDimension')
    time_dimension = routing.GetDimensionOrDie('TimeDimension')
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        total_workload = solution.Min(time_dimension.CumulVar(routing.End(0)))
        total_travel_time = total_workload - total_visit_time
        return total_workload, total_travel_time
    else:
        return 99999, 99999

def get_cluster_tio(cluster_visits):
    """Tính tổng TIO của 1 cluster."""
    return sum(v['Visit Time (min)'] for v in cluster_visits)

def get_cluster_centroids(clusters):
    """Tính tâm (trung bình tọa độ) của các cluster."""
    centroids = {}
    for day, visits in clusters.items():
        if visits:
            coords = [v['coords'] for v in visits if not pd.isna(v['coords'][0])]
            if coords:
                centroids[day] = np.mean(coords, axis=0)
    return centroids

# --- 2c. Hàm Lập lịch chính (Step 4, 5, 6 - "Bọc lại") ---
# === HÀM ĐÃ SỬA PROGRESS BAR (PHIÊN BẢN MỚI 70/30) ===
def run_master_scheduler(df_cust, df_dist, selected_route_ids, params, main_progress_bar):
    """
    Hàm chính chạy toàn bộ logic lập lịch (Steps 4, 5, 6)
    Chỉ chạy cho các 'selected_route_ids'
    """
    
    # Lọc data theo các route đã chọn
    df_cust_filtered = df_cust[df_cust['RouteID'].isin(selected_route_ids)].copy()
    
    # --- BẮT ĐẦU LOGIC GỐC STEP 4 ---
    print("Pre-assigning Freq 1 & 2 customers to weeks...")
    low_freq_assignments = {}
    freq_counters = {1: {'W1': 0, 'W2': 0, 'W3': 0, 'W4': 0},
                     2: {'W1W3': 0, 'W2W4': 0}}
    for index, cust in df_cust_filtered.iterrows():
        freq = cust['Frequency']
        cust_code = cust['Customer code']
        if freq == 2:
            if freq_counters[2]['W1W3'] <= freq_counters[2]['W2W4']:
                low_freq_assignments[cust_code] = ['W1', 'W3']
                freq_counters[2]['W1W3'] += 1
            else:
                low_freq_assignments[cust_code] = ['W2', 'W4']
                freq_counters[2]['W2W4'] += 1
        elif freq == 1:
            min_week = min(freq_counters[1], key=freq_counters[1].get)
            low_freq_assignments[cust_code] = [min_week]
            freq_counters[1][min_week] += 1
    
    depot_lookup = df_dist.set_index('RouteID').to_dict('index')
    grouped_by_route = df_cust_filtered.groupby('RouteID')
    final_clusters_by_route = {}
    balancer_stop_reasons = {} # Bỏ qua trong output, nhưng vẫn cần cho logic
    
    # === SỬA LOGIC PROGRESS BAR (70/30) ===
    total_routes_to_process = len(selected_route_ids)
    
    # 1. Khai báo trọng số
    CLUSTERING_WEIGHT = 0.7  # 70% cho Phân cụm (nặng)
    SEQUENCING_WEIGHT = 1.0 - CLUSTERING_WEIGHT # 30% cho Sắp xếp (nhẹ)
    
    # Xóa các biến không cần thiết
    # total_tasks = total_routes_to_process * 2 
    # task_counter = 0
    
    main_progress_bar.progress(0, text="Đang xử lý... 0%")
    # === HẾT SỬA ===
    
    # --- BẮT ĐẦU LOGIC GỐC STEP 5 (THE BRAIN) ---
    for i, (route_id, route_df) in enumerate(grouped_by_route):
        
        # === SỬA: CẬP NHẬT TIẾN TRÌNH (STEP 1/2) ===
        percent_step1 = i / total_routes_to_process
        percent_complete = percent_step1 * CLUSTERING_WEIGHT # Tính % trên 70%
        
        progress_text = f"Đang xử lý... {percent_complete:.0%}" # SỬA: Text đơn giản
        main_progress_bar.progress(percent_complete, text=progress_text)
        # === HẾT SỬA ===
        
        if route_id not in depot_lookup: continue
        try:
            depot_coords = (depot_lookup[route_id].get('Latitude'), depot_lookup[route_id].get('Longitude'))
            if pd.isna(depot_coords[0]): continue
        except Exception as e:
            continue
            
        final_clusters_for_this_route = {}

        for week_idx, week in enumerate(params['WEEKS']): # SỬA: Thêm enumerate
            
            # 5b. "Explode" Visits
            weekly_visit_items = []
            for index, cust in route_df.iterrows():
                cust_code = cust['Customer code']
                freq = cust['Frequency']
                tio = params['VISIT_TIME_MAP'].get(cust['Customer Type'], params['VISIT_TIME_MAP']['default'])
                visits_per_week = params['FREQ_TO_WEEKLY_VISITS'].get(freq, 0)
                
                if visits_per_week > 0:
                    for v_num in range(visits_per_week):
                        weekly_visit_items.append({
                            'Customer code': cust_code, 'coords': (cust['Latitude'], cust['Longitude']),
                            'Visit Time (min)': tio, 'Frequency': freq, 'Visit_Num': v_num,
                            **cust.to_dict() # Thêm tất cả data gốc để hover
                        })
                elif cust_code in low_freq_assignments:
                    if week in low_freq_assignments[cust_code]:
                        weekly_visit_items.append({
                            'Customer code': cust_code, 'coords': (cust['Latitude'], cust['Longitude']),
                            'Visit Time (min)': tio, 'Frequency': freq, 'Visit_Num': 0,
                            **cust.to_dict() # Thêm tất cả data gốc để hover
                        })
            
            if not weekly_visit_items:
                for day in params['DAYS_OF_WEEK']:
                    final_clusters_for_this_route[f"{week}-{day}"] = []
                continue

            # 5c. K-Means Constrained
            df_week_visits = pd.DataFrame(weekly_visit_items)
            coords = np.array([v['coords'] for v in weekly_visit_items])
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)

            total_visits_in_week = len(df_week_visits)
            avg_visits_per_day = total_visits_in_week / 5.5
            min_size = int(avg_visits_per_day * (1 - params['COUNT_TOLERANCE_PERCENT']))
            max_size = int(avg_visits_per_day * (1 + params['COUNT_TOLERANCE_PERCENT']))
            
            if min_size < 1: min_size = 1
            if max_size * params['N_CLUSTERS_PER_WEEK'] < total_visits_in_week:
                 max_size = int(total_visits_in_week / params['N_CLUSTERS_PER_WEEK']) + 2
            if min_size * params['N_CLUSTERS_PER_WEEK'] > total_visits_in_week:
                min_size = 1

            kmeans = KMeansConstrained(
                n_clusters=params['N_CLUSTERS_PER_WEEK'],
                size_min=min_size,
                size_max=max_size,
                random_state=42, n_init=50
            )
            try:
                df_week_visits['Cluster_ID'] = kmeans.fit_predict(coords_scaled)
            except Exception as e:
                # Fallback
                visit_time_weights = np.array([v['Visit Time (min)'] for v in weekly_visit_items])
                kmeans_fallback = KMeans(n_clusters=params['N_CLUSTERS_PER_WEEK'], random_state=42, n_init=50)
                df_week_visits['Cluster_ID'] = kmeans_fallback.fit_predict(coords_scaled, sample_weight=visit_time_weights)

            # Smart-Assign Saturday
            temp_clusters = {i: [] for i in range(params['N_CLUSTERS_PER_WEEK'])}
            for index, row in df_week_visits.iterrows():
                temp_clusters[row['Cluster_ID']].append(weekly_visit_items[index])

            cluster_workloads = {}
            for cluster_id, visits in temp_clusters.items():
                if not visits:
                    cluster_workloads[cluster_id] = 0
                else:
                    workload, _ = get_optimal_route_workload(visits, depot_coords, params)
                    cluster_workloads[cluster_id] = workload
            
            smallest_workload_cluster_id = min(cluster_workloads, key=cluster_workloads.get)
            cluster_map = {smallest_workload_cluster_id: 'Sat'}
            weekdays_to_assign = [d for d in params['DAYS_OF_WEEK'] if d != 'Sat']
            cluster_id_to_assign = [i for i in range(params['N_CLUSTERS_PER_WEEK']) if i != smallest_workload_cluster_id]
            for i_day in range(len(weekdays_to_assign)):
                cluster_map[cluster_id_to_assign[i_day]] = weekdays_to_assign[i_day]

            df_week_visits['Day'] = df_week_visits['Cluster_ID'].map(cluster_map)
            clusters = {day: df_week_visits[df_week_visits['Day'] == day].to_dict('records') for day in params['DAYS_OF_WEEK']}

            # 5d. Frequency Fix
            freq_fix_made = True
            while freq_fix_made:
                freq_fix_made = False
                cluster_centroids = get_cluster_centroids(clusters)
                if not cluster_centroids: break
                for day, visits in clusters.items():
                    if day not in cluster_centroids: continue
                    cust_codes = [v['Customer code'] for v in visits]
                    duplicates = {c for c in cust_codes if cust_codes.count(c) > 1}
                    for dup_code in duplicates:
                        visits_to_move = [v for v in visits if v['Customer code'] == dup_code][1:]
                        for v_move in visits_to_move:
                            day_center = cluster_centroids[day]
                            best_new_day = None
                            min_dist = np.inf
                            for neighbor_day, neighbor_visits in clusters.items():
                                if neighbor_day == day or neighbor_day not in cluster_centroids: continue
                                if tuple(sorted((day, neighbor_day))) in params['CONSECUTIVE_PAIRS']:
                                    continue
                                dist = geodesic(cluster_centroids[neighbor_day], day_center).km
                                if dist < min_dist:
                                    tio_neighbor = get_cluster_tio(neighbor_visits)
                                    if tio_neighbor + v_move['Visit Time (min)'] <= params['DAY_CAPACITY_MAP'][neighbor_day]:
                                        min_dist = dist
                                        best_new_day = neighbor_day
                            if best_new_day:
                                clusters[best_new_day].append(v_move)
                                clusters[day].remove(v_move)
                                freq_fix_made = True
                                break
                        if freq_fix_made: break
                    if freq_fix_made: break
            
            # 5e. TBO-Balancing Loop
            stop_reason = f"Timed Out ({params['BALANCE_ITERATION_LIMIT']} iter)"
            cust_schedule_lookup = {}
            for day, visits in clusters.items():
                for v in visits:
                    cust_code = v['Customer code']
                    if cust_code not in cust_schedule_lookup:
                        cust_schedule_lookup[cust_code] = set()
                    cust_schedule_lookup[cust_code].add(day)

            for i_balance in range(params['BALANCE_ITERATION_LIMIT']):
                cluster_workloads = {}
                for day, visits in clusters.items():
                    workload, _ = get_optimal_route_workload(visits, depot_coords, params)
                    cluster_workloads[day] = workload
                
                cluster_centroids = get_cluster_centroids(clusters)
                if not cluster_centroids:
                    stop_reason = "Empty Clusters"
                    break
                
                cluster_deltas = {}
                active_deltas = []
                for day, workload in cluster_workloads.items():
                    if day in cluster_centroids:
                        target = params['DAY_CAPACITY_MAP'][day]
                        delta = workload - target
                        cluster_deltas[day] = delta
                        active_deltas.append(delta)
                
                if not active_deltas:
                    stop_reason = "Empty Clusters"
                    break
                
                max_overage = max(active_deltas)
                max_over_day = max(cluster_deltas, key=cluster_deltas.get)
                
                best_neighbor_day = None
                min_neighbor_delta = np.inf
                for day, center in cluster_centroids.items():
                    if day == max_over_day: continue
                    neighbor_delta = cluster_deltas.get(day, -np.inf)
                    if neighbor_delta < min_neighbor_delta:
                        min_neighbor_delta = neighbor_delta
                        best_neighbor_day = day
                
                if best_neighbor_day is None or cluster_deltas[max_over_day] <= cluster_deltas[best_neighbor_day]:
                    stop_reason = "Stuck (Bad Trade)"
                    break
                
                if max_overage <= params['BALANCE_TOLERANCE_MIN']:
                    stop_reason = f"Balanced (All days within +{params['BALANCE_TOLERANCE_MIN']}min)"
                    break
                
                visits_on_max_day = clusters[max_over_day]
                if not visits_on_max_day: # Thêm 1 bước check
                    stop_reason = "Stuck (Empty Max Day)"
                    break
                
                neighbor_center = cluster_centroids[best_neighbor_day]
                visits_on_max_day.sort(key=lambda v: geodesic(v['coords'], neighbor_center).km)
                
                move_made = False
                for customer_to_move in visits_on_max_day:
                    cust_code = customer_to_move['Customer code']
                    destination_day = best_neighbor_day
                    if destination_day in cust_schedule_lookup.get(cust_code, set()):
                        continue
                    existing_days = cust_schedule_lookup.get(cust_code, set()) - {max_over_day}
                    is_illegal_consecutive = False
                    for existing_day in existing_days:
                        if tuple(sorted((destination_day, existing_day))) in params['CONSECUTIVE_PAIRS']:
                            is_illegal_consecutive = True
                            break
                    if is_illegal_consecutive:
                        continue

                    clusters[best_neighbor_day].append(customer_to_move)
                    clusters[max_over_day].remove(customer_to_move)
                    cust_schedule_lookup[cust_code].add(destination_day)
                    cust_schedule_lookup[cust_code].remove(max_over_day)
                    move_made = True
                    break
                
                if not move_made:
                    stop_reason = "Stuck (No Legal Moves)"
                    break
            
            balancer_stop_reasons[f"{route_id}-{week}"] = stop_reason

            # 5f. Save Final Clusters
            for day, visits in clusters.items():
                week_day = f"{week}-{day}"
                final_clusters_for_this_route[week_day] = visits

        final_clusters_by_route[route_id] = final_clusters_for_this_route
        
    print("\n--- All Routes Clustered and Re-Balanced ---")
    
    # === SỬA: XÓA task_counter ===
    # (Không cần dòng 'task_counter = total_routes_to_process' nữa)
    # === HẾT SỬA ===

    # --- BẮT ĐẦU LOGIC GỐC STEP 6 (FINAL SEQUENCING) ---
    print("\n--- Starting Final Sequencing ---")
    all_results = []
    
    total_routes_to_sequence = len(final_clusters_by_route)
    
    for i, (route_id, clusters) in enumerate(final_clusters_by_route.items()): 
        
        # === SỬA: CẬP NHẬT TIẾN TRÌNH (STEP 2/2) ===
        percent_step2 = (i / total_routes_to_sequence) if total_routes_to_sequence > 0 else 0 # Tránh lỗi chia cho 0
        percent_complete = CLUSTERING_WEIGHT + (percent_step2 * SEQUENCING_WEIGHT) # Tính % từ 70% -> 100%
        
        progress_text = f"Đang xử lý... {percent_complete:.0%}" # SỬA: Text đơn giản
        main_progress_bar.progress(percent_complete, text=progress_text)
        # === HẾT SỬA ===

        if route_id not in depot_lookup: continue
        depot_coords = (depot_lookup[route_id].get('Latitude'), depot_lookup[route_id].get('Longitude'))
        
        for week_day, visits in clusters.items():
            if not visits: continue
            
            locations = [depot_coords] + [v['coords'] for v in visits]
            num_locations = len(locations)
            manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
            routing = pywrapcp.RoutingModel(manager)
            time_matrix = np.zeros((num_locations, num_locations), dtype=int)
            
            for i_loc in range(num_locations):
                for j_loc in range(i_loc + 1, num_locations):
                    travel_val = int(calculate_travel_time(locations[i_loc], locations[j_loc], params))
                    time_matrix[i_loc, j_loc] = travel_val
                    time_matrix[j_loc, i_loc] = travel_val
            
            def time_callback_final(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return time_matrix[from_node, to_node]
            
            transit_callback_index_final = routing.RegisterTransitCallback(time_callback_final)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index_final)
            search_parameters_final = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters_final.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            solution = routing.SolveWithParameters(search_parameters_final)

            if solution:
                sequence = 1
                index = routing.Start(0)
                prev_location_coords = depot_coords
                agg_distance = 0
                agg_travel_time = 0
                agg_visit_time = 0
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:
                        cust_data = visits[node_index - 1]
                        current_location_coords = cust_data['coords']
                        dist_km = geodesic(prev_location_coords, current_location_coords).kilometers * params['CIRCUTIY_FACTOR']
                        travel_min = (dist_km / params['AVG_SPEED_KMH']) * 60
                        visit_time = cust_data['Visit Time (min)']
                        agg_distance += dist_km
                        agg_travel_time += travel_min
                        agg_visit_time += visit_time
                        total_workload = agg_travel_time + agg_visit_time
                        week, day = week_day.split('-')
                        
                        # Chuẩn bị data cho hover trên bản đồ
                        hover_data = cust_data.copy()
                        # Thêm kết quả tính toán
                        result_data = {
                            'Week': week, 'Day': day, 'Week&Day': week_day,
                            'Sequence': sequence,
                            'Distance from previous (km)': round(dist_km, 2),
                            'Travel time from previous (min)': round(travel_min, 2),
                            'Visit Time (min)': visit_time,
                            'Total Workload (min)': round(total_workload, 2),
                            'Aggregate distance (km)': round(agg_distance, 2),
                            'Aggregate travel time (min)': round(agg_travel_time, 2),
                        }
                        # Gộp data gốc và data kết quả
                        hover_data.update(result_data)
                        all_results.append(hover_data)
                        
                        sequence += 1
                        prev_location_coords = current_location_coords
                    index = solution.Value(routing.NextVar(index))
            
    print(f"\n--- Master Scheduling Process Complete ---")

    if not all_results:
        return pd.DataFrame(), pd.DataFrame() 

    df_final_output = pd.DataFrame(all_results)
    
    # 1. Daily Workload Summary (theo yêu cầu)
    df_summary = df_final_output.groupby(['RouteID', 'Week&Day']).agg(
        Total_TIO_min=('Visit Time (min)', 'sum'),
        Total_TBO_min=('Travel time from previous (min)', 'sum'),
        Num_Customers=('Customer code', 'count')
    ).reset_index()
    df_summary['Total_Workload_min'] = df_summary['Total_TIO_min'] + df_summary['Total_TBO_min']
    df_summary['Total_TIO (h)'] = (df_summary['Total_TIO_min'] / 60).round(2)
    df_summary['Total_TBO (h)'] = (df_summary['Total_TBO_min'] / 60).round(2)
    df_summary['Total_Workload (h)'] = (df_summary['Total_Workload_min'] / 60).round(2)
    df_summary = df_summary[['RouteID', 'Week&Day', 'Num_Customers', 'Total_TIO (h)', 'Total_TBO (h)', 'Total_Workload (h)']]
    df_summary = df_summary.sort_values(by=['RouteID', 'Total_Workload (h)'])

    df_final_output = df_final_output.sort_values(
        by=['RouteID', 'Week', 'Day', 'Sequence']
    ).reset_index(drop=True)

    # Trả về 2 dataframe
    return df_final_output, df_summary 


################################################################################
# ------------------------------------------------------------------------------
# PHẦN 3: CÁC HÀM HỖ TRỢ GIAO DIỆN (UI HELPERS)
# ------------------------------------------------------------------------------
################################################################################

# --- Các cột bắt buộc cho việc xác thực ---
REQUIRED_COLS_CUST = {
    'RouteID': 'RouteID',
    'Customer code': 'Customer code',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'Frequency': 'Frequency',
    'Customer Type': 'Customer Type'
}
REQUIRED_COLS_DIST = {
    'RouteID': 'RouteID',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude'
}

@st.cache_data
def create_template_excel(cols_dict):
    """Tạo file Excel template trong bộ nhớ."""
    df_template = pd.DataFrame(columns=list(cols_dict.keys()))
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, sheet_name='Template', index=False)
    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def to_excel_output(df_master, df_summary):
    """Tạo file Excel kết quả (2 sheet) trong bộ nhớ."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_master.to_excel(writer, sheet_name='Master_Schedule', index=False)
        df_summary.to_excel(writer, sheet_name='Daily_Workload_Summary', index=False)
    processed_data = output.getvalue()
    return processed_data

def find_default_index(column_list, target_name):
    """Hàm thông minh tự tìm cột để map, ví dụ 'customer code' khớp với 'Customer code'."""
    target_lower = str(target_name).lower().strip()
    for i, col in enumerate(column_list):
        if str(col).lower().strip() == target_lower:
            return i
    return 0 # Nếu không tìm thấy, trả về cột đầu tiên

def validate_data(df_cust, df_dist):
    """Hàm xác thực dữ liệu theo yêu cầu của bạn."""
    errors = []
    warnings = []
    
    # 1. Kiểm tra trùng lặp
    cust_duplicates = df_cust.duplicated(subset=['Customer code']).sum()
    if cust_duplicates > 0:
        warnings.append(f"Có {cust_duplicates} khách hàng bị trùng lặp 'Customer code' (sẽ giữ lại bản đầu tiên).")
        df_cust = df_cust.drop_duplicates(subset=['Customer code'], keep='first')
        
    dist_duplicates = df_dist.duplicated(subset=['RouteID']).sum()
    if dist_duplicates > 0:
        warnings.append(f"Có {dist_duplicates} NPP bị trùng lặp 'RouteID' (sẽ giữ lại bản đầu tiên).")
        df_dist = df_dist.drop_duplicates(subset=['RouteID'], keep='first')

    # 2. Kiểm tra thiếu Lat/Long
    cust_missing_latlong = df_cust['Latitude'].isna() | df_cust['Longitude'].isna()
    cust_missing_count = cust_missing_latlong.sum()
    if cust_missing_count > 0:
        warnings.append(f"Có {cust_missing_count} KH bị thiếu Lat/Long (sẽ bị loại bỏ khỏi tính toán).")
        df_cust = df_cust[~cust_missing_latlong] # Loại bỏ KH
        
    dist_missing_latlong = df_dist['Latitude'].isna() | df_dist['Longitude'].isna()
    dist_missing_count = dist_missing_latlong.sum()
    if dist_missing_count > 0:
        missing_routes = df_dist[dist_missing_latlong]['RouteID'].tolist()
        errors.append(f"Có {dist_missing_count} NPP bị thiếu Lat/Long (VD: {missing_routes[:3]}). Các route này sẽ không được xử lý.")
        df_dist = df_dist[~dist_missing_latlong] # Loại bỏ NPP

    # 3. Kiểm tra RouteID không khớp
    cust_routes = set(df_cust['RouteID'].unique())
    dist_routes = set(df_dist['RouteID'].unique())
    
    routes_in_cust_not_in_dist = list(cust_routes - dist_routes)
    if routes_in_cust_not_in_dist:
        errors.append(f"Có {len(routes_in_cust_not_in_dist)} RouteID trong file Customers nhưng không có NPP (VD: {routes_in_cust_not_in_dist[:3]}). Các KH này sẽ bị bỏ qua.")
        
    return df_cust, df_dist, errors, warnings


# === HÀM BẢN ĐỒ MỚI (DÙNG STREAMLIT-FOLIUM) ===
# === SỬA: ĐỔI TÊN HÀM ===
def create_folium_map(df_filtered):
    """Tạo bản đồ Leaflet với đường nối, sequence, màu sắc và hover."""
    
    if df_filtered.empty:
        st.info("Không có dữ liệu để hiển thị trên bản đồ với bộ lọc này.")
        return
        
    # SỬA LỖI SORTING: KHÔNG convert 'Sequence' sang string ở đây.
    # Giữ nó ở dạng số (int) để sort_values hoạt động đúng.
    # df_filtered['Sequence'] = df_filtered['Sequence'].astype(str) # XÓA DÒNG NÀY

    # 1. Tạo màu cho các ngày (Leaflet dùng mã Hex)
    color_map = {
        'Mon': '#FF0000', # Red
        'Tue': '#008000', # Green
        'Wed': '#0000FF', # Blue
        'Thu': '#FFA500', # Orange
        'Fri': '#800080', # Purple
        'Sat': '#000000', # Black
    }
    df_filtered['color'] = df_filtered['Day'].apply(lambda day: color_map.get(day, "#808080")) # Xám
    
    # 2. Cấu hình bản đồ (OpenStreetMap như bạn yêu cầu)
    map_center = [df_filtered['Latitude'].mean(), df_filtered['Longitude'].mean()]
    
    # SỬA: Dùng "OpenStreetMap" (mặc định) để có bản đồ màu
    m = folium.Map(
        location=map_center, 
        zoom_start=12, 
        tiles="OpenStreetMap" # Yêu cầu bản đồ màu
    )
    
    # 3. Helper function để tạo HTML cho popup
    def generate_tooltip_html(row):
        html = "<b>Thông tin Điểm bán</b><br/>"
        # Bỏ các cột nội bộ
        cols_to_drop = ['coords', 'color', 'tooltip'] 
        row_data = row.drop(labels=cols_to_drop, errors='ignore')
        for col, val in row_data.items():
            html += f"<b>{col}:</b> {val}<br/>"
        return html

    # 4. Lặp qua từng nhóm (route/week/day) để tạo đường và điểm
    for (route, week, day), group in df_filtered.groupby(['RouteID', 'Week', 'Day']):
        
        color = group['color'].iloc[0]
        
        # 5a. SỬA LỖI SORTING: Sắp xếp theo cột SỐ 'Sequence'
        group_sorted = group.sort_values('Sequence') 
        
        # 5b. Tạo đường nối (Polyline)
        locations = group_sorted[['Latitude', 'Longitude']].values.tolist()
        folium.PolyLine(
            locations=locations, 
            color=color, 
            weight=3,
            opacity=0.7
        ).add_to(m)
        
        # 5c. Tạo các điểm (Marker + DivIcon)
        for _, row in group_sorted.iterrows(): # Dùng group_sorted
            
            # 1. Tạo popup khi click (giữ nguyên)
            popup = folium.Popup(generate_tooltip_html(row), max_width=300)
            
            # 2. Lấy số sequence và màu sắc
            seq_str = str(row['Sequence'])
            # 'color' đã được định nghĩa ở group level, ta dùng luôn
            
            # 3. Tạo HTML cho icon: 1 hình tròn (bằng div) chứa số
            icon_html = f"""
            <div style="
                font-size: 10px;
                font-weight: bold;
                color: white;
                background: {color};
                border-radius: 50%;
                width: 16px;      /* radius=8 -> diameter=16 */
                height: 16px;     /* radius=8 -> diameter=16 */
                text-align: center;
                line-height: 16px; /* Căn giữa text theo chiều dọc */
            ">
                {seq_str}
            </div>
            """
        
            # 4. Tạo DivIcon
            icon = folium.DivIcon(
                icon_size=(16, 16),
                icon_anchor=(8, 8), # Anker (mỏ neo) ở tâm icon
                html=icon_html
            )
            
            # 5. Tạo Marker (thay vì CircleMarker) với icon tùy chỉnh
            folium.Marker(
                location=(row['Latitude'], row['Longitude']),
                icon=icon,
                popup=popup
            ).add_to(m)
    
    # 7. Render bản đồ
    st_folium(
        m, 
        height=600, 
        use_container_width=True # SỬA: Thêm để bản đồ vừa khung
    )
    
    # 8. Chú giải (Legend) - Vẫn giữ nguyên
    st.subheader("Chú giải màu theo ngày")
    legend_html = ""
    for day, color_hex in color_map.items():
        # Dùng mã hex
        legend_html += f'<span style="background-color: {color_hex}; border-radius: 50%; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></span> {day} &nbsp;&nbsp;'
    st.markdown(legend_html, unsafe_allow_html=True)


################################################################################
# ------------------------------------------------------------------------------
# PHẦN 4: KHỞI TẠO "SESSION STATE"
# ------------------------------------------------------------------------------
################################################################################

if 'stage' not in st.session_state:
    st.session_state.stage = '1_upload' # Giai đoạn của app
if 'df_cust' not in st.session_state:
    st.session_state.df_cust = None # Data khách hàng (đã map)
if 'df_dist' not in st.session_state:
    st.session_state.df_dist = None # Data NPP (đã map)
if 'validation_errors' not in st.session_state:
    st.session_state.validation_errors = []
if 'validation_warnings' not in st.session_state:
    st.session_state.validation_warnings = []
if 'validation_success' not in st.session_state:
    st.session_state.validation_success = False # Cờ cho luồng xác thực
if 'validated_route_ids' not in st.session_state:
    st.session_state.validated_route_ids = []
if 'df_final_output' not in st.session_state:
    st.session_state.df_final_output = None # Kết quả cuối cùng
if 'df_summary' not in st.session_state:
    st.session_state.df_summary = None # Bảng summary
# XÓA: df_input_summary


################################################################################
# ------------------------------------------------------------------------------
# PHẦN 5: GIAO DIỆN NGƯỜI DÙNG (UI) - SIDEBAR (PHẦN 1)
# ------------------------------------------------------------------------------
################################################################################

st.sidebar.title("Bảng điều khiển 🚀")

# --- 1. Tải file mẫu ---
st.sidebar.subheader("1. Tải file mẫu")
template_cust_bytes = create_template_excel(REQUIRED_COLS_CUST)
st.sidebar.download_button(
    label="Tải mẫu Customers.xlsx",
    data=template_cust_bytes,
    file_name="Customers_Template.xlsx"
)
template_dist_bytes = create_template_excel(REQUIRED_COLS_DIST)
st.sidebar.download_button(
    label="Tải mẫu Distributors.xlsx",
    data=template_dist_bytes,
    file_name="Distributors_Template.xlsx"
)
st.sidebar.caption("File mẫu chứa các cột bắt buộc có dấu (*)")

# --- 2. Tải dữ liệu ---
st.sidebar.subheader("2. Tải dữ liệu (Tối đa 200MB)")
uploaded_cust_file = st.sidebar.file_uploader("Tải file Customers", type=['xlsx', 'xls'])
uploaded_dist_file = st.sidebar.file_uploader("Tải file Distributors", type=['xlsx', 'xls'])
st.sidebar.info("Sau khi tải lên, chọn cột tương ứng.")

# --- 3. Tham số (Read-only) ---
st.sidebar.subheader("3. Tham số (Tham khảo)")
with st.sidebar.expander("Xem tham số tính toán"):
    st.markdown(f"**Tốc độ trung bình:** `{BUSINESS_PARAMS['AVG_SPEED_KMH']}` km/h")
    st.markdown("---")
    st.markdown("**Thời gian Viếng thăm (phút):**")
    for key, val in BUSINESS_PARAMS['VISIT_TIME_MAP'].items():
        st.markdown(f"- **{key}:** `{val}` phút")


################################################################################
# ------------------------------------------------------------------------------
# PHẦN 6: GIAO DIỆN NGƯỜI DÙNG (UI) - MAIN AREA (PHẦN 2)
# ------------------------------------------------------------------------------
################################################################################

st.title("Ứng dụng Lập Lịch Viếng Thăm (Visit Planner)")

# ------------------------------------------
# GIAI ĐOẠN 1: ÁNH XẠ & XÁC THỰC
# ------------------------------------------
if st.session_state.stage == '1_upload':
    st.header("Bước 1: Tải lên dữ liệu")
    
    if uploaded_cust_file and uploaded_dist_file:
        
        # Đọc data (chưa lưu) để lấy tên cột
        try:
            df_cust_raw = pd.read_excel(uploaded_cust_file)
            df_dist_raw = pd.read_excel(uploaded_dist_file)
            cust_cols = list(df_cust_raw.columns)
            dist_cols = list(df_dist_raw.columns)

            with st.form("mapping_form"):
                st.subheader("File Customers")
                mapping_cust = {}
                cols_cust_ui = st.columns(len(REQUIRED_COLS_CUST))
                for i, (key, default_name) in enumerate(REQUIRED_COLS_CUST.items()):
                    with cols_cust_ui[i]:
                        mapping_cust[key] = st.selectbox(
                            f"{key} (*)", 
                            cust_cols, 
                            index=find_default_index(cust_cols, default_name)
                        )
                
                st.subheader("File Distributors")
                mapping_dist = {}
                cols_dist_ui = st.columns(len(REQUIRED_COLS_DIST))
                for i, (key, default_name) in enumerate(REQUIRED_COLS_DIST.items()):
                    with cols_dist_ui[i]:
                        mapping_dist[key] = st.selectbox(
                            f"{key} (*)", 
                            dist_cols, 
                            index=find_default_index(dist_cols, default_name)
                        )

                submitted = st.form_submit_button("Xác thực & Tóm tắt Dữ liệu")
                
                if submitted:
                    with st.spinner("Đang xác thực dữ liệu..."):
                        # Reset cờ
                        st.session_state.validation_success = False
                        
                        # Đổi tên cột theo map
                        df_cust_mapped = df_cust_raw[list(mapping_cust.values())].rename(columns={v: k for k, v in mapping_cust.items()})
                        df_dist_mapped = df_dist_raw[list(mapping_dist.values())].rename(columns={v: k for k, v in mapping_dist.items()})
                        
                        # Thêm các cột khác từ file gốc (dùng cho hover)
                        other_cust_cols = [col for col in cust_cols if col not in mapping_cust.values()]
                        df_cust_mapped = pd.concat([df_cust_mapped, df_cust_raw[other_cust_cols]], axis=1)

                        # Chạy validation
                        df_cust_validated, df_dist_validated, errors, warnings = validate_data(df_cust_mapped, df_dist_mapped)
                        
                        st.session_state.validation_errors = errors
                        st.session_state.validation_warnings = warnings
                        
                        if errors:
                            st.error("Dữ liệu có lỗi. Vui lòng sửa file và tải lại:")
                            for err in errors:
                                st.error(f"- {err}")
                        else:
                            st.success("Xác thực thành công!")
                            for warn in warnings:
                                st.warning(f"- {warn}")
                            
                            # Lưu data đã validate vào state
                            st.session_state.df_cust = df_cust_validated
                            st.session_state.df_dist = df_dist_validated
                            st.session_state.validated_route_ids = list(df_dist_validated['RouteID'].unique())
                            
                            st.metric("Tổng số RouteID (đã lọc)", len(st.session_state.validated_route_ids))
                            st.metric("Tổng số Khách hàng (đã lọc)", len(st.session_state.df_cust))
                            
                            # Đặt cờ thành công
                            st.session_state.validation_success = True
                            
            # === NÚT TIẾP TỤC (ĐÃ SỬA) ===
            # Nằm bên ngoài form, chỉ hiển thị sau khi nhấn nút "Xác thực" và thành công
            if st.session_state.get('validation_success', False):
                st.markdown("---")
                if st.button("Tiếp tục đến Bước 2: Xếp lịch viếng thăm ➡️"):
                    st.session_state.stage = '2_planning'
                    st.session_state.validation_success = False # Reset cờ
                    st.rerun() # SỬA: Dùng hàm mới

        except Exception as e:
            st.error(f"Không thể đọc file. Lỗi: {e}")
            
    else:
        st.info("Vui lòng tải lên cả 2 file Customers và Distributors ở thanh bên trái.")

# ------------------------------------------
# GIAI ĐOẠN 2: LẬP LỊCH
# ------------------------------------------
elif st.session_state.stage == '2_planning':
    st.header("Bước 2: Chọn RouteID muốn xếp lịch viếng thăm")
    
    selected_routes = st.multiselect(
        "Chọn RouteID", 
        options=st.session_state.validated_route_ids,
        default=st.session_state.validated_route_ids[:1] # Mặc định chọn 1 route
    )
    
    if len(selected_routes) > 15:
        st.warning("Bạn đã chọn hơn 15 route. Quá trình xử lý có thể mất nhiều thời gian hơn.")
    st.caption("Lưu ý: Nên chọn dưới 15 route một lúc để đảm bảo app chạy tốt.")
    
    # === KHỐI NÚT BẤM (ĐÃ SỬA) ===
    if st.button("🚀 Bắt đầu xếp lịch", disabled=(not selected_routes)):
        # Chạy logic TẠI ĐÂY
        main_progress_bar = st.progress(0, text="Bắt đầu...") # SỬA: Đổi tên thanh bar
        
        with st.spinner("Đang xếp lịch viếng thăm... (Có thể mất vài phút)"):
            try:
                # SỬA: Chỉ nhận 2 dataframe
                df_final, df_sum = run_master_scheduler(
                    st.session_state.df_cust,
                    st.session_state.df_dist,
                    selected_routes,
                    BUSINESS_PARAMS,
                    main_progress_bar # SỬA: Gửi đúng thanh bar
                )
                
                # Lưu kết quả vào state
                st.session_state.df_final_output = df_final
                st.session_state.df_summary = df_sum
                # XÓA: df_input_summary
                
                st.session_state.stage = '3_results' # Đánh dấu là đã có kết quả
                main_progress_bar.progress(1.0, text="Hoàn tất!") # SỬA
                st.success("Hoàn tất! Xem kết quả...")
                time.sleep(1) # Chờ 1s cho đẹp
                st.rerun() # SỬA: Dùng hàm mới
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi nghiêm trọng khi lập lịch: {e}")
                main_progress_bar.progress(1.0, text="Thất bại!") # SỬA
                # (Vẫn ở GĐ 2 để người dùng có thể thử lại)


# ------------------------------------------
# GIAI ĐOẠN 3: XEM KẾT QUẢ (PHIÊN BẢN ĐÃ SỬA)
# ------------------------------------------
elif st.session_state.stage == '3_results':
    st.header("Kết quả Lập lịch")
    
    # Lấy data từ state
    df_final = st.session_state.df_final_output
    df_summary = st.session_state.df_summary
    
    if df_final is None or df_final.empty:
        st.error("Không có kết quả nào được tạo ra. Vui lòng thử lại.")
        # Thêm nút để quay lại
        if st.button("Xếp lịch lại"):
            st.session_state.stage = '2_planning'
            st.session_state.df_final_output = None # Xóa kết quả cũ
            st.rerun()
        
    else:
        # --- Nút Tải về (đặt lên đầu cho dễ thấy) ---
        excel_bytes = to_excel_output(df_final, df_summary)
        st.download_button(
            label="💾 Tải File Excel Kết Quả (2 sheet)",
            data=excel_bytes,
            file_name="MASTER_SCHEDULE_OUTPUT.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # --- Chia Tab ---
        tab1, tab2 = st.tabs(["🗺️ Bản đồ Lịch trình", "📊 Chi tiết lịch viếng thăm"])

        # --- Tab 1: Bản đồ ---
        with tab1:
            st.subheader("Bản đồ Lịch trình Viếng thăm")
            
            # === SỬA BỘ LỌC BẢN ĐỒ ===
            # Lấy các giá trị duy nhất
            all_routes = df_final['RouteID'].unique()
            all_weeks = df_final['Week'].unique()
            all_days = df_final['Day'].unique()
            
            # Bộ lọc riêng cho Bản đồ
            with st.form(key="map_filters"):
                st.write("Filter")
                map_cols = st.columns([2, 1, 1, 1])
                with map_cols[0]:
                    map_route = st.multiselect(
                        "RouteID", 
                        options=all_routes, 
                        default=[all_routes[0]] # SỬA: Chỉ chọn 1 tuyến đầu tiên
                    )
                with map_cols[1]:
                    map_week = st.multiselect(
                        "Week", 
                        options=all_weeks, 
                        default=['W1'] if 'W1' in all_weeks else [all_weeks[0]] # SỬA: Chỉ chọn W1
                    )
                with map_cols[2]:
                    map_day = st.multiselect(
                        "Day", 
                        options=all_days, 
                        default=all_days # SỬA: Chọn tất cả các ngày
                    )
                with map_cols[3]:
                    st.write("") # Thêm 1 chút khoảng trống
                    submitted = st.form_submit_button("Refresh")
            # === HẾT SỬA BỘ LỌC ===

            # Lọc data theo bộ lọc map
            df_map_filtered = df_final[
                df_final['RouteID'].isin(map_route) &
                df_final['Week'].isin(map_week) &
                df_final['Day'].isin(map_day)
            ]
            
            # SỬA: ĐỔI TÊN HÀM GỌI
            create_folium_map(df_map_filtered) 

        # --- Tab 2: Bảng Phân tích (ĐÃ SỬA THEO YÊU CẦU) ---
        with tab2:
            st.subheader("Bảng Phân tích Chi tiết")
            
            # SỬA: Bộ lọc chung cho Tab 2 (Chỉ RouteID)
            st.write("Lọc dữ liệu cho các bảng bên dưới:")
            tab_route = st.multiselect("Filter RouteID", options=df_final['RouteID'].unique(), default=df_final['RouteID'].unique())
            
            # Lọc data
            df_tab_filtered_master = df_final[
                df_final['RouteID'].isin(tab_route)
            ]
            
            # --- Dashboard Workload (Ngang hàng) ---
            st.subheader("Tóm lược") # SỬA: Đổi tên
            
            if not df_tab_filtered_master.empty:
                dash_col1, dash_col2 = st.columns(2)
                
                # Tách Week&Day cho data summary
                df_summary_copy = df_summary.copy() # Tránh lỗi cache
                df_summary_copy[['Week', 'Day']] = df_summary_copy['Week&Day'].str.split('-', expand=True)

                # SỬA: Lọc data summary (Chỉ theo Route)
                df_tab_filtered_summary = df_summary_copy[
                    df_summary_copy['RouteID'].isin(tab_route)
                ]
                
                with dash_col1:
                    st.markdown("**Số call/day**") # SỬA: Đổi tên
                    pivot_count = pd.pivot_table(
                        df_tab_filtered_summary,
                        values='Num_Customers',
                        index='Week',
                        columns='Day',
                        aggfunc='mean' # Trung bình nếu chọn nhiều route
                    ).reindex(columns=BUSINESS_PARAMS['DAYS_OF_WEEK']).fillna(0) # SỬA LỖI KEY
                    st.dataframe(pivot_count.style.format(precision=0).background_gradient(cmap='Greens'))

                with dash_col2:
                    st.markdown("**Tổng giờ làm việc/ngày (gồm thời gian viếng thăm & di chuyển)**") # SỬA: Đổi tên
                    pivot_workload = pd.pivot_table(
                        df_tab_filtered_summary,
                        values='Total_Workload (h)',
                        index='Week',
                        columns='Day',
                        aggfunc='mean' # Trung bình nếu chọn nhiều route
                    ).reindex(columns=BUSINESS_PARAMS['DAYS_OF_WEEK']).fillna(0) # SỬA LỖI KEY
                    st.dataframe(pivot_workload.style.format(precision=1).background_gradient(cmap='Reds'))
            
            else:
                st.info("Không có dữ liệu cho dashboard với bộ lọc này.")

            # --- Bảng Chi tiết ---
            st.subheader("Chi tiết Lịch trình") # SỬA: Đổi tên
            st.dataframe(df_tab_filtered_master)

        # --- Nút quay lại (SỬA THEO YÊU CẦU) ---
        
        # 1. Thêm line ngang
        st.markdown("---") 

        # 2. Thêm CSS cho nút màu đỏ
        st.markdown("""
        <style>
        button[data-testid="baseButton-primary"] {
            background-color: #FF4B4B; /* Màu đỏ */
            color: white;
            border: 1px solid #FF4B4B;
        }
        button[data-testid="baseButton-primary"]:hover {
            background-color: #D32F2F; /* Màu đỏ đậm hơn khi hover */
            color: white;
            border: 1px solid #D32F2F;
        }
        button[data-testid="baseButton-primary"]:focus {
            background-color: #FF4B4B;
            color: white;
            border: 1px solid #FF4B4B;
            box-shadow: 0 0 0 0.2rem rgba(255, 75, 75, 0.5);
        }
        </style>
        """, unsafe_allow_html=True)

        # 3. Căn giữa nút
        _, col_center, _ = st.columns([2, 1, 2]) # Căn giữa (tỷ lệ 2:1:2)
        with col_center:
            # 4. Đổi tên nút và dùng type="primary" để CSS bắt được
            if st.button("Bắt đầu lại", type="primary", use_container_width=True):
                st.session_state.stage = '2_planning'
                st.session_state.df_final_output = None # Xóa kết quả cũ
                st.session_state.df_summary = None
                st.rerun()