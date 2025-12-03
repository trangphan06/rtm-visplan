import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
import math
import warnings

# === THƯ VIỆN GIAO DIỆN & BẢN ĐỒ ===
import folium
from streamlit_folium import st_folium

# === THƯ VIỆN LOGIC ===
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# Cấu hình trang & CSS
st.set_page_config(layout="wide", page_title="RTM Visit Planner Pro")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2.5rem !important; 
            padding-bottom: 2rem !important;
        }
        /* CSS cho Heatmap */
        [data-testid="stDataFrame"] td {
            text-align: center !important;
            font-size: 11px !important;
            padding: 0px !important;
            white-space: nowrap !important;
        }
        [data-testid="stDataFrame"] th {
            text-align: center !important;
            font-size: 11px !important;
            padding: 2px !important;
        }
        /* Ẩn bớt toolbars của bảng */
        [data-testid="stDataFrame"] th button { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE LOGIC (CACHED)
# ==========================================

@st.cache_data
def calculate_haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_dynamic_travel_time(dist_km, speed_slow, speed_fast):
    speed_kmh = speed_slow if dist_km < 2.0 else speed_fast
    return (dist_km / speed_kmh) * 60

def calculate_dynamic_quantum(df_route, target_points=1000):
    total_time = (df_route['Visit Time (min)'] * df_route['Total_Visits_Count']).sum()
    raw_quantum = total_time / target_points
    return max(raw_quantum, 0.5)

def explode_data_by_quantum(df_week, quantum):
    df_process = df_week.copy()
    weighted_time = df_process['Visit Time (min)'] * df_process['Weight_Factor']
    df_process['quantum_points'] = np.ceil(weighted_time / quantum).fillna(1).astype(int)
    df_exploded = df_process.loc[df_process.index.repeat(df_process['quantum_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    return df_exploded, df_process['quantum_points'].sum()

def solve_saturday_strategy(df_exploded, total_points):
    coords = df_exploded[['Latitude', 'Longitude']].values.astype(np.float32)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    n_chunks = 11 if total_points >= 50 else 6
    avg_chunk_size = total_points / n_chunks
    min_size = max(1, int(avg_chunk_size * 0.90))
    max_size = int(avg_chunk_size * 1.10)
    if max_size * n_chunks < total_points:
        max_size = int(total_points / n_chunks) + 2

    try:
        kmeans = KMeansConstrained(n_clusters=n_chunks, size_min=min_size, size_max=max_size, random_state=42, n_init=10)
        chunk_labels = kmeans.fit_predict(coords_scaled)
    except:
        from sklearn.cluster import KMeans
        kmeans_fallback = KMeans(n_clusters=n_chunks, random_state=42, n_init=10)
        chunk_labels = kmeans_fallback.fit_predict(coords_scaled)

    df_exploded['Chunk_ID'] = chunk_labels
    chunk_centers = df_exploded.groupby('Chunk_ID')[['Latitude', 'Longitude']].mean()
    dists = np.sqrt((chunk_centers['Latitude'] - df_exploded['Latitude'].mean())**2 + 
                    (chunk_centers['Longitude'] - df_exploded['Longitude'].mean())**2)
    saturday_chunk_id = dists.idxmax()
    
    df_exploded['Day'] = np.where(df_exploded['Chunk_ID'] == saturday_chunk_id, 'Sat', None)
    
    weekday_mask = df_exploded['Chunk_ID'] != saturday_chunk_id
    if weekday_mask.any():
        weekday_coords = coords_scaled[weekday_mask]
        n_days = 5
        try:
            w_total = len(weekday_coords)
            w_avg = w_total / 5
            w_min = int(w_avg * 0.90)
            w_max = int(w_avg * 1.10)
            kmeans_5 = KMeansConstrained(n_clusters=n_days, size_min=w_min, size_max=w_max, random_state=42, n_init=10)
            day_labels_idx = kmeans_5.fit_predict(weekday_coords)
            days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
            df_exploded.loc[weekday_mask, 'Day'] = [days_map[i] for i in day_labels_idx]
        except:
             df_exploded.loc[weekday_mask, 'Day'] = 'Mon'
    return df_exploded

def collapse_to_original(df_exploded, original_df):
    final_assignments = df_exploded.groupby('original_index')['Day'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Mon')
    df_result = original_df.copy()
    df_result['Assigned_Day'] = final_assignments
    df_result['Assigned_Day'].fillna('Mon', inplace=True)
    return df_result

def build_time_matrix_haversine(locations, speed_slow, speed_fast):
    size = len(locations)
    matrix = np.zeros((size, size), dtype=int)
    lats = np.array([loc[0] for loc in locations])
    lons = np.array([loc[1] for loc in locations])
    for i in range(size):
        for j in range(size):
            if i == j: continue
            dist_km = calculate_haversine_distance_km(lats[i], lons[i], lats[j], lons[j])
            speed = speed_slow if dist_km < 2.0 else speed_fast
            matrix[i][j] = int((dist_km / speed) * 3600)
    return matrix.tolist()

def solve_tsp_final(visits, depot_coords, speed_slow, speed_fast, mode='closed', end_coords=None):
    if not visits: return []
    locations = [depot_coords] + [v['coords'] for v in visits]
    has_end_point = (mode == 'open' and end_coords is not None)
    if has_end_point: locations.append(end_coords) 
        
    num_locations = len(locations)
    time_matrix = build_time_matrix_haversine(locations, speed_slow, speed_fast)
    
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [num_locations - 1] if has_end_point else [0])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 1 

    solution = routing.SolveWithParameters(search_parameters)
    ordered_visits = []
    
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                if has_end_point and node_index == num_locations - 1: pass 
                else: ordered_visits.append(visits[node_index - 1])
            index = solution.Value(routing.NextVar(index))
    return ordered_visits

def run_master_scheduler(df_cust, depot_coords, selected_route_ids, route_config_dict, visit_time_config, speed_config, progress_bar):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    df_cust = df_cust.copy()
    df_cust['Frequency'] = pd.to_numeric(df_cust['Frequency'], errors='coerce').fillna(1).round(0).astype(int)
    df_cust['Customer code'] = df_cust['Customer code'].astype(str).str.strip()
    df_cust_filtered = df_cust[df_cust['RouteID'].isin(selected_route_ids)].copy()
    
    cust_week_map = {} 
    f2_counter, f1_counter = 0, 0
    WEEKS = ['W1', 'W2', 'W3', 'W4']
    DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    SPACING_MAP_F8 = {'Mon': 'Thu', 'Tue': 'Fri', 'Wed': 'Sat', 'Thu': 'Mon', 'Fri': 'Tue', 'Sat': 'Wed'}
    SPACING_MAP_F12 = {'Mon': ['Mon', 'Wed', 'Fri'], 'Tue': ['Tue', 'Thu', 'Sat'], 'Wed': ['Mon', 'Wed', 'Fri'], 
                       'Thu': ['Tue', 'Thu', 'Sat'], 'Fri': ['Mon', 'Wed', 'Fri'], 'Sat': ['Tue', 'Thu', 'Sat']}
    
    for _, row in df_cust_filtered.iterrows():
        code, freq = row['Customer code'], row['Frequency']
        if freq == 2:
            cust_week_map[code] = ['W1', 'W3'] if f2_counter % 2 == 0 else ['W2', 'W4']
            f2_counter += 1
        elif freq == 1:
            cust_week_map[code] = [WEEKS[f1_counter % 4]]
            f1_counter += 1
            
    final_output_rows = []
    grouped = df_cust_filtered.groupby('RouteID')
    total_routes = len(selected_route_ids)
    
    for i, (route_id, route_df) in enumerate(grouped):
        progress_bar.progress(i / total_routes, text=f"Đang xử lý Route {route_id} - ({i+1}/{total_routes})")
        route_df['Visit Time (min)'] = route_df['Segment'].map(visit_time_config).fillna(visit_time_config.get('default', 10.0))
        route_df['Weight_Factor'] = 1.0
        
        for week in WEEKS:
            week_visits_all = [] 
            for _, row in route_df.iterrows():
                freq, code = row['Frequency'], row['Customer code']
                is_in_week = False
                num_visits = 0
                if freq >= 4: num_visits, is_in_week = int(freq // 4), True
                else:
                    if week in cust_week_map.get(code, []): is_in_week, num_visits = True, 1
                
                if is_in_week:
                    for v_i in range(int(num_visits)):
                        r = row.copy() 
                        r['Visit_ID_Internal'] = f"{code}_{week}_{v_i}" 
                        r['Visit_Order'] = v_i
                        r['Total_Visits_Count'] = num_visits
                        week_visits_all.append(r)
            
            if not week_visits_all: continue
            
            best_df, best_score = None, float('inf')
            for iteration in range(3):
                full_df = pd.DataFrame(week_visits_all)
                df_core = full_df[full_df['Visit_Order'] == 0].copy()
                quantum = calculate_dynamic_quantum(df_core, 1200)
                df_exploded, total_pts = explode_data_by_quantum(df_core, quantum)
                df_labeled = solve_saturday_strategy(df_exploded, total_pts)
                df_core_res = collapse_to_original(df_labeled, df_core)
                
                anchor_map = df_core_res.set_index('Customer code')['Assigned_Day'].to_dict()
                df_dependent = full_df[full_df['Visit_Order'] > 0].copy()
                if not df_dependent.empty:
                    dep_days = []
                    for _, r_d in df_dependent.iterrows():
                        anchor = anchor_map.get(r_d['Customer code'], 'Mon')
                        day = anchor
                        if r_d['Total_Visits_Count'] == 2 and r_d['Visit_Order'] == 1: day = SPACING_MAP_F8.get(anchor, 'Thu')
                        elif r_d['Total_Visits_Count'] == 3 and r_d['Visit_Order'] < 3: 
                            day = SPACING_MAP_F12.get(anchor, ['Mon', 'Wed', 'Fri'])[r_d['Visit_Order']]
                        dep_days.append(day)
                    df_dependent['Assigned_Day'] = dep_days
                
                df_combined = pd.concat([df_core_res, df_dependent])
                
                day_stats, total_work = {}, 0
                for day in DAY_ORDER:
                    d_visits = df_combined[df_combined['Assigned_Day'] == day]
                    if d_visits.empty: day_stats[day] = 0; continue
                    work = d_visits['Visit Time (min)'].sum()
                    day_stats[day] = work
                    total_work += work
                
                unit_work = total_work / 11
                max_dev = 0
                weights = {}
                for day, act in day_stats.items():
                    tgt = unit_work * (1 if day == 'Sat' else 2)
                    if tgt == 0: continue
                    ratio = act / tgt
                    max_dev = max(max_dev, abs(1 - ratio))
                    weights[day] = max(0.5, min(1 + (ratio - 1) * 0.7, 2.0))
                
                if max_dev < best_score: best_score, best_df = max_dev, df_combined.copy()
                if max_dev <= 1.10: break 
                
                for item in week_visits_all:
                    if item['Visit_Order'] == 0:
                        try:
                            day = df_core_res[df_core_res['Visit_ID_Internal'] == item['Visit_ID_Internal']]['Assigned_Day'].iloc[0]
                            item['Weight_Factor'] *= weights.get(day, 1.0)
                        except: pass

            for day in DAY_ORDER:
                d_visits = best_df[best_df['Assigned_Day'] == day]
                if d_visits.empty: continue
                tsp_in = []
                for _, row in d_visits.iterrows():
                    d = row.to_dict()
                    d['coords'] = (row['Latitude'], row['Longitude'])
                    tsp_in.append(d)
                
                end_cfg = route_config_dict.get(route_id)
                mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
                ordered = solve_tsp_final(tsp_in, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
                
                prev, seq, agg_time, agg_dist = depot_coords, 1, 0, 0
                for item in ordered:
                    curr = item['coords']
                    dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
                    travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
                    agg_time += travel + item['Visit Time (min)']
                    agg_dist += dist
                    
                    res = item.copy()
                    res.update({'RouteID': route_id, 'Week': week, 'Day': day, 'Week&Day': f"{week}-{day}",
                                'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                                'Distance (km)': round(dist, 2), 'Total Workload (min)': round(agg_time, 2)})
                    
                    for k in ['coords', 'angle', 'Weight_Factor', 'quantum_points']: 
                        if k in res: del res[k]
                    final_output_rows.append(res)
                    prev, seq = curr, seq+1

    if not final_output_rows: return pd.DataFrame()
    df_final = pd.DataFrame(final_output_rows)
    df_final['Day'] = pd.Categorical(df_final['Day'], categories=DAY_ORDER, ordered=True)
    return df_final.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])

def recalculate_routes(df_edited, depot_coords, route_config, speed_config, impacted_groups=None):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    new_rows = []
    
    for (r_id, week, day), group in df_edited.groupby(['RouteID', 'Week', 'Day']):
        should_optimize = True
        if impacted_groups is not None:
            should_optimize = (r_id, week, day) in impacted_groups
            
        if should_optimize:
            tsp_input = []
            for _, row in group.iterrows():
                d = row.to_dict()
                d['coords'] = (row['Latitude'], row['Longitude'])
                tsp_input.append(d)
            end_cfg = route_config.get(r_id)
            mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
            ordered = solve_tsp_final(tsp_input, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
        else:
            ordered = [row.to_dict() for _, row in group.sort_values('Sequence').iterrows()]
            for item in ordered: item['coords'] = (item['Latitude'], item['Longitude'])

        prev, seq, agg_time, agg_dist = depot_coords, 1, 0, 0
        for item in ordered:
            curr = item['coords']
            dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
            travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
            agg_time += travel + item['Visit Time (min)']
            agg_dist += dist
            
            res = item.copy()
            res.update({
                'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                'Distance (km)': round(dist, 2), 'Total Workload (min)': round(agg_time, 2)
            })
            if 'coords' in res: del res['coords']
            new_rows.append(res)
            prev, seq = curr, seq+1
    return pd.DataFrame(new_rows)

def get_changed_visits(df_orig, df_curr):
    if df_orig is None or df_curr is None: return []
    # [FIXED] Compare strictly on Visit_ID level (Granular)
    df1 = df_orig.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    df2 = df_curr.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    common = df1.index.intersection(df2.index)
    diff = (df1.loc[common] != df2.loc[common]).any(axis=1)
    # Return specific Visit IDs that changed
    return diff[diff].index.tolist()

# ==========================================
# 2. UI HELPER & STATE
# ==========================================

REQUIRED_COLS_CUST = {
    'RouteID': 'RouteID', 'Customer code': 'Customer code', 
    'Customer Name': 'Customer Name', 'Latitude': 'Latitude', 
    'Longitude': 'Longitude', 'Frequency': 'Frequency', 'Segment': 'Segment'
}
REQUIRED_COLS_DIST = {
    'Distributor Code': 'Distributor Code', 'Distributor Name': 'Distributor Name', 
    'Latitude': 'Latitude', 'Longitude': 'Longitude'
}

@st.cache_data
def create_template_excel(cols_dict, is_dist=False):
    if is_dist:
        data = {
            'Distributor Code': ['12345678', '23456789'],
            'Distributor Name': ['Công ty ABC', 'NPP Thành Phát'],
            'Latitude': [10.7769, 21.0285],
            'Longitude': [106.7009, 105.8542]
        }
    else:
        data = {
            'RouteID': ['VN123456', 'VN234567'],
            'Customer code': ['12345678', '23456789'],
            'Customer Name': ['Tạp hóa Cô Hai', 'Quán nhậu Cây đa'],
            'Latitude': [10.77, 10.78],
            'Longitude': [106.70, 106.71],
            'Frequency': [4, 2],
            'Segment': ['Gold', 'Silver']
        }
    
    df_sample = pd.DataFrame(data)
    for col in cols_dict.keys():
        if col not in df_sample.columns:
            df_sample[col] = ''
            
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_sample.to_excel(writer, sheet_name='Template', index=False)
    return output.getvalue()

@st.cache_data
def to_excel_output(df_master):
    output = BytesIO()
    df_export = df_master.drop(columns=['Visit_ID_Internal'], errors='ignore').copy()
    
    df_export = df_export.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])
    
    df_export['Agg_Dist'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Distance (km)'].cumsum()
    df_export['Agg_Travel'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Travel Time (min)'].cumsum()
    
    rename_map = {
        'Day': 'Ngày',
        'Week': 'Tuần',
        'Week&Day': 'Ngày & Tuần',
        'Sequence': 'Thứ tự',
        'Distance (km)': 'Khoảng cách từ KH trước',
        'Travel Time (min)': 'Thời gian di chuyển từ KH trước',
        'Agg_Dist': 'Khoảng cách từ đầu ngày',
        'Agg_Travel': 'Thời gian di chuyển từ đầu ngày',
        'Visit Time (min)': 'Thời gian viếng thăm điểm bán',
        'Total Workload (min)': 'Tổng thời gian làm việc từ đầu ngày'
    }
    df_export_final = df_export.rename(columns=rename_map)
    
    df_sum = df_master.groupby(['RouteID', 'Week', 'Day']).agg(
        Total_TIO_min=('Visit Time (min)', 'sum'),
        Total_TBO_min=('Travel Time (min)', 'sum'),
        Num_Customers=('Customer code', 'count')
    ).reset_index()
    
    df_sum['Total_Workload_min'] = df_sum['Total_TIO_min'] + df_sum['Total_TBO_min']
    
    df_sum['Total_TIO_h'] = (df_sum['Total_TIO_min'] / 60).round(2)
    df_sum['Total_TBO_h'] = (df_sum['Total_TBO_min'] / 60).round(2)
    df_sum['Total_Workload_h'] = (df_sum['Total_Workload_min'] / 60).round(2)
    
    sum_rename = {
        'Week': 'Tuần',
        'Day': 'Ngày',
        'Total_TIO_h': 'Tổng thời gian viếng thăm (Giờ)',
        'Total_TBO_h': 'Tổng thời gian di chuyển (Giờ)',
        'Num_Customers': 'Số KH',
        'Total_Workload_h': 'Tổng thời gian làm việc (Giờ)'
    }
    df_sum_final = df_sum.rename(columns=sum_rename)
    cols_sum = ['RouteID', 'Tuần', 'Ngày', 'Tổng thời gian viếng thăm (Giờ)', 
                'Tổng thời gian di chuyển (Giờ)', 'Số KH', 'Tổng thời gian làm việc (Giờ)']
    df_sum_final = df_sum_final[cols_sum]
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export_final.to_excel(writer, sheet_name='Lịch viếng thăm', index=False)
        df_sum_final.to_excel(writer, sheet_name='Tổng quan', index=False)
    return output.getvalue()

@st.cache_data
def create_folium_map(df_filtered_dict, col_mapping):
    df_filtered = pd.DataFrame.from_dict(df_filtered_dict)
    if df_filtered.empty: return None
    
    center = [df_filtered['Latitude'].mean(), df_filtered['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    
    legend_html = '''
     <div style="position: fixed; bottom: 30px; left: 30px; width: 80px; height: 130px; 
     border:2px solid grey; z-index:9999; font-size:12px; background-color:white; padding: 10px; opacity: 0.9;">
     <b>Chú giải:</b><br>
     <i style="background:red; width:10px; height:10px; display:inline-block;"></i> T2<br>
     <i style="background:green; width:10px; height:10px; display:inline-block;"></i> T3<br>
     <i style="background:blue; width:10px; height:10px; display:inline-block;"></i> T4<br>
     <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> T5<br>
     <i style="background:purple; width:10px; height:10px; display:inline-block;"></i> T6<br>
     <i style="background:black; width:10px; height:10px; display:inline-block;"></i> T7<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    color_map = {'T2': 'red', 'T3': 'green', 'T4': 'blue', 'T5': 'orange', 'T6': 'purple', 'T7': 'black'}
    
    for (r, w, d), group in df_filtered.groupby(['RouteID', 'Week', 'Day']):
        color = color_map.get(d, 'gray')
        grp = group.sort_values('Sequence')
        
        folium.PolyLine(grp[['Latitude', 'Longitude']].values.tolist(), color=color, weight=3, opacity=0.7).add_to(m)
        
        for _, row in grp.iterrows():
            tooltip_parts = []
            excluded_cols = ['Latitude', 'Longitude', 'Total Workload (min)', 'Visit_ID_Internal', 
                             'quantum_points', 'Weight_Factor']
            for std_key, orig_label in col_mapping.items():
                if std_key not in excluded_cols and std_key in row and pd.notna(row[std_key]):
                    tooltip_parts.append(f"<b>{orig_label}:</b> {row[std_key]}")
            
            tooltip_parts.append(f"<b>Thứ tự:</b> {row['Sequence']}")
            popup_txt = "<br>".join(tooltip_parts)
            icon_html = f"""<div style="background:{color};color:white;border-radius:50%;width:20px;height:20px;text-align:center;font-size:12px;font-weight:bold;line-height:20px;border:1px solid white;">{row['Sequence']}</div>"""
            folium.Marker(
                location=(row['Latitude'], row['Longitude']),
                icon=folium.DivIcon(html=icon_html),
                tooltip=popup_txt 
            ).add_to(m)
    return m

@st.cache_data
def create_heatmap(df_dict, value_col, agg_mode, fmt="{:.1f}", title="Heatmap"):
    df_data = pd.DataFrame.from_dict(df_dict)
    weeks = ['W1', 'W2', 'W3', 'W4']
    days = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    
    if agg_mode == 'count':
        pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='count')
    elif agg_mode == 'sum_time':
        pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.sum()/60)
    elif agg_mode == 'mean_time':
        pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.mean()/60)
    elif agg_mode == 'mean_qty':
        pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='mean')
        
    pivot = pivot.reindex(index=weeks, columns=days).fillna(0)
    pivot.index.name = None
    st.markdown(f"**{title}**")
    st.dataframe(
        pivot.style.format(fmt).background_gradient(cmap='RdYlGn_r', axis=None), 
        height=140, use_container_width=True,
        column_config={col: st.column_config.Column(width="small") for col in days}
    )

def find_col_index(df_cols, target_name):
    for i, col in enumerate(df_cols):
        if str(col).strip().lower() == target_name.lower():
            return i
    return 0

# Initialize Session State
if 'stage' not in st.session_state: st.session_state.stage = '1_upload'
if 'df_final' not in st.session_state: st.session_state.df_final = None 
if 'df_editing' not in st.session_state: st.session_state.df_editing = None 
if 'map_clicked_code' not in st.session_state: st.session_state.map_clicked_code = None
if 'editor_filter_mode' not in st.session_state: st.session_state.editor_filter_mode = 'all' 
if 'map_version' not in st.session_state: st.session_state.map_version = 0
if 'col_map_main' not in st.session_state: st.session_state.col_map_main = {}
if 'has_changes' not in st.session_state: st.session_state.has_changes = False
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False

# ==========================================
# 3. MAIN APP FLOW
# ==========================================

st.title("Công cụ xếp lịch viếng thăm - RTM Visit Planner")

# --- SCREEN 1: DATA INPUT ---
if st.session_state.stage == '1_upload':
    st.markdown("### Bước 1: Tải lên dữ liệu")
    c1, c2 = st.columns(2)
    c1.download_button("📥 Tải Template Customers", create_template_excel(REQUIRED_COLS_CUST, False), "Customers_Template.xlsx")
    c2.download_button("📥 Tải Template Distributors", create_template_excel(REQUIRED_COLS_DIST, True), "Distributors_Template.xlsx")
    
    u1, u2 = st.columns(2)
    up_cust = u1.file_uploader("Upload File Customers", type=['xlsx'])
    up_dist = u2.file_uploader("Upload File Distributors", type=['xlsx'])
    
    if up_cust and up_dist:
        df_c, df_d = pd.read_excel(up_cust), pd.read_excel(up_dist)
        st.markdown("---")
        with st.form("mapping"):
            c1, c2 = st.columns(2)
            map_c = {k: c1.selectbox(f"File Customers: {k}", df_c.columns, index=find_col_index(df_c.columns, k)) for k in REQUIRED_COLS_CUST}
            map_d = {k: c2.selectbox(f"File Distributors: {k}", df_d.columns, index=find_col_index(df_d.columns, k)) for k in REQUIRED_COLS_DIST}
            
            if st.form_submit_button("Tiếp tục >>"):
                st.session_state.col_map_main = map_c 
                
                df_c = df_c.rename(columns={v: k for k, v in map_c.items()})
                df_d = df_d.rename(columns={v: k for k, v in map_d.items()})
                
                if 'Customer code' in df_c.columns:
                    df_c['Customer code'] = df_c['Customer code'].astype(str).str.strip()
                if 'RouteID' in df_c.columns:
                    df_c['RouteID'] = df_c['RouteID'].astype(str).str.strip()
                if 'Distributor Code' in df_d.columns:
                    df_d['Distributor Code'] = df_d['Distributor Code'].astype(str).str.strip()
                
                if df_c['Customer code'].duplicated().any(): st.warning("Loại bỏ KH trùng lặp.")
                df_c = df_c.drop_duplicates('Customer code').dropna(subset=['Latitude', 'Longitude'])
                st.session_state.df_cust = df_c
                st.session_state.df_dist = df_d
                st.session_state.stage = '2_planning'
                st.rerun()

# --- SCREEN 2: CONFIGURATION ---
elif st.session_state.stage == '2_planning':
    st.markdown("### Bước 2: Điều chỉnh")
    
    unique_dist = st.session_state.df_dist.drop_duplicates(subset=['Distributor Code'])
    dist_opts = unique_dist.apply(lambda x: f"{x['Distributor Code']} - {x['Distributor Name']}", axis=1)
    sel_dist = st.selectbox("Chọn Nhà Phân Phối:", dist_opts)
    
    sel_code = sel_dist.split(' - ')[0]
    depot_row = st.session_state.df_dist[st.session_state.df_dist['Distributor Code'] == sel_code].iloc[0]
    st.session_state.depot_coords = (depot_row['Latitude'], depot_row['Longitude'])
    
    all_routes = sorted(st.session_state.df_cust['RouteID'].unique().astype(str))
    sel_routes = st.multiselect("Chọn RouteID thuộc NPP:", all_routes, default=all_routes[:1])
    
    route_end_point_configs = {}
    if sel_routes:
        st.markdown("**Chọn Điểm Kết Thúc ngày làm việc:**")
        for r_id in sel_routes:
            c1, c2, c3 = st.columns([1, 2, 3])
            c1.write(f"🏷️ **{r_id}**")
            mode = c2.selectbox(f"Chế độ {r_id}", ["Quay về NPP", "Kết thúc tại 1 KH"], label_visibility="collapsed")
            if "Kết thúc" in mode:
                custs = st.session_state.df_cust[st.session_state.df_cust['RouteID'] == r_id]
                opts = custs.apply(lambda x: f"{x['Customer code']} - {x.get('Customer Name','')}", axis=1)
                sel_c = c3.selectbox(f"Chọn KH {r_id}", opts, label_visibility="collapsed")
                if sel_c:
                    c_row = custs[custs['Customer code'] == sel_c.split(' - ')[0]].iloc[0]
                    route_end_point_configs[r_id] = (c_row['Latitude'], c_row['Longitude'])
            else:
                route_end_point_configs[r_id] = None

    with st.expander("⚙️ Tùy chỉnh Tốc độ & Thời gian (Nhấn để mở)", expanded=False):
        c1, c2 = st.columns(2)
        s_slow = c1.number_input("KH cách nhau < 2km (đơn vị: km/h)", min_value=10, max_value=60, value=20, step=5)
        s_fast = c2.number_input("KH cách nhau > 2km (đơn vị: km/h)", min_value=30, max_value=100, value=40, step=5)
        st.write("Thời gian viếng thăm (phút):")
        cols = st.columns(6)
        vt_cfg = {}
        for i, (k, v) in enumerate({'MT':19.5, 'Cooler':18.0, 'Gold':9.0, 'Silver':7.8, 'Bronze':6.8, 'trống/mặc định':10.0}.items()):
            vt_cfg[k] = cols[i].number_input(k, 0.0, 60.0, v, step=1.0)
    
    c_back, c_run = st.columns([1, 5])
    if c_back.button("<< Quay lại"):
        st.session_state.stage = '1_upload'
        st.rerun()
    
    if c_run.button("🚀 Chạy xếp lịch viếng thăm", type="primary", disabled=not sel_routes):
        pb = st.progress(0, "Đang xử lý...")
        try:
            st.session_state.route_cfg = route_end_point_configs
            st.session_state.speed_cfg = {'slow': s_slow, 'fast': s_fast}
            
            df_res = run_master_scheduler(
                st.session_state.df_cust, st.session_state.depot_coords, sel_routes,
                route_end_point_configs, vt_cfg, st.session_state.speed_cfg, pb
            )
            day_map = {'Mon': 'T2', 'Tue': 'T3', 'Wed': 'T4', 'Thu': 'T5', 'Fri': 'T6', 'Sat': 'T7'}
            df_res['Day'] = df_res['Day'].map(day_map)
            
            st.session_state.df_final = df_res.copy()
            st.session_state.df_editing = df_res.copy() 
            st.session_state.stage = '3_results'
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: {e}")
            import traceback
            st.text(traceback.format_exc())

# --- SCREEN 3: DASHBOARD & EDITOR ---
elif st.session_state.stage == '3_results':
    st.markdown("### Kết quả")
    
    all_r = ['All Routes'] + sorted(st.session_state.df_editing['RouteID'].unique().tolist())
    c_sel, _ = st.columns([2, 8])
    with c_sel: sel_r_view = st.selectbox("Xem Route:", all_r)

    df_view = st.session_state.df_editing.copy()
    df_view['Workload_Single_Min'] = df_view['Visit Time (min)'] + df_view['Travel Time (min)']
    
    heatmap_data = {}
    agg_mode_cust, agg_mode_time = 'count', 'sum_time'
    col_cust, col_work, col_visit, col_travel = 'Customer code', 'Workload_Single_Min', 'Visit Time (min)', 'Travel Time (min)'
    
    if sel_r_view == 'All Routes':
        df_route_daily = df_view.groupby(['RouteID', 'Week', 'Day']).agg(
            num_cust=('Customer code', 'count'),
            total_work=('Workload_Single_Min', 'sum'),
            total_visit=('Visit Time (min)', 'sum'),
            total_travel=('Travel Time (min)', 'sum')
        ).reset_index()
        heatmap_data = df_route_daily.to_dict('list')
        agg_mode_cust, agg_mode_time = 'mean_qty', 'mean_time'
        col_cust, col_work, col_visit, col_travel = 'num_cust', 'total_work', 'total_visit', 'total_travel'
    else:
        df_view = df_view[df_view['RouteID'] == sel_r_view]
        heatmap_data = df_view.to_dict('list')
        
    df_editor_view = df_view.copy()
    
    r1c1, r1c2 = st.columns(2)
    with r1c1: create_heatmap(heatmap_data, col_cust, agg_mode_cust, "{:.0f}", "Số lượng KH/ngày (TB)" if sel_r_view == 'All Routes' else "Số lượng KH/ngày")
    with r1c2: create_heatmap(heatmap_data, col_work, agg_mode_time, "{:.1f}", "Tổng giờ làm việc/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ làm việc/ngày")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1: create_heatmap(heatmap_data, col_visit, agg_mode_time, "{:.1f}", "Tổng giờ viếng thăm/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ viếng thăm/ngày")
    with r2c2: create_heatmap(heatmap_data, col_travel, agg_mode_time, "{:.1f}", "Tổng giờ di chuyển/ngày (TB)" if sel_r_view == 'All Routes' else "Tổng giờ di chuyển/ngày")
    
    st.markdown("---")

    col_map, col_edit = st.columns([7, 3])
    
    with col_map:
        mf1, mf2 = st.columns([1, 2])
        all_weeks = sorted(df_view['Week'].unique())
        def_week = ['W1'] if 'W1' in all_weeks else all_weeks
        weeks = mf1.multiselect("Lọc Tuần:", all_weeks, default=def_week)
        days = mf2.multiselect("Lọc Ngày:", ['T2','T3','T4','T5','T6','T7'], default=['T2','T3','T4','T5','T6','T7'])
        
        df_map = df_view[(df_view['Week'].isin(weeks)) & (df_view['Day'].isin(days))]
        st.caption("💡 Click vào điểm trên bản đồ để sửa nhanh bên phải.")
        
        map_data = st_folium(
            create_folium_map(df_map.to_dict('list'), st.session_state.col_map_main), 
            height=550, use_container_width=True,
            key=f"folium_map_{st.session_state.map_version}",
            returned_objects=["last_object_clicked"]
        )
        if map_data and map_data.get("last_object_clicked"):
            lat, lng = map_data["last_object_clicked"]['lat'], map_data["last_object_clicked"]['lng']
            mask = (np.isclose(st.session_state.df_editing['Latitude'], lat, atol=1e-5)) & \
                   (np.isclose(st.session_state.df_editing['Longitude'], lng, atol=1e-5))
            found = st.session_state.df_editing[mask]
            if not found.empty:
                clicked_code = found.iloc[0]['Customer code']
                if st.session_state.map_clicked_code != clicked_code:
                    st.session_state.map_clicked_code = clicked_code
                    st.session_state.editor_filter_mode = 'single'
                    st.rerun()

    with col_edit:
        st.subheader("🛠️ Chỉnh sửa Thủ công")
        
        # [CRITICAL FIX] Use changed_visit_IDs logic
        changed_ids = get_changed_visits(st.session_state.df_final, st.session_state.df_editing)
        
        # Flag specific rows using Visit_ID
        df_editor_view['Đã sửa'] = df_editor_view['Visit_ID_Internal'].apply(lambda x: "✏️" if x in changed_ids else "")

        if st.session_state.editor_filter_mode == 'single' and st.session_state.map_clicked_code:
            df_editor_view = df_editor_view[df_editor_view['Customer code'] == st.session_state.map_clicked_code]
            st.info(f"Đang sửa KH: {st.session_state.map_clicked_code}")
        elif st.session_state.editor_filter_mode == 'changed':
             df_editor_view = df_editor_view[df_editor_view['Visit_ID_Internal'].isin(changed_ids)]
             if df_editor_view.empty: st.info("Chưa có KH nào bị thay đổi.")

        edited_df = st.data_editor(
            df_editor_view,
            column_config={
                "Customer code": st.column_config.TextColumn("Mã KH", disabled=True),
                "Customer Name": st.column_config.TextColumn("Tên KH", disabled=True),
                "Week": st.column_config.SelectboxColumn("Tuần", options=['W1','W2','W3','W4'], required=True),
                "Day": st.column_config.SelectboxColumn("Ngày", options=['T2','T3','T4','T5','T6','T7'], required=True),
                "Sequence": st.column_config.NumberColumn("Thứ tự", disabled=True),
                "Đã sửa": st.column_config.TextColumn("Đã sửa", disabled=True, width="small"),
            },
            column_order=['Customer code', 'Customer Name', 'Week', 'Day', 'Sequence', 'Đã sửa'],
            hide_index=True,
            use_container_width=True,
            height=400,
            key="data_editor_widget"
        )
        
        c_up, c_rst = st.columns(2)
        if c_up.button("💾 Cập nhật", type="primary", use_container_width=True):
            with st.spinner("Đang tính toán lại lộ trình..."):
                impacted_groups = set()
                
                for idx, row in edited_df.iterrows():
                    if idx in df_editor_view.index:
                        visit_id = df_editor_view.loc[idx, 'Visit_ID_Internal']
                        mask = st.session_state.df_editing['Visit_ID_Internal'] == visit_id
                        
                        if mask.any():
                            current_row = st.session_state.df_editing.loc[mask].iloc[0]
                            old_r, old_w, old_d = current_row['RouteID'], current_row['Week'], current_row['Day']
                            new_w, new_d = row['Week'], row['Day']
                            
                            if (old_w != new_w) or (old_d != new_d):
                                impacted_groups.add((old_r, old_w, old_d))
                                impacted_groups.add((old_r, new_w, new_d))
                                st.session_state.df_editing.loc[mask, ['Week', 'Day']] = [new_w, new_d]
                
                if impacted_groups:
                    st.session_state.df_editing = recalculate_routes(
                        st.session_state.df_editing, 
                        st.session_state.depot_coords, 
                        st.session_state.route_cfg, 
                        st.session_state.speed_cfg,
                        impacted_groups=impacted_groups
                    )
                    st.session_state.map_version += 1
                    st.session_state.has_changes = True 
                    st.success("Đã cập nhật!")
                    time.sleep(0.5) 
                    st.rerun()
                else:
                    st.info("Không có thay đổi về Ngày/Tuần để cập nhật.")

        if not st.session_state.confirm_reset:
            if c_rst.button("🔄 Hủy bỏ chỉnh sửa", use_container_width=True, disabled=not st.session_state.has_changes):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("Hủy bỏ toàn bộ chỉnh sửa thủ công & Quay về phiên bản ban đầu?")
            c_yes, c_no = st.columns(2)
            if c_yes.button("✅ Đồng ý", use_container_width=True):
                st.session_state.df_editing = st.session_state.df_final.copy()
                st.session_state.editor_filter_mode = 'all'
                st.session_state.map_version += 1
                st.session_state.has_changes = False
                st.session_state.confirm_reset = False
                st.rerun()
            if c_no.button("❌ Không", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
            
        f1, f2 = st.columns(2)
        if f1.button("🌪️ Lọc KH đã sửa", use_container_width=True, disabled=not st.session_state.has_changes):
            st.session_state.editor_filter_mode = 'changed'
            st.rerun()
            
        is_filtering = st.session_state.editor_filter_mode != 'all'
        if f2.button("✖ Bỏ lọc", use_container_width=True, disabled=not is_filtering):
            st.session_state.editor_filter_mode = 'all'
            st.session_state.map_clicked_code = None
            st.rerun()

    st.markdown("---")
    if st.button("📥 Tải File Excel Kết Quả Cuối Cùng", type="primary"):
        excel_data = to_excel_output(st.session_state.df_editing)
        st.download_button("Download .xlsx", excel_data, "Final_RTM_Schedule.xlsx")

    if st.button("<< Quay lại từ đầu"):
        st.session_state.stage = '1_upload'
        st.rerun()