import pandas as pd, numpy as np

def add_features(X):
    d = X.copy()
    eps = 1e-6
    W = (d.X_Maximum - d.X_Minimum).clip(lower=1)     # 결함 가로 길이(px)
    H = (d.Y_Maximum - d.Y_Minimum).clip(lower=1)     # 결함 세로 길이(px)
    bbox = W*H
    L = d.Length_of_Conveyer

    # --- 1. 크기/모양 ---
    d['f_width']        = W
    d['f_height']       = H
    d['f_aspect']       = np.log1p(W) - np.log1p(H)            # 가로로 긴가 세로로 긴가
    d['f_bbox_area']    = np.log1p(bbox)
    d['f_fill_rate']    = d.Pixels_Areas/(bbox+eps)            # 사각형을 얼마나 꽉 채우나
    d['f_compact']      = d.Pixels_Areas/((d.X_Perimeter+d.Y_Perimeter)**2+eps)  # 원형에 가까운 정도
    d['f_perim_ratio']  = np.log1p(d.X_Perimeter) - np.log1p(d.Y_Perimeter)
    d['f_rough_x']      = d.X_Perimeter/(W+eps)                # 가로 테두리 울퉁불퉁함
    d['f_rough_y']      = d.Y_Perimeter/(H+eps)
    d['f_perim_area']   = (d.X_Perimeter+d.Y_Perimeter)/np.sqrt(d.Pixels_Areas+eps)
    d['f_elong']        = np.maximum(W,H)/np.minimum(W,H)

    # --- 2. 밝기 ---
    d['f_lum_mean']     = d.Sum_of_Luminosity/(d.Pixels_Areas+eps)   # 결함 평균 밝기
    d['f_lum_range']    = d.Maximum_of_Luminosity - d.Minimum_of_Luminosity
    d['f_lum_contrast'] = d.f_lum_range/(d.Maximum_of_Luminosity+d.Minimum_of_Luminosity+eps)
    d['f_lum_pos']      = (d.f_lum_mean - d.Minimum_of_Luminosity)/(d.f_lum_range+eps)
    d['f_lum_density']  = d.Sum_of_Luminosity/(bbox+eps)
    d['f_lum_per_perim']= d.Sum_of_Luminosity/((d.X_Perimeter+d.Y_Perimeter)+eps)

    # --- 3. 강판 위 위치 (가장자리 결함 vs 중앙 결함) ---
    cx = (d.X_Minimum + d.X_Maximum)/2
    d['f_xpos_rel']     = cx/(L+eps)
    d['f_edge_dist']    = (0.5 - (cx/(L+eps) - 0.5).abs()).clip(lower=0)  # 판 가장자리까지 상대거리
    d['f_margin_left']  = d.X_Minimum/(L+eps)
    d['f_margin_right'] = (L - d.X_Maximum)/(L+eps)
    d['f_touch_edge']   = ((d.X_Minimum<=5)|((L-d.X_Maximum)<=5)).astype(int)
    d['f_width_rel']    = W/(L+eps)

    # --- 4. 공정 조건 상호작용 ---
    d['f_thick_x_type'] = d.Steel_Plate_Thickness*d.TypeOfSteel_A300
    d['f_area_per_thick']= np.log1p(d.Pixels_Areas)/np.log1p(d.Steel_Plate_Thickness)
    return d
