import pandas as pd

def detect_ball_bounces(ball_detections, ball_shot_frames, court_keypoints):    
    if not ball_shot_frames or len(ball_shot_frames) < 2:
        print("   ⚠️ Need at least 2 shots\n")
        return []
    
    # X bounds: use top singles lines (keypoints 4 and 6)
    singles_left = court_keypoints[4 * 2]   
    singles_right = court_keypoints[6 * 2]  
    
    # Y bounds: use the full court length (same for singles and doubles)
    court_top = min(court_keypoints[0 * 2 + 1], court_keypoints[1 * 2 + 1])   
    court_bottom = max(court_keypoints[2 * 2 + 1], court_keypoints[3 * 2 + 1])
    
    court_height = court_bottom - court_top
    court_center_y = (court_top + court_bottom) / 2
    
    print(f"   Singles court bounds:")
    print(f"   X: {singles_left:.0f} to {singles_right:.0f}")
    print(f"   Y: {court_top:.0f} to {court_bottom:.0f}\n")
    
    # Build position lookup
    ball_pos = {}
    for f in range(len(ball_detections)):
        if 1 in ball_detections[f]:
            box = ball_detections[f][1]
            ball_pos[f] = {
                'x': (box[0] + box[2]) / 2,
                'y': (box[1] + box[3]) / 2
            }
    
    bounce_positions = []
    
    # Find bounce in each shot interval
    for shot_idx in range(len(ball_shot_frames) - 1):
        shot_frame = ball_shot_frames[shot_idx]
        next_shot = ball_shot_frames[shot_idx + 1]
        
        if shot_frame not in ball_pos:
            continue
        
        player_in_top = ball_pos[shot_frame]['y'] < court_center_y
        
        print(f"  Bounce {shot_idx + 1}: Between shots at frames {shot_frame} and {next_shot}")
        print(f"      Player in {'TOP' if player_in_top else 'BOTTOM'} half")
        
        # Search for bounce
        search_start = shot_frame + 10
        search_end = next_shot - 10
        
        trajectory = []
        for f in range(search_start, search_end + 1):
            if f in ball_pos:
                trajectory.append({
                    'frame': f,
                    'x': ball_pos[f]['x'],
                    'y': ball_pos[f]['y']
                })
        
        if len(trajectory) < 5:
            print(f"      ⚠️ Not enough points\n")
            continue
        
        # Find where ball is closest to ground
        if player_in_top:
            bounce = max(trajectory, key=lambda p: p['y'])
        else:
            bounce = min(trajectory, key=lambda p: p['y'])
                
        # Horizontal: must be within singles lines
        x_in_bounds = singles_left <= bounce['x'] <= singles_right
        
        # Vertical: with margin for behind baseline
        margin = court_height * 0.10
        
        if player_in_top:
            y_min = court_center_y
            y_max = court_bottom + margin
        else:
            y_min = court_top - margin
            y_max = court_center_y
        
        y_in_bounds = y_min <= bounce['y'] <= y_max
        
        is_in_bounds = x_in_bounds and y_in_bounds
        
        # Debug
        if not x_in_bounds:
            side = "LEFT" if bounce['x'] < singles_left else "RIGHT"
            print(f"      ⚠️ OUT: Ball at X={bounce['x']:.0f} is {side} of singles line")
        
        if not y_in_bounds:
            print(f"      ⚠️ OUT: Ball at Y={bounce['y']:.0f} outside range [{y_min:.0f}, {y_max:.0f}]")
        
        height_ratio = (bounce['y'] - court_top) / court_height
        status = "IN ✅" if is_in_bounds else "OUT ❌"
        
        print(f"      Found at: Frame {bounce['frame']}, "
              f"Position ({bounce['x']:.0f}, {bounce['y']:.0f}) - {status}\n")
        
        bounce_positions.append({
            'frame': bounce['frame'],
            'x': bounce['x'],
            'y': bounce['y'],
            'shot_idx': shot_idx,
            'height_ratio': height_ratio,
            'player_side': 'top' if player_in_top else 'bottom',
            'is_in_bounds': is_in_bounds
        })
    
    in_count = sum(1 for b in bounce_positions if b['is_in_bounds'])
    out_count = len(bounce_positions) - in_count
    
    print(f"✅ Detected {len(bounce_positions)} bounces")
    print(f"   IN: {in_count} ✅ | OUT: {out_count} ❌\n")
    
    return bounce_positions


def get_ball_shots(ball_detections, video_fps=50):
    # Extract ball positions
    ball_positions = [x.get(1, []) for x in ball_detections]
    df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    
    # Interpolate missing values
    df = df.interpolate()
    df = df.bfill()
    
    # Calculate ball center Y position
    df['mid_y'] = (df['y1'] + df['y2']) / 2
    
    # Apply rolling mean to smooth trajectory
    df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    
    # Calculate velocity (change in Y position)
    df['delta_y'] = df['mid_y_rolling_mean'].diff()
    
    # Initialize hit detection
    df['ball_hit'] = 0
    minimum_change_frames_for_hit = 13
    
    # Detect velocity direction changes (shots)
    for i in range(60, len(df) - int(minimum_change_frames_for_hit * 1.2)):
        # Check if velocity changes from positive to negative or vice versa
        negative_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
        positive_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0
        
        if negative_change or positive_change:
            change_count = 0
            
            # Verify the direction change persists
            for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                negative_following = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[change_frame] < 0
                positive_following = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[change_frame] > 0
                
                if (negative_change and negative_following) or (positive_change and positive_following):
                    change_count += 1
            
            # If change persists for minimum frames, mark as shot
            if change_count > minimum_change_frames_for_hit - 1:
                df.loc[i, 'ball_hit'] = 1
    
    # Extract shot frames
    shot_frames = df[df['ball_hit'] == 1].index.tolist()
    
    # Display results
    print(f"📊 Results:")
    print(f"   Total frames: {len(df)}")
    print(f"   Shots detected: {len(shot_frames)}\n")
    
    if shot_frames:
        print(f"🎯 Shot Frames:")
        print(f"   {shot_frames}\n")
        
        print(f"📋 Detailed Breakdown:")
        for i, frame in enumerate(shot_frames, 1):
            if i < len(shot_frames):
                next_frame = shot_frames[i]
                interval = next_frame - frame
                time_sec = interval / video_fps
                print(f"   Shot {i}: Frame {frame:4d} → Frame {next_frame:4d} "
                      f"({interval} frames, {time_sec:.2f}s)")
            else:
                print(f"   Shot {i}: Frame {frame:4d} (final)")
        
        print(f"\n🎾 Expected Bounce Locations:")
        for i in range(len(shot_frames) - 1):
            shot = shot_frames[i]
            next_shot = shot_frames[i + 1]
            # Bounce typically occurs 60-70% through the interval
            expected_bounce = shot + int((next_shot - shot) * 0.65)
            print(f"   After Shot {i+1}: ~Frame {expected_bounce}")
        
    return shot_frames


def calculate_ball_distance(ball_start, ball_end, court_keypoints):
    """Calculate ball distance with light perspective correction"""
    import constants
    
    court_left = min(court_keypoints[0], court_keypoints[4])
    court_right = max(court_keypoints[2], court_keypoints[6])
    court_top = min(court_keypoints[1], court_keypoints[3])
    court_bottom = max(court_keypoints[5], court_keypoints[7])
    
    court_width_px = court_right - court_left
    court_height_px = court_bottom - court_top
    
    COURT_WIDTH_M = constants.DOUBLE_LINE_WIDTH
    COURT_LENGTH_M = constants.HALF_COURT_LINE_HEIGHT * 2
    
    dx_px = ball_end[0] - ball_start[0]
    dy_px = ball_end[1] - ball_start[1]
    
    # Simple conversion
    px_per_meter_x = court_width_px / COURT_WIDTH_M
    dx_m = abs(dx_px) / px_per_meter_x
    
    # Light perspective adjustment
    avg_y = (ball_start[1] + ball_end[1]) / 2
    y_normalized = (avg_y - court_top) / court_height_px
    perspective_factor = 1.0 + (y_normalized * 0.10)
    
    px_per_meter_y = (court_height_px / COURT_LENGTH_M) * perspective_factor
    dy_m = abs(dy_px) / px_per_meter_y
    
    distance_m = (dx_m**2 + dy_m**2)**0.5
    
    dx_m_signed = dx_px / px_per_meter_x
    dy_m_signed = dy_px / px_per_meter_y
    
    return distance_m, dx_m_signed, dy_m_signed


def is_bounce_in_bounds(bounce_position, mini_court):
    x, y = bounce_position
    
    singles_left_x = mini_court.drawing_key_points[4 * 2]  
    singles_right_x = mini_court.drawing_key_points[6 * 2] 
    
    top_y = mini_court.drawing_key_points[0 * 2 + 1]   
    bottom_y = mini_court.drawing_key_points[2 * 2 + 1]  
    
    # Check bounds
    x_in_bounds = singles_left_x <= x <= singles_right_x
    y_in_bounds = top_y <= y <= bottom_y
    
    return x_in_bounds and y_in_bounds