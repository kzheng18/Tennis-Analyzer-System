def normalize_player_ids(player_detections, court_keypoints):
    court_top = min(court_keypoints[1], court_keypoints[3])
    court_bottom = max(court_keypoints[5], court_keypoints[7])
    court_center_y = (court_top + court_bottom) / 2
    
    court_left = min(court_keypoints[0], court_keypoints[4])
    court_right = max(court_keypoints[2], court_keypoints[6])
    court_center_x = (court_left + court_right) / 2
    
    normalized_detections = []
    
    for frame_num, frame_players in enumerate(player_detections):
        # Separate players by court half
        top_players = []
        bottom_players = []
        
        for player_id, bbox in frame_players.items():
            player_y = (bbox[1] + bbox[3]) / 2
            
            if player_y < court_center_y:
                top_players.append(bbox)
            else:
                bottom_players.append(bbox)
        
        # Rebuild frame with normalized IDs
        normalized_frame = {}
        
        # remapping ids
        if bottom_players:
            if len(bottom_players) == 1:
                normalized_frame[1] = bottom_players[0]
            else:
                # Multiple players - pick closest to center
                best = min(bottom_players, 
                          key=lambda bbox: abs((bbox[0] + bbox[2])/2 - court_center_x))
                normalized_frame[1] = best
        
        if top_players:
            if len(top_players) == 1:
                normalized_frame[2] = top_players[0]
            else:
                best = min(top_players, 
                          key=lambda bbox: abs((bbox[0] + bbox[2])/2 - court_center_x))
                normalized_frame[2] = best
        
        normalized_detections.append(normalized_frame)
    
    return normalized_detections