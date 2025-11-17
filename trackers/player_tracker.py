from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_bbox
import pandas as pd
import numpy as np

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections
    
    def detect_frame(self, frame):
        # Only track people (class 0) and keep tracker state
        results = self.model.track(
            frame, persist=True, classes=[0], verbose=False, max_det=50
        )[0]

        id_name_dict = results.names  # unused now but kept for parity

        candidates = []
        for box in results.boxes:
            # some boxes won't have IDs on certain frames → skip safely
            tid = int(box.id.item()) if (hasattr(box, "id") and box.id is not None) else None
            if tid is None:
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            # prefer larger boxes (players on-court tend to be largest)
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            conf = float(box.conf.item()) if hasattr(box, "conf") and box.conf is not None else 0.0

            candidates.append((tid, xyxy, area, conf))

        # sort by area (desc), tie-break by confidence (desc)
        candidates.sort(key=lambda t: (t[2], t[3]), reverse=True)

        # keep at most 9 people
        kept = candidates[:9]

        player_dict = {tid: xyxy for tid, xyxy, _, _ in kept}
        return player_dict
    
    def choose_players(self, court_keypoints, player_dict):
        """
        Choose 2 players closest to the court keypoints.
        Strategy: Find players on opposite ends (top/bottom) closest to court.
        Filters out officials/ball kids who are far from court or at edges.
        """
        if len(player_dict) < 2:
            return list(player_dict.keys())
        
        # Calculate distance to court and other metrics for each player
        candidates = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_bbox(bbox)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Find minimum distance to any court keypoint
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            
            candidates.append({
                'id': track_id,
                'distance': min_distance,
                'y_pos': player_center[1],
                'area': area,
                'center': player_center
            })
        
        # This removes umpire, spectators, and people on sidelines
        candidates = [c for c in candidates if c['distance'] < 350]
        
        if len(candidates) < 2:
            print(f"WARNING: Only {len(candidates)} players close enough to court")
            return [c['id'] for c in candidates]
        
        # Calculate median area and keep only those above 40% of median
        areas = [c['area'] for c in candidates]
        median_area = sorted(areas)[len(areas)//2]
        min_area = median_area * 0.4
        
        candidates = [c for c in candidates if c['area'] >= min_area]
        
        if len(candidates) < 2:
            print(f"WARNING: Only {len(candidates)} players with sufficient size")
            return [c['id'] for c in candidates]
        
        # Sort by distance to court (closest first)
        candidates.sort(key=lambda x: x['distance'])
        
        # If we have at least 2, try to pick from opposite ends (top/bottom)
        if len(candidates) >= 2:
            # Get the closest player to court
            first_player = candidates[0]
            
            # Try to find second player from opposite end
            first_y = first_player['y_pos']
            opposite_end = [c for c in candidates[1:] if abs(c['y_pos'] - first_y) > 300]
            
            if opposite_end:
                # Choose closest to court from opposite end
                second_player = opposite_end[0]
                print(f"Selected players: ID {first_player['id']} (dist={first_player['distance']:.1f}, y={first_player['y_pos']:.0f}) and ID {second_player['id']} (dist={second_player['distance']:.1f}, y={second_player['y_pos']:.0f})")
            else:
                # Just choose second closest overall
                second_player = candidates[1]
                print(f"WARNING: Both players in same area. IDs: {first_player['id']}, {second_player['id']}")
            
            chosen_players = [first_player['id'], second_player['id']]
        else:
            # Less than 2 players, return what we have
            chosen_players = [c['id'] for c in candidates[:2]]
        
        return chosen_players
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        smart remapping filtering: takes < 10 people and filter the ones on court only
        """
        chosen_ids = []
        reference_frame_idx = None
        
        for i in range(min(30, len(player_detections))):
            if len(player_detections[i]) >= 2:
                chosen_ids = self.choose_players(court_keypoints, player_detections[i])
                if len(chosen_ids) >= 2:
                    reference_frame_idx = i
                    break
        
        if not chosen_ids or reference_frame_idx is None:
            print("WARNING: Could not find 2 players to track")
            return player_detections
        
        print(f"✓ Selected player IDs: {chosen_ids} from frame {reference_frame_idx}")
        
        # Track which YOLO IDs map to our tracked IDs
        id_mapping = {yolo_id: yolo_id for yolo_id in chosen_ids}
        
        # Store player tracking info
        last_positions = {}
        for pid in chosen_ids:
            if pid in player_detections[reference_frame_idx]:
                bbox = player_detections[reference_frame_idx][pid]
                center = get_center_bbox(bbox)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                last_positions[pid] = {
                    'center': center,
                    'area': area,
                    'bbox': bbox,
                    'frames_missing': 0
                }
        
        # Determine player roles
        player_roles = {}
        if len(chosen_ids) >= 2:
            ids = list(chosen_ids)
            y0 = last_positions[ids[0]]['center'][1]
            y1 = last_positions[ids[1]]['center'][1]
            
            if y0 < y1:
                player_roles[ids[0]] = 'top'
                player_roles[ids[1]] = 'bottom'
                print(f"  Player {ids[0]} = TOP (y={y0:.0f})")
                print(f"  Player {ids[1]} = BOTTOM (y={y1:.0f})")
            else:
                player_roles[ids[0]] = 'bottom'
                player_roles[ids[1]] = 'top'
                print(f"  Player {ids[0]} = BOTTOM (y={y0:.0f})")
                print(f"  Player {ids[1]} = TOP (y={y1:.0f})")
        
        # Filter with smart remapping
        filtered = []
        
        for frame_idx, det in enumerate(player_detections):
            mapped_frame = {}
            
            # Add detections with known mapped IDs
            for yolo_id, bbox in det.items():
                if yolo_id in id_mapping:
                    tracked_id = id_mapping[yolo_id]
                    mapped_frame[tracked_id] = bbox
                    
                    # Update last known position
                    # BUT: Only if player is fully visible (not cut off at edges)
                    center = get_center_bbox(bbox)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    # Check if player is too close to frame edges (might be cut off)
                    # Don't use this position for future matching if near edges
                    is_near_edge = (center[0] < 100 or center[0] > 1820 or  # Left/right edges
                                   center[1] < 100 or center[1] > 980)      # Top/bottom edges
                    
                    if not is_near_edge:
                        # Good position - player fully visible
                        last_positions[tracked_id] = {
                            'center': center,
                            'area': area,
                            'bbox': bbox,
                            'frames_missing': 0
                        }
                    else:
                        # Near edge - just reset missing counter but keep old position for matching
                        last_positions[tracked_id]['frames_missing'] = 0
            
            # Update missing counters
            for tracked_id in chosen_ids:
                if tracked_id not in mapped_frame:
                    last_positions[tracked_id]['frames_missing'] += 1
            
            # Smart remapping: Only if player missing for 10+ frames
            missing_ids = set(chosen_ids) - set(mapped_frame.keys())
            
            if missing_ids and det:
                unmapped_ids = [yid for yid in det.keys() if yid not in id_mapping]
                
                for missing_id in missing_ids:
                    frames_missing = last_positions[missing_id]['frames_missing']
                    
                    # Only try remapping if missing for 10-100 frames
                    if frames_missing < 10 or frames_missing > 100:
                        if frames_missing == 10 or frames_missing == 101:
                            print(f"  Player {missing_id} ({player_roles.get(missing_id)}): missing {frames_missing}f - {'waiting' if frames_missing == 10 else 'gave up'}")
                        continue
                    
                    # Only check every 5 frames to reduce overhead
                    if frames_missing % 5 != 0:
                        continue
                    
                    last_pos = last_positions[missing_id]
                    
                    # Debug
                    print(f"\n🔍 Frame {frame_idx}: Looking for missing Player {missing_id} ({player_roles.get(missing_id)}, missing {frames_missing}f)")
                    print(f"   Last known: center={last_pos['center']}, area={last_pos['area']:.0f}")
                    print(f"   Checking {len(unmapped_ids)} unmapped IDs: {unmapped_ids}")
                    
                    best_candidate = None
                    best_score = float('inf')
                    rejection_log = []
                    
                    # Evaluate each unmapped detection
                    for yolo_id in unmapped_ids:
                        bbox = det[yolo_id]
                        center = get_center_bbox(bbox)
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        # HARD CONSTRAINT 1: Size must be very similar
                        area_ratio = area / last_pos['area'] if last_pos['area'] > 0 else 1.0
                        if area_ratio < 0.7 or area_ratio > 1.4:
                            rejection_log.append(f"   ✗ ID {yolo_id}: area ratio {area_ratio:.2f} (need 0.7-1.4)")
                            continue
                        
                        # HARD CONSTRAINT 2: Must be on court
                        min_court_dist = float('inf')
                        for i in range(0, len(court_keypoints), 2):
                            court_pt = (court_keypoints[i], court_keypoints[i+1])
                            d = measure_distance(center, court_pt)
                            if d < min_court_dist:
                                min_court_dist = d
                        
                        if min_court_dist > 350:
                            rejection_log.append(f"   ✗ ID {yolo_id}: court dist {min_court_dist:.0f} (need <350)")
                            continue
                        
                        # HARD CONSTRAINT 3: Must be in correct half
                        if len(last_positions) >= 2:
                            other_players_y = [p['center'][1] for tid, p in last_positions.items() 
                                             if tid != missing_id and p['frames_missing'] == 0]
                            
                            if other_players_y:
                                y_boundary = (min(other_players_y + [last_pos['center'][1]]) + 
                                            max(other_players_y + [last_pos['center'][1]])) / 2.0
                                
                                expected_top = (player_roles.get(missing_id) == 'top')
                                current_top = (center[1] < y_boundary)
                                
                                if expected_top != current_top:
                                    rejection_log.append(f"   ✗ ID {yolo_id}: wrong half (y={center[1]:.0f}, boundary={y_boundary:.0f}, expected {'top' if expected_top else 'bottom'})")
                                    continue
                        
                        # Calculate match score
                        area_diff = abs(1.0 - area_ratio)
                        distance = measure_distance(last_pos['center'], center)
                        
                        # Score heavily favors size match
                        score = (area_diff * 1000) + (distance * 0.3) + (min_court_dist * 0.1)
                        
                        print(f"   ✓ ID {yolo_id}: CANDIDATE - area={area:.0f} (ratio {area_ratio:.2f}), center={center}, court_dist={min_court_dist:.0f}, score={score:.0f}")
                        
                        if score < best_score:
                            best_score = score
                            best_candidate = yolo_id
                    
                    # break if no candidates
                    if best_candidate is None:
                        print(f"   ❌ No candidates passed validation:")
                        for log in rejection_log[:10]: 
                            print(log)
                    
                    # Thresholds
                    if best_candidate is not None and best_score < 250:
                        print(f"   ✅ REMAPPING: ID {best_candidate} → Player {missing_id} (score {best_score:.0f})")
                        
                        id_mapping[best_candidate] = missing_id
                        mapped_frame[missing_id] = det[best_candidate]
                        
                        # Update tracking
                        last_positions[missing_id] = {
                            'center': get_center_bbox(det[best_candidate]),
                            'area': (det[best_candidate][2] - det[best_candidate][0]) * (det[best_candidate][3] - det[best_candidate][1]),
                            'bbox': det[best_candidate],
                            'frames_missing': 0
                        }
                    elif best_candidate is not None:
                        print(f"   ⚠️  Best candidate ID {best_candidate} score {best_score:.0f} > threshold 250 (rejected)")
            
            filtered.append(mapped_frame)
        
        both = sum(1 for d in filtered if len(d) >= 2)
        one = sum(1 for d in filtered if len(d) == 1)
        none = sum(1 for d in filtered if len(d) == 0)
        
        print(f"✓ Tracking: {both} frames with both, {one} with one, {none} with none")
        print(f"  (Short gaps <10f will be filled by interpolation)")
        
        return filtered
    
    def interpolate_player_positions(self, player_detections):
        all_ids = set()
        for det in player_detections:
            all_ids.update(det.keys())
        
        if not all_ids:
            return player_detections
        
        print(f"\n✓ Interpolating positions for players: {sorted(all_ids)}")
        
        num_frames = len(player_detections)
        interpolated = [{} for _ in range(num_frames)]
        
        for player_id in all_ids:
            # Extract positions
            positions = []
            for det in player_detections:
                if player_id in det:
                    positions.append(det[player_id])
                else:
                    positions.append([np.nan, np.nan, np.nan, np.nan])
            
            df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
            
            # Count original detections
            orig_valid = df['x1'].notna().sum()
            
            # Interpolate linearly
            df = df.interpolate(method='linear', limit_direction='both', limit=60)
            
            # Forward fill for any remaining NaNs at start
            df = df.ffill(limit=30)
            
            # Backward fill for any remaining NaNs at end
            df = df.bfill(limit=30)
            
            # Count after interpolation
            interp_valid = df['x1'].notna().sum()
            
            # Store interpolated positions
            for i, row in df.iterrows():
                if not pd.isna(row['x1']):
                    interpolated[i][player_id] = [
                        float(row['x1']), float(row['y1']), 
                        float(row['x2']), float(row['y2'])
                    ]
            
            print(f"  Player {player_id}: {orig_valid} -> {interp_valid} frames ({interp_valid - orig_valid} filled)")
        
        orig_count = sum(1 for d in player_detections if len(d) > 0)
        interp_count = sum(1 for d in interpolated if len(d) > 0)
        print(f"✓ Total: {orig_count} -> {interp_count} frames with detections\n")
        
        return interpolated
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", 
                           (int(x1), int(y1 - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                             (0, 255, 0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames