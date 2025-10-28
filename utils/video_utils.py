"""
Video processing utilities.
"""

# CRITICAL: Configure OpenCV environment variables BEFORE any OpenCV imports
import os
os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '16384')  # Increase read attempts
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # Reduce verbose warnings
os.environ.setdefault('OPENCV_FFMPEG_DEMUXER_OPTIONS', 'analyzeduration;10000000;probesize;50000000')  # Better stream analysis
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'video_stream_id;0')  # Force video stream only

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
import time
from pathlib import Path


class VideoProcessor:
    """
    Video processing utilities for loading and preprocessing videos.
    """
    
    def __init__(self, config):
        """
        Initialize video processor.
        
        Args:
            config: Processing configuration
        """
        self.target_size = config.target_size
        self.frame_rate = config.frame_rate
        
    def load_video(self, video_path: str, max_frames: Optional[int] = None, 
                  max_retries: int = 3, skip_corrupted: bool = True) -> np.ndarray:
        """
        Load video frames from file with robust error handling.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            max_retries: Maximum number of retry attempts for corrupted videos
            skip_corrupted: Whether to skip corrupted frames instead of failing
            
        Returns:
            Array of frames with shape (N, H, W, C)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be loaded after retries
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure OpenCV environment variables are properly set (should be set at module level, but verify)
        os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '16384')  # Increase read attempts
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # Reduce verbose warnings
        os.environ.setdefault('OPENCV_FFMPEG_DEMUXER_OPTIONS', 'analyzeduration;10000000;probesize;50000000')  # Better stream analysis
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'video_stream_id;0')  # Force video stream only
        
        for attempt in range(max_retries):
            try:
                return self._load_video_attempt(video_path, max_frames, skip_corrupted, attempt)
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Video loading attempt {attempt + 1} failed for {video_path}: {e}. Retrying...")
                    time.sleep(0.5)  # Brief pause before retry
                else:
                    raise RuntimeError(f"Failed to load video {video_path} after {max_retries} attempts: {e}")
    
    def _load_video_attempt(self, video_path: str, max_frames: Optional[int], 
                          skip_corrupted: bool, attempt: int) -> np.ndarray:
        """
        Single attempt to load video with error handling.
        """
        cap = None
        try:
            # Try different backends for better compatibility, prioritizing non-FFmpeg backends first  
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_ANY]  # Try Windows-specific backends first
            cap_opened = False
            
            for backend in backends:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    cap_opened = True
                    break
                if cap:
                    cap.release()
            
            if not cap_opened:
                raise RuntimeError(f"Failed to open video with any backend: {video_path}")
            
            # Check basic video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_count <= 0 or fps <= 0:
                # Try to read first frame to validate
                ret, test_frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Video appears corrupted - no readable frames: {video_path}")
                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frames = []
            frames_read = 0
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while frames_read < (max_frames or float('inf')):
                ret, frame = cap.read()
                
                if not ret:
                    # Check if we've reached the end or hit an error
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    
                    if current_pos >= total_frames - 1:
                        break  # Normal end of video
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        if skip_corrupted and frames:
                            logging.warning(f"Stopping at corrupted section of {video_path} with {len(frames)} frames")
                            break
                        else:
                            raise RuntimeError(f"Too many consecutive read failures in {video_path}")
                    continue
                
                consecutive_failures = 0  # Reset on successful read
                
                # Validate frame
                if frame is None or frame.size == 0:
                    if skip_corrupted:
                        continue
                    else:
                        raise RuntimeError(f"Empty frame encountered in {video_path}")
                
                try:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame if needed
                    if self.target_size:
                        frame = cv2.resize(frame, self.target_size)
                    
                    frames.append(frame)
                    frames_read += 1
                    
                except cv2.error as e:
                    if skip_corrupted:
                        logging.warning(f"Skipping corrupted frame {frames_read} in {video_path}: {e}")
                        continue
                    else:
                        raise RuntimeError(f"Frame processing error in {video_path}: {e}")
            
            if not frames:
                raise RuntimeError(f"No valid frames loaded from video: {video_path}")
            
            return np.array(frames)
            
        finally:
            if cap:
                cap.release()
    
    def validate_video(self, video_path: str) -> dict:
        """
        Validate video file and return diagnostic information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with validation results and video info
        """
        result = {
            'valid': False,
            'error': None,
            'file_exists': False,
            'file_size': 0,
            'can_open': False,
            'has_video_stream': False,
            'frame_count': 0,
            'fps': 0,
            'duration': 0,
            'resolution': (0, 0),
            'codec': 'unknown'
        }
        
        try:
            # Check file existence and size
            if not os.path.exists(video_path):
                result['error'] = 'File does not exist'
                return result
            
            result['file_exists'] = True
            result['file_size'] = os.path.getsize(video_path)
            
            if result['file_size'] == 0:
                result['error'] = 'File is empty'
                return result
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                result['error'] = 'Cannot open with OpenCV'
                return result
            
            result['can_open'] = True
            
            # Get video properties
            result['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            result['resolution'] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            
            if result['fps'] > 0:
                result['duration'] = result['frame_count'] / result['fps']
            
            # Try to read first frame
            ret, frame = cap.read()
            if ret and frame is not None:
                result['has_video_stream'] = True
                result['valid'] = True
            else:
                result['error'] = 'Cannot read frames'
            
            cap.release()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_frame_rate(self, video_path: str) -> float:
        """Get the frame rate of a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 30.0  # Default fallback
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return fps if fps > 0 else 30.0
    
    def get_video_info(self, video_path: str) -> dict:
        """Get comprehensive video information."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def extract_keyframes(self, boundaries: List[int], frames: np.ndarray, 
                         output_dir: str) -> None:
        """
        Extract keyframes at shot boundaries.
        
        Args:
            boundaries: List of boundary frame indices
            frames: Video frames
            output_dir: Directory to save keyframes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, boundary in enumerate(boundaries):
            if 0 <= boundary < len(frames):
                keyframe = frames[boundary]
                
                # Convert RGB to BGR for saving
                keyframe_bgr = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(output_dir, f"keyframe_{i:04d}_frame_{boundary:06d}.jpg")
                cv2.imwrite(output_path, keyframe_bgr)
    
    def create_shot_segments(self, boundaries: List[int], video_path: str, 
                           output_dir: str) -> None:
        """
        Create video segments based on shot boundaries.
        
        Args:
            boundaries: List of boundary frame indices
            video_path: Input video path
            output_dir: Directory to save segments
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Add start and end boundaries
        all_boundaries = [0] + boundaries + [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        
        for i in range(len(all_boundaries) - 1):
            start_frame = all_boundaries[i]
            end_frame = all_boundaries[i + 1]
            
            output_path = os.path.join(output_dir, f"shot_{i:04d}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
        
        cap.release()