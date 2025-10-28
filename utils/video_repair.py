"""
Video repair utilities for handling common video corruption issues.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging


class VideoRepairer:
    """
    Utility class for attempting to repair common video file issues.
    """
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize video repairer.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable (auto-detected if None)
        """
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.logger = logging.getLogger(__name__)
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Try to find ffmpeg executable in system PATH."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass
        
        # Try common installation paths
        common_paths = [
            'ffmpeg',
            'ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg'
        ]
        
        for path in common_paths:
            if shutil.which(path):
                return path
        
        return None
    
    def can_repair(self) -> bool:
        """Check if repair functionality is available (requires ffmpeg)."""
        return self.ffmpeg_path is not None
    
    def repair_moov_atom(self, input_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Attempt to repair MP4 files with missing or corrupted moov atom.
        
        Args:
            input_path: Path to corrupted video file
            output_path: Path for repaired file (auto-generated if None)
            
        Returns:
            Tuple of (success, output_path_or_error_message)
        """
        if not self.can_repair():
            return False, "ffmpeg not available for repair"
        
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}"
        
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_repaired{input_path_obj.suffix}")
        
        try:
            # Use ffmpeg to rebuild the moov atom
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-c', 'copy',  # Copy streams without re-encoding
                '-movflags', '+faststart',  # Move moov atom to beginning
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                # Verify the repaired file
                if self._verify_video(output_path):
                    return True, output_path
                else:
                    os.remove(output_path)
                    return False, "Repaired file failed verification"
            else:
                error_msg = result.stderr if result.stderr else "Unknown ffmpeg error"
                return False, f"ffmpeg repair failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Repair timeout (file too large or severely corrupted)"
        except Exception as e:
            return False, f"Repair error: {e}"
    
    def transcode_video(self, input_path: str, output_path: Optional[str] = None,
                       target_codec: str = 'h264') -> Tuple[bool, str]:
        """
        Transcode video to a more compatible format as a repair attempt.
        
        Args:
            input_path: Path to problematic video file
            output_path: Path for transcoded file (auto-generated if None)
            target_codec: Target video codec
            
        Returns:
            Tuple of (success, output_path_or_error_message)
        """
        if not self.can_repair():
            return False, "ffmpeg not available for transcoding"
        
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}"
        
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_transcoded.mp4")
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-c:v', target_codec,
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                '-fflags', '+genpts',  # Generate missing presentation timestamps
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(output_path):
                if self._verify_video(output_path):
                    return True, output_path
                else:
                    os.remove(output_path)
                    return False, "Transcoded file failed verification"
            else:
                error_msg = result.stderr if result.stderr else "Unknown transcoding error"
                return False, f"Transcoding failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Transcoding timeout"
        except Exception as e:
            return False, f"Transcoding error: {e}"
    
    def _verify_video(self, video_path: str) -> bool:
        """Quick verification that a video file is readable."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Try to read first few frames
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    cap.release()
                    return False
            
            cap.release()
            return True
            
        except Exception:
            return False
    
    def attempt_repair(self, input_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt multiple repair strategies on a corrupted video.
        
        Args:
            input_path: Path to corrupted video
            output_dir: Directory for repaired files (same as input if None)
            
        Returns:
            Dictionary with repair results
        """
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_path_obj = Path(input_path)
        base_name = input_path_obj.stem
        
        results = {
            'input_file': input_path,
            'original_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0,
            'repair_attempts': [],
            'successful_repairs': [],
            'best_repair': None
        }
        
        if not self.can_repair():
            results['error'] = "ffmpeg not available for repairs"
            return results
        
        # Strategy 1: Fix moov atom
        moov_output = os.path.join(output_dir, f"{base_name}_moov_repaired.mp4")
        success, result_msg = self.repair_moov_atom(input_path, moov_output)
        
        attempt_result = {
            'strategy': 'moov_repair',
            'success': success,
            'output_path': result_msg if success else None,
            'error': result_msg if not success else None
        }
        results['repair_attempts'].append(attempt_result)
        
        if success:
            results['successful_repairs'].append(attempt_result)
            if results['best_repair'] is None:
                results['best_repair'] = attempt_result
        
        # Strategy 2: Transcode to H.264
        transcode_output = os.path.join(output_dir, f"{base_name}_transcoded.mp4")
        success, result_msg = self.transcode_video(input_path, transcode_output)
        
        attempt_result = {
            'strategy': 'transcode_h264',
            'success': success,
            'output_path': result_msg if success else None,
            'error': result_msg if not success else None
        }
        results['repair_attempts'].append(attempt_result)
        
        if success:
            results['successful_repairs'].append(attempt_result)
            # Prefer transcoded version if moov repair didn't work
            if results['best_repair'] is None or results['best_repair']['strategy'] == 'moov_repair':
                results['best_repair'] = attempt_result
        
        return results


def attempt_video_repair(video_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to attempt video repair and return the best result.
    
    Args:
        video_path: Path to corrupted video
        output_dir: Directory for repaired files
        
    Returns:
        Path to successfully repaired video, or None if repair failed
    """
    repairer = VideoRepairer()
    results = repairer.attempt_repair(video_path, output_dir)
    
    if results['best_repair'] and results['best_repair']['success']:
        return results['best_repair']['output_path']
    
    return None


def batch_repair_videos(video_dir: str, output_dir: Optional[str] = None, 
                       file_patterns: list = None) -> Dict[str, Any]:
    """
    Attempt to repair multiple videos in a directory.
    
    Args:
        video_dir: Directory containing videos to repair
        output_dir: Directory for repaired videos (same as input if None)
        file_patterns: File patterns to match (default: common video extensions)
        
    Returns:
        Dictionary with batch repair results
    """
    if file_patterns is None:
        file_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    if output_dir is None:
        output_dir = video_dir
    
    video_dir_path = Path(video_dir)
    video_files = []
    
    for pattern in file_patterns:
        video_files.extend(video_dir_path.glob(pattern))
    
    repairer = VideoRepairer()
    
    batch_results = {
        'total_files': len(video_files),
        'attempted_repairs': 0,
        'successful_repairs': 0,
        'failed_repairs': 0,
        'repair_details': {}
    }
    
    for video_file in video_files:
        try:
            # Quick check if file is already readable
            import cv2
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    # File seems OK, skip repair
                    continue
            
            batch_results['attempted_repairs'] += 1
            
            results = repairer.attempt_repair(str(video_file), output_dir)
            batch_results['repair_details'][str(video_file)] = results
            
            if results['best_repair'] and results['best_repair']['success']:
                batch_results['successful_repairs'] += 1
            else:
                batch_results['failed_repairs'] += 1
                
        except Exception as e:
            batch_results['failed_repairs'] += 1
            batch_results['repair_details'][str(video_file)] = {'error': str(e)}
    
    return batch_results