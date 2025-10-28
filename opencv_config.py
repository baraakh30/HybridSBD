"""
OpenCV Configuration Module
This module MUST be imported before any other modules that use OpenCV.
It configures environment variables to handle problematic video files.
"""

import os
import sys

def configure_opencv():
    """Configure OpenCV environment variables for better video handling."""
    
    # Set OpenCV FFmpeg options for better video compatibility
    env_vars = {
        'OPENCV_FFMPEG_READ_ATTEMPTS': '16384',  # Increase read attempts for problematic videos
        'OPENCV_LOG_LEVEL': 'ERROR',  # Reduce OpenCV warning verbosity  
        'OPENCV_FFMPEG_DEMUXER_OPTIONS': 'analyzeduration;10000000;probesize;50000000',  # Better stream analysis
        'OPENCV_FFMPEG_CAPTURE_OPTIONS': 'video_stream_id;0',  # Force video stream only
        'OPENCV_VIDEOIO_PRIORITY_FFMPEG': '1',  # Prioritize FFmpeg backend
        'OPENCV_VIDEOIO_DEBUG': '0',  # Disable debug output
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")
        else:
            print(f"{key} already set to: {os.environ[key]}")

# Configure immediately when module is imported
configure_opencv()

# Force environment variable verification
def verify_opencv_config():
    """Verify that OpenCV environment variables are properly set."""
    expected_attempts = '16384'
    current_attempts = os.environ.get('OPENCV_FFMPEG_READ_ATTEMPTS', 'NOT_SET')
    
    if current_attempts != expected_attempts:
        print(f"WARNING: OPENCV_FFMPEG_READ_ATTEMPTS is {current_attempts}, expected {expected_attempts}")
        # Force set it again
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = expected_attempts
        print(f"Force set OPENCV_FFMPEG_READ_ATTEMPTS to {expected_attempts}")
    
    return current_attempts == expected_attempts

# Verify configuration
verify_opencv_config()