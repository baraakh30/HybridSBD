"""
OpenCV Configuration Module
This module MUST be imported before any other modules that use OpenCV.
It configures environment variables to handle problematic video files.
"""

import os
import sys

# Flag to track if we've already printed config messages
_OPENCV_CONFIG_LOGGED = False

def configure_opencv(verbose: bool = False):
    """Configure OpenCV environment variables for better video handling."""
    global _OPENCV_CONFIG_LOGGED
    
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
            if verbose and not _OPENCV_CONFIG_LOGGED:
                print(f"Set {key}={value}")
        # Don't print "already set" messages - they're just noise
    
    _OPENCV_CONFIG_LOGGED = True

# Configure immediately when module is imported (silently)
configure_opencv(verbose=False)

# Force environment variable verification
def verify_opencv_config():
    """Verify that OpenCV environment variables are properly set."""
    expected_attempts = '16384'
    current_attempts = os.environ.get('OPENCV_FFMPEG_READ_ATTEMPTS', 'NOT_SET')
    
    if current_attempts != expected_attempts:
        # Force set it again (silently)
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = expected_attempts
    
    return current_attempts == expected_attempts

# Verify configuration
verify_opencv_config()