"""
Extract the first video's results from detection_results.json files
and create smaller files for easier viewing.
"""
import json
from pathlib import Path


def extract_first_video_results(results_file_path, output_file_path):
    """
    Extract results for the first video from a detection results JSON file.
    Only keeps metrics, not the detected boundaries to save space.
    
    Args:
        results_file_path: Path to the full detection_results.json file
        output_file_path: Path where to save the first video's results
    """
    print(f"Loading results from: {results_file_path}")
    
    with open(results_file_path, 'r') as f:
        all_results = json.load(f)
    
    # Get the first video ID
    video_ids = list(all_results.keys())
    if not video_ids:
        print("No videos found in results file!")
        return None
    
    first_video_id = video_ids[0]
    first_video_data = all_results[first_video_id]
    
    # Remove detected_boundaries to save space, keep only metrics
    cleaned_results = {}
    for feature_type, feature_data in first_video_data.items():
        cleaned_results[feature_type] = {}
        for preprocessing, prep_data in feature_data.items():
            cleaned_results[feature_type][preprocessing] = {}
            for method, method_data in prep_data.items():
                if isinstance(method_data, dict) and 'metrics' in method_data:
                    # Keep only metrics, remove detected_boundaries
                    cleaned_results[feature_type][preprocessing][method] = {
                        'metrics': method_data['metrics'],
                        'num_boundaries_detected': len(method_data.get('detected_boundaries', []))
                    }
                else:
                    cleaned_results[feature_type][preprocessing][method] = method_data
    
    first_video_results = {first_video_id: cleaned_results}
    
    print(f"Extracted results for video: {first_video_id}")
    print(f"Number of total videos in file: {len(video_ids)}")
    
    # Save the first video's results
    with open(output_file_path, 'w') as f:
        json.dump(first_video_results, f, indent=2)
    
    print(f"Saved first video results to: {output_file_path} (without detected boundaries)")
    
    return first_video_id, first_video_results


def main():
    base_dir = Path(__file__).parent / "threshold_detection_results"
    
    # Process sampling_8 results
    print("\n" + "="*60)
    print("Processing SAMPLING_8 results")
    print("="*60)
    sampling_input = base_dir / "sampling_8" / "detection_results.json"
    sampling_output = base_dir / "sampling_8" / "detection_results_first_video.json"
    
    if sampling_input.exists():
        video_id, results = extract_first_video_results(sampling_input, sampling_output)
    else:
        print(f"File not found: {sampling_input}")
    
    # Process no_sampling results
    print("\n" + "="*60)
    print("Processing NO_SAMPLING results")
    print("="*60)
    no_sampling_input = base_dir / "no_sampling" / "detection_results.json"
    no_sampling_output = base_dir / "no_sampling" / "detection_results_first_video.json"
    
    if no_sampling_input.exists():
        video_id, results = extract_first_video_results(no_sampling_input, no_sampling_output)
    else:
        print(f"File not found: {no_sampling_input}")
    
    print("\n" + "="*60)
    print("Extraction completed!")
    print("="*60)


if __name__ == "__main__":
    main()
