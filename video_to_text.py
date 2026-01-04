"""
Video Transcription and Analysis Pipeline
=========================================
A comprehensive system for converting video content to descriptive text with scene analysis,
object detection, and temporal understanding using state-of-the-art multimodal AI models.
"""

import cv2
import torch
import numpy as np
import json
import gc
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from PIL import Image

# Computer Vision and AI Imports
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class VideoToText:
    """
    Main class for video-to-text conversion pipeline.
    """

    def __init__(self, device: str = 'auto', model_size: str = 'balanced') -> None:
        """
        Initialize the video analysis pipeline.
        """

        # Hardware configuration with automatic detection
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device.upper()}")

        # Configuration presets
        configs = {
            'fast': {
                'yolo': 'yolov8n.pt',
                'blip': None,
                'sample_rate': 0.3,
                'frame_size': 320,
                'description': 'Fast processing with object detection only'
            },
            'balanced': {
                'yolo': 'yolov8m.pt',
                'blip': 'Salesforce/blip2-opt-2.7b',
                'sample_rate': 0.5,
                'frame_size': 480,
                'description': 'Balanced speed and quality with full analysis'
            },
            'accurate': {
                'yolo': 'yolov8x.pt',
                'blip': 'Salesforce/blip2-opt-2.7b',
                'sample_rate': 1.0,
                'frame_size': 640,
                'description': 'Maximum quality with full frame analysis'
            }
        }

        if model_size not in configs:
            raise ValueError(f"Invalid model_size: {model_size}. Choose from {list(configs.keys())}")
        self.cfg = configs[model_size]
        print(f"Mode: {model_size.upper()} - {self.cfg['description']}")

        # Initialize YOLOv8
        print(f"Loading YOLOv8 model: {self.cfg['yolo']}")
        self.yolo = YOLO(self.cfg['yolo']).to(self.device)

        # Initialize BLIP-2 if configured
        if self.cfg['blip']:
            print(f"Loading BLIP-2 model: {self.cfg['blip']}")
            self.processor = Blip2Processor.from_pretrained(self.cfg['blip'])
            self.blip = Blip2ForConditionalGeneration.from_pretrained(
                self.cfg['blip'],
                torch_dtype=torch.float16 if self.device in ['cuda', 'mps'] else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.blip.eval()
        else:
            self.blip = None
            self.processor = None
            print("BLIP-2 disabled for fast processing mode")

    def analyze(self, video_path: str, output: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis pipeline for video-to-text conversion.
        """

        print(f"\n{'=' * 60}")
        print(f"Starting video analysis: {video_path}")
        print(f"{'=' * 60}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Extract video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Validate video properties
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: FPS={fps}, Frames={total_frames}")

        print(f"Video Metadata:")
        print(f"  Duration: {self._format_time(duration)}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")

        # Calculate sampling
        sampling_interval = max(1, int(fps / self.cfg['sample_rate']))
        frames_to_process = total_frames // sampling_interval

        print(f"\nProcessing Configuration:")
        print(f"  Sample Rate: {self.cfg['sample_rate'] * 100:.0f}% of frames")
        print(f"  Frames to Process: {frames_to_process:,}")

        # Initialize data structures
        all_objects = []
        captions = []
        timeline = []
        frame_idx = 0
        prev_gray = None

        # Progress bar
        progress_bar = tqdm(
            total=frames_to_process,
            desc="Processing Video Frames",
            unit="frame"
        )

        print(f"\n{'=' * 60}")
        print("Starting Frame Processing")
        print(f"{'=' * 60}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sampling_interval == 0:
                # Resize frame
                height, width = frame.shape[:2]
                if width > height:
                    new_width = self.cfg['frame_size']
                    new_height = max(1, int(height * self.cfg['frame_size'] / width))
                else:
                    new_height = self.cfg['frame_size']
                    new_width = max(1, int(width * self.cfg['frame_size'] / height))

                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Motion detection
                current_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    frame_diff = cv2.absdiff(current_gray, prev_gray)
                    mean_diff = np.mean(frame_diff)

                    motion_threshold = 10.0
                    if mean_diff < motion_threshold:
                        frame_idx += 1
                        continue

                prev_gray = current_gray

                # Object detection with YOLOv8
                results = self.yolo(resized_frame, verbose=False)[0]

                frame_objects = []
                if len(results.boxes) > 0:
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        label = self.yolo.names[class_id]
                        confidence = float(box.conf[0])

                        if confidence > 0.3:
                            frame_objects.append(label)
                            all_objects.append(label)

                # Scene description with BLIP-2 (if enabled)
                if self.blip and len(frame_objects) > 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    inputs = self.processor(
                        images=pil_image,
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        caption_ids = self.blip.generate(
                            **inputs,
                            max_new_tokens=30,
                            num_beams=5,
                            early_stopping=True
                        )

                    caption = self.processor.decode(caption_ids[0], skip_special_tokens=True)
                    captions.append(caption)

                    current_time = frame_idx / fps
                    timeline_entry = {
                        "timestamp_seconds": current_time,
                        "formatted_time": self._format_time(current_time),
                        "frame_index": frame_idx,
                        "caption": caption,
                        "objects": frame_objects,
                        "object_count": len(frame_objects),
                        "unique_objects": list(set(frame_objects))
                    }
                    timeline.append(timeline_entry)

                progress_bar.update(1)

                # Memory management
                if progress_bar.n % 50 == 0:
                    if self.device == 'mps':
                        torch.mps.empty_cache()
                    elif self.device == 'cuda':
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()

            frame_idx += 1

        # Cleanup
        cap.release()
        progress_bar.close()

        print(f"\n{'=' * 60}")
        print(f"Processing Complete")
        print(f"{'=' * 60}")
        print(f"Processed {progress_bar.n:,} frames")
        print(f"Generated {len(captions):,} captions")
        print(f"Detected {len(set(all_objects)):,} unique object types")

        # Generate object statistics
        object_counter = Counter(all_objects)
        unique_objects_list = list(set(all_objects))
        top_10_objects = object_counter.most_common(10)

        if timeline:
            print(f"\n{'=' * 60}")
            print("Scene Timeline Analysis (first 10 scenes)")
            print(f"{'=' * 60}")

            for i, entry in enumerate(timeline[:10]):
                obj_counter = Counter(entry['objects'])
                top_objects = obj_counter.most_common(3)

                if top_objects:
                    object_str = ", ".join(f"{obj}({count})" for obj, count in top_objects)
                else:
                    object_str = "no objects detected"

                print(f"{entry['formatted_time']} — {entry['caption']} [{object_str}]")

            if len(timeline) > 10:
                print(f"... and {len(timeline) - 10} more scenes")

        if all_objects:
            print(f"\n{'=' * 60}")
            print("Top 10 Most Frequent Objects")
            print(f"{'=' * 60}")
            for obj, count in top_10_objects:
                percentage = (count / len(all_objects)) * 100
                print(f"{obj:20s} {count:5d} occurrences ({percentage:.1f}%)")

        # Prepare final output structure - ИСПРАВЛЕННАЯ ЧАСТЬ
        result = {
            "video_metadata": {
                "file_path": video_path,
                "duration_seconds": duration,
                "formatted_duration": self._format_time(duration),
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": progress_bar.n,
                "processing_mode": self.cfg['description']
            },
            "analysis_summary": {
                "overall_description": f"Video analysis detected {len(unique_objects_list)} "
                                       f"unique object types across {len(timeline)} key scenes.",
                "primary_objects": unique_objects_list[:10] if unique_objects_list else [],
                "total_object_detections": len(all_objects),
                "caption_count": len(captions)
            },
            "detailed_timeline": timeline,
            "object_statistics": {
                "frequency_distribution": dict(object_counter),
                "unique_objects_count": len(unique_objects_list),  # Это число, а не список!
                "unique_objects_list": unique_objects_list,  # Добавляем список уникальных объектов
                "most_common_objects": dict(top_10_objects) if all_objects else {}
            },
            "processing_configuration": {
                "device": self.device,
                "model_size": self.cfg['description'],
                "sample_rate": self.cfg['sample_rate'],
                "frame_size": self.cfg['frame_size'],
                "yolo_model": self.cfg['yolo'],
                "blip_model": self.cfg['blip'] or "disabled"
            }
        }

        # Save to JSON file if output path provided
        if output:
            try:
                with open(output, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\nResults saved to: {output}")
            except Exception as e:
                print(f"\nWarning: Could not save to {output}: {e}")

        print(f"\n{'=' * 60}")
        print(f"Analysis Complete - {self._format_time(duration)} of video processed")
        print(f"{'=' * 60}")

        return result

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds_remainder = int(seconds % 60)
        return f"{minutes:02d}:{seconds_remainder:02d}"

    def __del__(self) -> None:
        """Cleanup method to free resources."""
        if hasattr(self, 'device'):
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                torch.mps.empty_cache()


def main() -> None:
    """Command-line interface for the VideoToText pipeline."""

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Convert video files to detailed textual descriptions using AI"
    )

    # Required arguments
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (MP4, AVI, MOV, etc.)"
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for JSON output file (optional)"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Processing mode: fast (speed), balanced (default), accurate (quality)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Hardware device to use (default: auto-detect)"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    if not os.path.isfile(args.video):
        print(f"Error: Path is not a file: {args.video}")
        return 1

    # Get file size
    file_size_mb = os.path.getsize(args.video) / (1024 * 1024)
    print(f"Input video: {args.video}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Create and run the analyzer
    try:
        analyzer = VideoToText(device=args.device, model_size=args.mode)
        result = analyzer.analyze(args.video, args.output)

        # Print summary if no output file specified
        if not args.output:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Duration: {result['video_metadata']['formatted_duration']}")
            print(f"Unique Objects: {result['object_statistics']['unique_objects_count']}")
            print(f"Key Scenes: {len(result['detailed_timeline'])}")
            print(f"Processing Mode: {result['processing_configuration']['model_size']}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)