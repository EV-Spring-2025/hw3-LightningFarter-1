from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser(description="Downscale GIF")
parser.add_argument("input", help="Path to input MP4 file")
parser.add_argument("output", help="Path to output GIF file")
parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
args = parser.parse_args()

clip = VideoFileClip(args.input)

# Compress by resizing and reducing FPS
clip_resized = clip.resize(0.8)  # 80% of original size
clip_resized.write_gif(args.output, fps=args.fps)
