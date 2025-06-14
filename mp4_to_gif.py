import argparse
from moviepy.editor import VideoFileClip

def mp4_to_gif(input_path, output_path, fps):
    clip = VideoFileClip(input_path)
    clip.write_gif(output_path, fps=fps)
    print(f"✅ GIF saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert MP4 to GIF")
    parser.add_argument("input", help="Path to input MP4 file")
    parser.add_argument("output", help="Path to output GIF file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")

    args = parser.parse_args()
    mp4_to_gif(args.input, args.output, args.fps)

if __name__ == "__main__":
    main()
