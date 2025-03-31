#!/bin/bash

# video_dir="$PROJECT_PATH/outputs"
# output_zip="$video_dir/detected_gifs.zip"

# for video in "$video_dir"/*.mp4 "$video_dir"/*.mkv; do
#   if [ -f "$video" ]; then
#     ffmpeg -i "$video" -vf "fps=10,scale=320:-1:flags=lanczos" -c:v gif "${video%.*}.gif"
#   fi
# done
# zip -r "$output_zip" "$video_dir"/*.gif

video_dir="$PROJECT_PATH/outputs"
output_zip="$video_dir/detected_gifs.zip"

for video in "$video_dir"/*.mp4 "$video_dir"/*.mkv; do
  if [ -f "$video" ]; then
    gif="${video%.*}.gif"
    palette="${video%.*}_palette.png"

    # Generate palette
    ffmpeg -y -i "$video" -vf "fps=10,scale=320:-1:flags=lanczos,palettegen" "$palette"

    # Use palette to create optimized GIF
    ffmpeg -y -i "$video" -i "$palette" \
      -filter_complex "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" \
      "$gif"

    rm "$palette"  # Clean up temp palette
  fi
done

zip -r "$output_zip" "$video_dir"/*.gif
