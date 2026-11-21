#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Checking Headers & Directories ==="
mkdir -p src data/input data/output
if [ ! -f "src/stb_image.h" ] || [ ! -f "src/stb_image_write.h" ]; then
  make setup
fi

# Generate 5 multi-resolution test images if data/input is currently empty
if [ -z "$(ls -A data/input 2>/dev/null)" ]; then
  echo "=== Step 2: Generating Sample Batch Images in data/input/ ==="
  python3 - << 'EOF'
import os, struct, zlib, math

def write_png(path, width, height, pattern_id):
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # Filter type 0 (None)
        for x in range(width):
            cx, cy = width // 2, height // 2
            dist = math.hypot(x - cx, y - cy)
            ring = 255 if int(dist / (15 + pattern_id * 5)) % 2 == 0 else 30
            grid = 220 if (x % (40 + pattern_id * 10) < 4 or y % (40 + pattern_id * 10) < 4) else 0
            r = min(255, ring)
            g = min(255, grid)
            b = min(255, int((x / width) * 255))
            raw_data.extend([r, g, b])
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw_data), 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)

specs = [
    ("sample_01_sd.png", 640, 480, 1),
    ("sample_02_hd.png", 1280, 720, 2),
    ("sample_03_fhd.png", 1920, 1080, 3),
    ("sample_04_qhd.png", 2560, 1440, 4),
    ("sample_05_4k.png", 3840, 2160, 5),
]
for name, w, h, pid in specs:
    write_png(os.path.join("data/input", name), w, h, pid)
    print(f"Created {name} ({w}x{h})")
EOF
fi

echo "=== Step 3: Compiling CUDA Project ==="
make

echo "=== Step 4: Running Batch GPU Edge Detection Pipeline ==="
./batch_edge_detector --input data/input --output data/output --csv data/output/benchmark_results.csv

echo "=== Execution Complete! Output Artifacts in data/output/ ==="
ls -lh data/output/
