# sign_paper.py -- LSB steganographic IP signing
# Maya-Prana Paper 9 | Nexus Learning Labs
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import argparse
import os
from PIL import Image

SIGNATURE = (
    "MayaNexusVS2026NLL_Bengaluru_Narasimha | "
    "ORCID:0000-0002-3315-7907 | "
    "Nexus Learning Labs Bengaluru | "
    "Maya-Prana Paper 9 | "
    "DOI:10.5281/zenodo.PENDING"
)

def encode_lsb(image_path: str, message: str, output_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())
    bits = ''.join(format(ord(c), '08b') for c in message) + '00000000'
    if len(bits) > len(pixels) * 3:
        raise ValueError(f"Image too small to encode signature ({len(bits)} bits needed, {len(pixels)*3} available)")
    encoded = []
    bit_idx = 0
    for r, g, b in pixels:
        if bit_idx < len(bits):
            r = (r & ~1) | int(bits[bit_idx]); bit_idx += 1
        if bit_idx < len(bits):
            g = (g & ~1) | int(bits[bit_idx]); bit_idx += 1
        if bit_idx < len(bits):
            b = (b & ~1) | int(bits[bit_idx]); bit_idx += 1
        encoded.append((r, g, b))
    out_img = Image.new("RGB", img.size)
    out_img.putdata(encoded)
    out_img.save(output_path)
    print(f"  [sign_paper] Signed: {output_path}")
    print(f"  [sign_paper] Bits embedded: {len(bits)}")

def decode_lsb(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())
    bits = ""
    for r, g, b in pixels:
        bits += str(r & 1)
        bits += str(g & 1)
        bits += str(b & 1)
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        c = chr(int(byte, 2))
        if c == '\x00':
            break
        chars.append(c)
    return ''.join(chars)

if __name__ == "__main__":
    print(f"Signature: {SIGNATURE}")
    parser = argparse.ArgumentParser(description="LSB steganographic figure signing -- Maya-Prana P9")
    parser.add_argument("--input",  required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output signed image path")
    parser.add_argument("--decode", action="store_true", help="Decode and verify signature")
    args = parser.parse_args()
    if args.decode:
        msg = decode_lsb(args.input)
        print(f"  [sign_paper] Decoded: {msg}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        encode_lsb(args.input, SIGNATURE, args.output)
        print("sign_paper.py ready -- Maya-Prana Paper 9.")