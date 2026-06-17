"""Minimal inference example.

Usage:
    python example.py --input image.jpg --output outputs/image_x4.png --weights models/residual-espcn_x4_best.keras
"""

from infer import main


if __name__ == "__main__":
    main()
