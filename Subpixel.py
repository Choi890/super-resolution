"""Legacy import path for the optimized Subpixel layer.

New code should import from ``super_resolution.layers``. This file remains so
old notebooks that use ``from Subpixel import Subpixel`` keep working.
"""

from super_resolution.layers import BicubicUpsample, ClipByValue, PixelShuffle, ResidualScale, Subpixel

__all__ = ["BicubicUpsample", "ClipByValue", "PixelShuffle", "ResidualScale", "Subpixel"]
