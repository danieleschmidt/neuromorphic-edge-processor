"""Minimal security module for testing."""

from .input_sanitizer import InputSanitizer, global_sanitizer

__all__ = [
    "InputSanitizer",
    "global_sanitizer",
]