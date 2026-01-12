"""
Circuit Extraction Module for Certified Proof-Carrying Circuits

This module provides tools for extracting sparse, interpretable circuits
from neural networks with certified error bounds using BlockCert-style
Lipschitz composition.
"""

from .circuit_extractor import (
    CircuitExtractor,
    CircuitComponent,
    ErrorBound,
    extract_transformer_circuit,
)

__all__ = [
    'CircuitExtractor',
    'CircuitComponent',
    'ErrorBound',
    'extract_transformer_circuit',
]

__version__ = '1.0.0'
