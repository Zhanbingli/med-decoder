"""
CardioVoice Backend Module
==========================

This module contains the core backend services for CardioVoice:

- unified_model_manager: MedASR + MedGemma unified management
- audio_receiver: Real-time audio stream receiver
- database: PostgreSQL database operations
- api: FastAPI endpoints

Author: CardioVoice Team
Date: January 2026
Version: 1.0.0
"""

from .unified_model_manager import (
    UnifiedModelManager,
    MedASRWrapper,
    MedGemmaWrapper,
    PatientInfo,
    TranscriptionResult,
    OutpatientRecord,
    ModelStatus,
    InferenceBackend,
    create_patient_info
)

__all__ = [
    'UnifiedModelManager',
    'MedASRWrapper', 
    'MedGemmaWrapper',
    'PatientInfo',
    'TranscriptionResult',
    'OutpatientRecord',
    'ModelStatus',
    'InferenceBackend',
    'create_patient_info'
]
