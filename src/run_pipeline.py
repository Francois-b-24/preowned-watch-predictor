#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter toutes les étapes d'analyse et modélisation.
Usage: python run_pipeline.py
"""

from models import (
    etape1_personas,
    etape2_segment_prix,
    etape3_baseline,
    etape4_ridge_te,
    etape5_additional,
    etape6_cv,
    etape7_tuning_ridge,
    etape8_tree,
    etape9_shap,
    etape10_curves,
    etape11_optuna,
    etape12_retrain_and_save
)

def main():
    """
    Exécute toutes les étapes en séquence.
    """
    etape1_personas()
    etape2_segment_prix()
    etape3_baseline()
    etape4_ridge_te()
    etape5_additional()
    etape6_cv()
    etape7_tuning_ridge()
    etape8_tree()
    etape9_shap()
    etape10_curves()
    etape11_optuna()
    etape12_retrain_and_save()

if __name__ == "__main__":
    main()
