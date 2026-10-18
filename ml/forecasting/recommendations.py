"""
AI Mango Orchard Health & Pest Intelligence — Recommendation Engine
Fetches source-cited treatment protocols based on disease and pest diagnosis.
"""

import json
import os
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "recommendations_db.json")

class RecommendationEngine:
    def __init__(self, db_path: str = DB_PATH):
        self.db = {}
        self.meta = {}
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.meta = data.get("_meta", {})
            self.db = {k: v for k, v in data.items() if not k.startswith("_")}

    def get_recommendations_for_condition(self, condition_name: str) -> List[Dict[str, Any]]:
        """Returns verified ICAR/TNAU recommendations for a given disease or pest (copies, tagged with the condition)."""
        return [{**rec, "condition": condition_name} for rec in self.db.get(condition_name, [])]

    @property
    def conditions(self) -> List[str]:
        return list(self.db.keys())

    def get_recommendations_for_scan(self, disease_name: str, pest_names: List[str]) -> List[Dict[str, Any]]:
        results = []
        if disease_name and disease_name != "Healthy":
            results.extend(self.get_recommendations_for_condition(disease_name))

        for pest in pest_names:
            results.extend(self.get_recommendations_for_condition(pest))

        return results

recommendation_engine = RecommendationEngine()
