"""Optional MongoDB persistence for anomaly reports.

MongoDB rather than a vector database because anomaly reports are
heterogeneous documents (nested metric rankings, free-text explanations)
queried by machine id and time range — there is no similarity search in
this pipeline, so embeddings would solve a problem we do not have. The
pipeline runs fine without a MongoDB instance; this module is additive.
"""

import json

from . import config


def save_reports(mongo_uri="mongodb://localhost:27017", db="crashlens"):
    from pymongo import MongoClient

    results = json.loads((config.ARTIFACT_DIR / "results.json").read_text())
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    coll = client[db]["anomaly_reports"]
    for seg in results["segments"]:
        coll.update_one(
            {"machine": results["machine"], "start": seg["start"]},
            {"$set": {**seg, "machine": results["machine"]}},
            upsert=True,
        )
    print(f"upserted {len(results['segments'])} reports "
          f"into {db}.anomaly_reports")


if __name__ == "__main__":
    save_reports()
