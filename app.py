#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import RateLimitError
import os, json, traceback
from collections import Counter
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
from pymongo import MongoClient
from bson import ObjectId
from openai import OpenAI
from functools import lru_cache
from datetime import datetime


# ============================================================
# ENV
# ============================================================
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

# ============================================================
# APP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
PORT = int(os.getenv("PORT", 8000))

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute"]
)



client = MongoClient(MONGO_URI)
db = client["OmniVillage"]
llm = OpenAI(api_key=OPENAI_API_KEY)

MAX_OUTPUT_TOKENS = 700


# ============================================================
# SAFE JSON SERIALIZATION (REQUIRED)
# ============================================================
from datetime import datetime
from bson import ObjectId

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def stringify(obj, indent=0):
    pad = " " * indent
    if isinstance(obj, dict):
        return "\n".join(
            f"{pad}{k}: {stringify(v, indent+2)}"
            for k, v in obj.items()
        )
    if isinstance(obj, list):
        return "\n".join(stringify(v, indent+2) for v in obj)
    return str(obj)

# ============================================================
# GENERIC DROPDOWN LOOKUP (GLOBAL)
# ============================================================
def build_lookup(collection):
    lookup = {}
    for d in collection.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    return lookup

# ============================================================
# 🔍 DOMAIN DETECTION
# ============================================================
def detect_domain(question):
    q = question.lower()

    if any(w in q for w in [
        "population", "demographic", "demography", "census",
        "health", "education", "skills", "occupation",
        "disability", "social"
    ]):
        return "demographics"

    if any(w in q for w in [
        "energy", "electricity", "petrol", "fuel",
        "power", "microgrid", "renewable", "grid"
    ]):
        return "energy"

    if any(w in q for w in [
        "crop", "crops", "food", "consume", "consumption",
        "agriculture", "grain", "vegetable", "fruit", "diet"
    ]):
        return "agriculture"

    if any(w in q for w in [
        "fish", "fisheries", "pond", "feed", "aquatic"
    ]):
        return "fisheries"

    if any(w in q for w in [
        "forest", "forestry", "timber"
    ]):
        return "forestry"

    if any(w in q for w in [
        "house", "housing", "amenities", "home"
    ]):
        return "housing"

    if any(w in q for w in [
        "hunt", "hunting", "wildlife", "meat"
    ]):
        return "hunting"
    
    if any(w in q for w in [
        "mobility", "transport", "road", "vehicle", "travel"
    ]):
        return "mobility"
    if any(w in q for w in [
    "water", "drinking water", "irrigation", "groundwater",
    "water source", "water quality", "scarcity", "sewage"
    ]):
       return "water"
    if any(w in q for w in ["household items", "personal items", "daily items"]):
       return "household_items"
    

    return "business"



# ============================================================
# HELPERS
# ============================================================
def num(v):
    try:
        return float(v)
    except:
        return 0.0

def safe_str(v):
    return v.strip() if isinstance(v, str) else ""

from functools import lru_cache

MAX_PROMPT_CHARS = 12000   # safe upper bound for gpt-4o-mini

def compact_summary(data, limit=MAX_PROMPT_CHARS):
    """
    Prevents token explosion by truncating serialized data.
    Does NOT change the data itself — only what is sent to LLM.
    """
    text = stringify(data)
    if len(text) > limit:
        return text[:limit] + "\n...[DATA TRUNCATED FOR MODEL SAFETY]"
    return text


# ============================================================
# 1️⃣ BUSINESS COMMERCIAL METRICS
# ============================================================
def business_commercial_metrics():
    summary = Counter()
    enterprises = []

    for b in db.business_commercials.find({}):
        emp = num(b.get("total_employee"))
        turn = num(b.get("annual_turnover"))
        prof = num(b.get("total_profit"))
        loss = num(b.get("total_loss"))

        summary["enterprises"] += 1
        summary["employees"] += emp
        summary["turnover"] += turn
        summary["profit"] += prof
        summary["loss"] += loss
        summary["active"] += 1 if b.get("status") == 1 else 0
        summary["inactive"] += 1 if b.get("status") == 0 else 0

        enterprises.append({
            "name": safe_str(b.get("business_name")),
            "type": safe_str(b.get("business_type")),
            "employees": emp,
            "turnover": turn,
            "profit": prof,
            "loss": loss,
            "water": num(b.get("water_consumption")),
            "energy": num(b.get("energy_consumption")),
            "land": num(b.get("land_area_utilised")),
            "status": "Active" if b.get("status") == 1 else "Inactive"
        })

    return {
        "summary": dict(summary),
        "enterprises": enterprises
    }

# ============================================================
# 2️⃣ BUSINESS BY USERS
# ============================================================
def business_by_users_metrics():
    intent = Counter()
    purposes = Counter()
    land_req = Counter()

    for d in db.business_by_users.find({}):
        if d.get("other_business_apart_farming"):
            intent["existing_non_farm_business"] += 1
        if d.get("plan_to_start_business"):
            intent["planning_new_business"] += 1
        if d.get("purpose_of_business"):
            purposes[safe_str(d.get("purpose_of_business")).lower()] += 1
        land_req[d.get("require_land", "unknown")] += 1

    return {
        "intent": dict(intent),
        "purposes": dict(purposes),
        "land_requirement": dict(land_req)
    }

# ============================================================
# 3️⃣ BUSINESS OFFICER ESTABLISHMENTS
# ============================================================
def business_officer_establishments():
    establishments = []
    types = Counter()

    for o in db.business_officers.find({}):
        for e in o.get("how_many_establishment", []):
            etype = safe_str(e.get("type"))
            establishments.append({
                "name": safe_str(e.get("name")),
                "type": etype,
                "purpose": safe_str(e.get("purpose")),
                "year_started": e.get("year_started")
            })
            if etype:
                types[etype.lower()] += 1

    return {
        "establishments": establishments,
        "type_distribution": dict(types)
    }

# ============================================================
# 4️⃣ BUSINESS DROPDOWNS
# ============================================================
def business_dropdowns():
    return {
        "business_dropdowns": list(db.business_dropdowns.find({}, {"_id": 0})),
        "business_officer_dropdowns": list(db.business_officer_dropdowns.find({}, {"_id": 0}))
    }

# ============================================================
# BUSINESS KNOWLEDGE GRAPH
# ============================================================
@lru_cache(maxsize=32)
def business_knowledge_cached():
    return {
        "individual_profiles": business_commercial_metrics()["enterprises"],
        "village_summary": business_commercial_metrics()["summary"],
        "officer_context": business_officer_establishments(),
        "supporting_metadata": business_dropdowns()
    }

def business_llm_payload(data):
    return {
        "village_summary": data["village_summary"],
        "enterprise_profiles": data["individual_profiles"],
        "officer_context": data["officer_context"],
        "metadata": data["supporting_metadata"]
    }


# ============================================================
# 🌾 AGRICULTURE KNOWLEDGE GRAPH (CORRECT & COMPLETE)
# ============================================================
def agriculture_knowledge():
    return {
        "individual_profiles": consumption_metrics(
            consumption_crops_master(),
            consumption_types_master()
        )["records"],

        "village_summary": consumption_metrics(
            consumption_crops_master(),
            consumption_types_master()
        )["village_summary"],

        "officer_context": [],
        "supporting_metadata": {
            "crops": crops_master(),
            "community_officer_dropdowns": community_officer_dropdowns()
        }
    }

def agriculture_llm_payload(data):
    return {
        "village_summary": data["village_summary"],
        "crop_records": data["individual_profiles"],
        "metadata": data["supporting_metadata"]
    }



# ============================================================
# 🌾 CROPS MASTER
# ============================================================
def crops_master():
    out = {}
    for c in db.crops.find({"status": 1}):
        out[str(c["_id"])] = {
            "name": c.get("name", {}).get("en", ""),
            "countries": c.get("country", []),
            "ideal_consumption": num(c.get("ideal_consumption_per_person")),
            "category": safe_str(c.get("category"))
        }
    return out


# ============================================================
# 🍎 CONSUMPTION CROPS MASTER
# ============================================================
def consumption_crops_master():
    out = {}
    for c in db.consumption_crops.find({}):
        out[str(c["_id"])] = {
            "name": safe_str(c.get("name")),
            "consumption_type_id": str(c.get("consumption_type_id"))
        }
    return out


# ============================================================
# 🍽️ CONSUMPTION TYPES MASTER
# ============================================================
def consumption_types_master():
    out = {}
    for t in db.consumption_types.find({}):
        out[str(t["_id"])] = t.get("name", {}).get("en", "")
    return out
# ============================================================
# 🍽️ CONSUMPTION METRICS
# ============================================================
def consumption_metrics(crop_lookup, type_lookup):
    village = Counter()
    per_crop = Counter()
    records = []

    for d in db.consumptions.find({"status": 1}):
        crop = crop_lookup.get(str(d.get("consumption_crop_id")), {})
        crop_name = crop.get("name", safe_str(d.get("consumption_crop_name")))
        ctype = type_lookup.get(str(d.get("consumption_type_id")), "")

        total = num(d.get("total_quantity"))
        market = num(d.get("purchased_from_market"))
        neighbour = num(d.get("purchased_from_neighbours"))
        self_grown = num(d.get("self_grown"))

        village["total"] += total
        village["market"] += market
        village["neighbour"] += neighbour
        village["self_grown"] += self_grown

        per_crop[crop_name] += total

        records.append({
            "crop": crop_name,
            "type": ctype,
            "unit": safe_str(d.get("weight_measurement")),
            "total": total,
            "market": market,
            "neighbour": neighbour,
            "self_grown": self_grown
        })

    return {
        "village_summary": dict(village),
        "crop_totals": dict(per_crop),
        "records": records
    }

# ============================================================
# 🌍 Community officer dropdowns
# ============================================================
def community_officer_dropdowns():
    out = {}
    for d in db.community_officer_dropdowns.find({}):
        dtype = safe_str(d.get("type"))
        name = d.get("name", {}).get("en", "")
        if dtype and name:
            out.setdefault(dtype, []).append(name)
    return out

# ============================================================
# 🌍 Demographic officers
# ============================================================

def demographic_officer_context():
    records = []
    for d in db.demographic_officers.find({}):
        records.append({
            "population_growth_rate": d.get("average_population_growth_rate"),
            "land_unit": d.get("common_land_measurement_unit"),
            "land_amount": d.get("how_much"),
            "local_language": d.get("local_language"),
            "housing_type": d.get("common_traditional_house"),
            "house_images": d.get("upload_house_picture", [])
        })
    return records
# ============================================================
# 🌍 Demographic officers
# ============================================================
def demographics_knowledge():

    # Build lookup directly (already exists logically)
    lookup = {}
    for d in db.demographic_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")

    def resolve(v):
        if isinstance(v, list):
            return [lookup.get(str(x), "") for x in v if str(x) in lookup]
        return lookup.get(str(v), "")

    individuals = []

    for d in db.demographics.find({"status": 1}):
        individuals.append({
            "health": resolve(d.get("motor_disability") or d.get("motor_disablity")),
            "education": resolve(d.get("education_status")),
            "occupation": resolve(d.get("occupation")),
            "skills": resolve(d.get("technical_vocational_skills"))
        })

    return make_json_safe({
        "individual_profiles": individuals,
        "village_summary": {},
        "officer_context": demographic_officer_context(),
        "supporting_metadata": lookup
    })

def demographics_llm_payload(data):
    return {
        "individual_profiles": data["individual_profiles"],
        "officer_context": data["officer_context"],
        "metadata": data["supporting_metadata"]
    }

# ============================================================
# ⚡ Energy dropdown lookup
# ============================================================
def energy_dropdown_lookup():
    lookup = {}
    for d in db.energy_officer_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    return lookup

# ============================================================
# ⚡ Energy officer context (village infrastructure)
# ============================================================
def energy_officer_context():
    dropdowns = energy_dropdown_lookup()
    records = []

    def resolve(v):
        if isinstance(v, list):
            return [dropdowns.get(str(x), "") for x in v if str(x) in dropdowns]
        return dropdowns.get(str(v), "")

    for d in db.energy_officers.find({}):
        records.append({
            "available_renewable_energy": resolve(d.get("available_renewable_energy")),
            "energy_source_present": d.get("energy_source"),
            "capacity": d.get("capacity"),
            "distribution_method": resolve(d.get("distribution_method")),
            "installation_cost": d.get("installation_cost"),
            "central_grid_fossil": d.get("central_grid_fossil"),
            "central_grid_renewable": d.get("central_grid_renewable"),
            "local_renewable_microgrid": d.get("local_renewable_microgrid"),
            "distance_to_pumps": resolve(d.get("distance_to_pumps"))
        })

    return records

# ============================================================
# ⚡ ENERGY KNOWLEDGE GRAPH
# ============================================================
def energy_knowledge():
    dropdowns = energy_dropdown_lookup()
    individuals = []

    def resolve(v):
        if isinstance(v, list):
            return [dropdowns.get(str(x), "") for x in v if str(x) in dropdowns]
        return dropdowns.get(str(v), "")

    for e in db.energy.find({"status": 1}):
        individuals.append({
            "energy_type": e.get("type"),

            "electricity": {
                "connected_to_grid": e.get("electric_grid"),
                "yearly_consumption": e.get("yearly_electricity_consumption"),
                "yearly_expenditure": e.get("yearly_expenditure_electricity"),
                "stability": e.get("electricity_stable")
            },

            "microgrid": {
                "installed": e.get("microgrid_installed"),
                "type": resolve(e.get("microgrid_type")),
                "usage": e.get("usage"),
                "installation_cost": e.get("installation_cost")
            },

            "petrol_usage": {
                "yearly_consumption": e.get("yearly_petrol_consumption"),
                "yearly_expenditure": e.get("yearly_expenditure_petrol"),
                "purpose": resolve(e.get("purpose_petrol_used_for"))
            },

            "fuel_sources": {
                "sources_used": resolve(e.get("source_of_fuels_used"))
            },

            "energy_sufficiency": {
                "sufficient": e.get("energy_sufficient"),
                "extent": e.get("extent")
            }
        })

    return make_json_safe({
        "individual_energy_profiles": individuals,
        "village_energy_infrastructure": energy_officer_context()
    })

def energy_llm_payload(data):
    return {
        "individual_energy_profiles": data["individual_energy_profiles"],
        "village_energy_infrastructure": data["village_energy_infrastructure"]
    }

# ============================================================
# Feeds
# ============================================================
def feeds_master():
    feeds = []

    for f in db.feeds.find({"status": 1}):
        feeds.append({
            "name": f.get("name", {}).get("en", ""),
            "type": f.get("type"),
            "countries": f.get("country", []),
            "status": "Active"
        })

    return feeds

# ============================================================
# Feeds fish
# ============================================================
def fish_feeds_master():
    feeds = []

    for f in db.fish_feeds.find({"status": 1}):
        feeds.append({
            "name": f.get("name", {}).get("en", ""),
            "type": f.get("type"),
            "countries": f.get("country", []),
            "status": "Active"
        })

    return feeds

# ============================================================
# Fishery crops
# ============================================================

def fishery_crops_master():
    crops = []

    for c in db.fishery_crops.find({"status": 1}):
        crops.append({
            "name": c.get("name", {}).get("en", "").strip(),
            "category": c.get("category"),
            "countries": c.get("country", []),
            "ideal_consumption_per_person": c.get("ideal_consumption_per_person", 0),
            "label": str(c.get("label")),
            "status": "Active"
        })

    return crops
 
#============================================================
# Fisheries
# ============================================================

def fisheries_knowledge():
    records = []
    village = Counter()

    for f in db.fisheries.find({"status": 1}):
        info = f.get("important_information", {})
        prod = f.get("production_information", {})

        village["total_fish"] += info.get("number_of_fishes", 0)
        village["total_output"] += prod.get("production_output", 0)
        village["income"] += prod.get("income_from_sale", 0)
        village["expenditure"] += prod.get("expenditure_on_inputs", 0)

        records.append({
            "fishery_type": f.get("fishery_type"),
            "number_of_fishes": info.get("number_of_fishes"),
            "type_of_feed": info.get("type_of_feed"),

            "production": {
                "total_feed": prod.get("total_feed"),
                "output": prod.get("production_output"),
                "self_consumed": prod.get("self_consumed"),
                "sold_to_neighbours": prod.get("sold_to_neighbours"),
                "sold_for_industry": prod.get("sold_for_industrial_use"),
                "wastage": prod.get("wastage"),
                "yield": prod.get("yeild")
            },

            "economics": {
                "income": prod.get("income_from_sale"),
                "expenditure": prod.get("expenditure_on_inputs")
            }
        })

    return {
        "individual_fisheries": records,
        "village_summary": dict(village)
    }

def fisheries_llm_payload(data):
    return {
        "individual_fisheries": data["individual_fisheries"],
        "village_summary": data["village_summary"]
    }

#============================================================
# forestry_dropdowns
# ============================================================

def forestry_dropdown_lookup():
    lookup = {}
    for d in db.forestry_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    return lookup

def forestry_knowledge():
    dropdowns = forestry_dropdown_lookup()
    households = []

    def resolve(v):
        if isinstance(v, list):
            return [dropdowns.get(str(x), "") for x in v]
        return dropdowns.get(str(v), "")

    for f in db.forestries.find({}):
        households.append({
            "forest_type": f.get("type"),
            "land_under_forest": f.get("land_owned_under_forest_cover"),
            "community_forest": f.get("community_forest"),
            "timber_harvested": f.get("timber_logs_harvested"),
            "forest_products": [
                {
                    "product": resolve(p.get("type")),
                    "quantity": p.get("quantity"),
                    "unit": resolve(p.get("quantity_unit")),
                    "purpose": resolve(p.get("purpose"))
                }
                for p in f.get("other_produced_harvested_from_forest", [])
            ],
            "unfulfilled_needs": f.get("unfulfilled_forest_needs"),
            "urgency": resolve(f.get("urgency"))
        })

    return make_json_safe({
    "household_forestry": households,
    "village_summary": forestry_metrics(),   
    "officer_context": forestry_officer_context()
})


#============================================================
# forestry_officers
# ============================================================


def forestry_officer_context():
    dropdowns = forestry_dropdown_lookup()
    records = []

    for o in db.forestry_officers.find({}):
        records.append({
            "forest_type": dropdowns.get(str(o.get("type_of_forest_accessible")), ""),
            "area_accessible": o.get("area_of_forest_accessible"),
            "flora_fauna_present": o.get("do_you_have_flora_fauna"),
            "forest_condition": dropdowns.get(str(o.get("condition_of_forest_accessible")), ""),
            "forest_fire_incidents": dropdowns.get(str(o.get("incident_of_forest_fire")), ""),
            "wildlife_conflict": o.get("incident_of_wildlife_conflict"),
            "illegal_activity": o.get("any_incident_of_illegal_forest_activities"),
            "remarks": o.get("describe")
        })

    return records
# ============================================================
# 🌲 FORESTRY METRICS (VILLAGE-LEVEL)
# ============================================================
def forestry_metrics():
    summary = Counter()

    for f in db.forestries.find({}):
        summary["households_using_forest"] += 1
        summary["timber_logs"] += f.get("timber_logs_harvested", 0)

        for p in f.get("other_produced_harvested_from_forest", []):
            summary["forest_products_collected"] += p.get("quantity", 0)

    return dict(summary)

def forestry_llm_payload(data):
    return {
        "household_forestry": data["household_forestry"],
        "village_summary": data["village_summary"],
        "officer_context": data["officer_context"]
    }

#============================================================
# housing_dropdowns
# ============================================================
def housing_dropdown_lookup():
    lookup = {}
    for d in db.housing_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    return lookup
#============================================================
# housing by users
# ============================================================
def housing_by_users_metrics():
    summary = Counter()

    for d in db.housing_by_users.find({}):
        if d.get("house_requirements"):
            summary["house_requirements"] += 1
        if d.get("need_new_unit"):
            summary["new_units_needed"] += 1
        if d.get("land_for_new_unit"):
            summary["land_required"] += 1

    return dict(summary)

#============================================================
# housing
# ============================================================
def housing_knowledge():
    dropdowns = housing_dropdown_lookup()
    houses = []

    def resolve(v):
        if isinstance(v, list):
            return [dropdowns.get(str(x), "") for x in v]
        return dropdowns.get(str(v), "")

    for h in db.housings.find({"status": 1}):
        houses.append({
            "house_name": h.get("name_of_the_house"),
            "house_type": h.get("type_of_house"),
            "land_used": h.get("land_utilised_for_family_housing"),
            "built_area": h.get("total_built_area"),
            "floors": h.get("no_of_floors"),
            "units": h.get("no_of_units_built"),
            "year_built": h.get("year_built"),
            "year_renovated": h.get("year_renovated"),
            "year_last_expanded": h.get("year_last_expanded"),

            "amenities": resolve(h.get("amenities")),
            "equipment": resolve(h.get("equipment")),
            "furnishing": resolve(h.get("furnishing")),

            "renovation_required": h.get("renovation_requirement"),
            "expansion_required": h.get("expansion_requirement"),

            "images": {
                "front": h.get("front_photo"),
                "back": h.get("back_photo"),
                "neighbourhood": h.get("neighbourhood_photo"),
                "inside": h.get("inside_living_photo"),
                "kitchen": h.get("kitchen_photo")
            }
        })

    return make_json_safe({
        "houses": houses,
        "housing_needs": housing_by_users_metrics()
    })

#============================================================
# Hunting Crops
# ============================================================
def hunting_crops_master():
    crops = {}
    for c in db.hunting_crops.find({"status": 1}):
        crops[str(c["_id"])] = {
            "name": c.get("name", {}).get("en", ""),
            "countries": c.get("country", []),
            "label": str(c.get("label")),
            "ideal_consumption": c.get("ideal_consumption_per_person")
        }
    return crops

#============================================================
# Hunting
# ============================================================
def hunting_knowledge():
    crops = hunting_crops_master()
    records = []
    village = Counter()

    for h in db.huntings.find({"status": 1}):
        crop = crops.get(str(h.get("hunting_crop_id")), {})
        name = crop.get("name")

        village["animals_hunted"] += h.get("number_hunted", 0)
        village["meat"] += h.get("meat", 0)
        village["income"] += h.get("income_from_sale", 0)

        records.append({
            "animal": name,
            "number_hunted": h.get("number_hunted"),
            "meat_obtained": h.get("meat"),
            "self_consumed": h.get("self_consumed"),
            "sold_neighbours": h.get("sold_to_neighbours"),
            "sold_market": h.get("sold_in_consumer_market"),
            "wastage": h.get("wastage"),
            "income": h.get("income_from_sale"),
            "expenditure": h.get("expenditure_on_inputs"),
            "yield": h.get("yeild"),
            "processing": h.get("processing_method")
        })

    return make_json_safe({
        "individual_hunting": records,
        "village_summary": dict(village)
    })

def housing_llm_payload(data):
    return {
        "houses": data["houses"],
        "housing_needs": data["housing_needs"]
    }


# ============================================================
# 🌱 LAND DROPDOWNS — RAW MAP (ID → NODE)
# ============================================================
def landholding_dropdown_raw():
    raw = {}

    for d in db.landholding_dropdowns.find({}):
        raw[str(d["_id"])] = {
            "name": d.get("name", {}).get("en", ""),
            "type": d.get("type"),
            "parent": str(d.get("parent")) if d.get("parent") else None
        }

    return raw
# ============================================================
# 🌱 LAND Measurement — RAW MAP (ID → NODE)
# ============================================================

def land_measurement_lookup():
    lookup = {}
    for d in db.land_measurements.find({}):   # ✅ EXACT collection name
        name = d.get("name", "")
        symbol = d.get("symbol", "")
        lookup[str(d["_id"])] = (
            f"{name} ({symbol})" if symbol else name
        )
    return lookup

def landholding_dropdown_lookup():
    raw = {}

    for d in db.landholding_dropdowns.find({}):
        raw[str(d["_id"])] = {
            "name": d.get("name", {}).get("en", ""),
            "parent": str(d.get("parent")) if d.get("parent") else None
        }

    resolved = {}

    for _id, item in raw.items():
        if item["parent"] and item["parent"] in raw:
            resolved[_id] = f"{raw[item['parent']]['name']} → {item['name']}"
        else:
            resolved[_id] = item["name"]

    return resolved

# ============================================================
# 🌱 LANDHOLDING_DROPDOWN — RAW MAP (ID → NODE)
# ============================================================

def landholding_knowledge():
    dropdowns = landholding_dropdown_lookup()
    units = land_measurement_lookup()

    households = []

    for l in db.landholdings.find({"status": 1}):
        households.append({
            "location": l.get("land_located"),
            "total_land_area": f"{l.get('total_land_area')} {units.get(str(l.get('area_unit')), '')}",
            "year_purchased": l.get("year_purchased"),
            "land_under_use": l.get("land_under_use"),
            "purposes": [
                {
                    "purpose": dropdowns.get(str(p.get("type")), ""),
                    "category": dropdowns.get(str(p.get("type_category")[0]), ""),
                    "area_utilised": p.get("total_land_area_utilised")
                }
                for p in l.get("purpose_land_utilised_for", [])
            ]
        })

    return make_json_safe({
        "individual_land_profiles": households,
        "village_land_context": landholding_officer_context()
    })
# ============================================================
# 🌱 Landholding — RAW MAP (ID → NODE)
# ============================================================
def land_knowledge():
    return make_json_safe({
        "individual_landholdings": landholding_knowledge(),
        "land_requirements": landholding_by_users_metrics(),
        "village_land_distribution": landholding_officer_context(),
        "land_units": land_measurement_lookup()
    })


# ============================================================
# 🌱 landholding_officers Metrics
# ============================================================

def landholding_officer_context():
    units = land_measurement_lookup()
    records = []

    for o in db.landholding_officers.find({}):
        records.append({
            "total_area_allocated": f"{o.get('total_area_allocated_village')} {units.get(str(o.get('area_unit')), '')}",
            "farming_infrastructure": o.get("farming_community_infrastructure"),
            "unutilized_area": o.get("unutilized_area"),
            "fallow_land": o.get("fallow"),
            "forest_cover": o.get("under_forest"),
            "grassland": o.get("under_grassland"),
            "other_land": o.get("others"),
            "non_resident_owned": o.get("land_owned_by_non_resident"),
            "private_land": o.get("total_area_privately_owned")
        })

    return records

# ============================================================
# 🌱 landholding_by_users
# ============================================================
def landholding_by_users_metrics():
    summary = Counter()

    for d in db.landholding_by_users.find({}):
        if d.get("land_requirements"):
            summary["households_needing_land"] += 1
        if d.get("required_area"):
            summary["total_required_area"] += d.get("required_area") or 0

    return dict(summary)

def land_llm_payload(data):
    return {
        "individual_landholdings": data["individual_landholdings"],
        "village_land_distribution": data["village_land_distribution"],
        "land_requirements": data["land_requirements"],
        "land_units": data["land_units"]
    }


# ============================================================
# 🌱 mobility_dropdowns
# ============================================================
def mobility_dropdown_lookup():
    lookup = {}
    for d in db.mobility_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    for d in db.mobility_officer_dropdowns.find({}):
        lookup[str(d["_id"])] = d.get("name", {}).get("en", "")
    return lookup

def mobility_knowledge():
    dropdowns = mobility_dropdown_lookup()

    def resolve(v):
        if isinstance(v, list):
            return [dropdowns.get(str(x), "") for x in v if str(x) in dropdowns]
        return dropdowns.get(str(v), "")

    household_usage = []
    household_needs = []
    officer_context = []

    # -------------------------------
    # Household mobility behaviour
    # -------------------------------
    for m in db.mobilities.find({"status": 1}):
        household_usage.append({
            "vehicle_type": resolve(m.get("type")),
            "distance_within_village": m.get("distance_travelled_within_village"),
            "distance_outside_village": m.get("distance_travelled_outside"),
            "travel_purpose": resolve(m.get("purpose_use_of_vehicle")),
            "frequency": m.get("frequency_of_usage")
        })

    # -------------------------------
    # Household mobility needs
    # -------------------------------
    for m in db.mobility_by_users.find({}):
        household_needs.append({
            "mobility_methods": resolve(m.get("methods_of_mobility")),
            "public_transport_access": m.get("access_to_public_transport"),
            "vehicle_required": m.get("vehicle_requirement"),
            "vehicles_needed": [
                {
                    "purpose": v.get("purpose"),
                    "vehicle_type": resolve(v.get("vehicle_type")),
                    "urgency": v.get("urgency")
                }
                for v in m.get("vehicles_needed", [])
            ]
        })
        
   


    # -------------------------------
    # Officer-level village context
    # -------------------------------
    for o in db.mobility_officers.find({}):
        officer_context.append({
            "houses_connected_to_internal_roads": o.get("house_connected_to_internal_road"),
            "houses_not_connected": o.get("house_not_connected_to_internal_road"),
            "highway_connectivity": o.get("village_connected_to_highway"),
            "mobility_requirements": resolve(o.get("mobility_requirement")),
            "road_condition": resolve(o.get("condition_of_internal_roads")),
            "road_safety_issues": o.get("safety_issues_on_roads"),
            "healthcare_connectivity": resolve(o.get("connectivity_to_healthcare_facilities")),
            "road_damage": o.get("road_infrastructure_damaged"),
            "road_damage_frequency": resolve(o.get("road_damage_frequency")),
            "remarks": o.get("describe")
        })

    return make_json_safe({
        "household_mobility_usage": household_usage,
        "household_mobility_needs": household_needs,
        "village_mobility_infrastructure": officer_context
    })

def water_knowledge():
    dropdowns = build_lookup(db.water_dropdowns)
    officer_dropdowns = build_lookup(db.water_officer_dropdowns)

    def resolve(v, source):
        if isinstance(v, list):
            return [source.get(str(x), "") for x in v]
        return source.get(str(v), "")

    household_profiles = []
    for w in db.waters.find({"status": 1}):
        household_profiles.append({
            "use_type": w.get("type"),
            "yearly_consumption": resolve(w.get("yearly_consumption"), dropdowns),
            "water_source": resolve(w.get("water_sourced_from"), dropdowns),
            "water_quality": resolve(w.get("water_quality"), dropdowns),
            "expense": w.get("expense"),
            "water_scarcity": w.get("water_scarcity"),
            "scarcity_severity": resolve(w.get("water_scarcity_severity"), dropdowns),
            "recycling": w.get("water_recycle")
        })

    village_context = []
    for o in db.water_officers.find({}):
        village_context.append({
            "sources_available": [
                {
                    "type": resolve(s.get("type"), officer_dropdowns),
                    "condition": resolve(s.get("condition"), officer_dropdowns),
                    "supply": s.get("sustainable_yearly_supply"),
                    "consumption": s.get("yearly_consumption")
                }
                for s in o.get("water_source_available", [])
            ],
            "sewage_treatment": o.get("sewage_treatment"),
            "storage_capacity": o.get("capacity"),
            "houses_connected": o.get("number_of_houses")
        })

    return make_json_safe({
        "individual_water_profiles": household_profiles,
        "village_water_infrastructure": village_context,
        "supporting_metadata": {
            "water_dropdowns": dropdowns,
            "water_officer_dropdowns": officer_dropdowns
        }
    })

def water_llm_payload(data):
    return {
        "individual_profiles": data["individual_water_profiles"],
        "village_infrastructure": data["village_water_infrastructure"],
        "metadata": data["supporting_metadata"]
    }

def other_household_items_knowledge():
    dropdowns = {}
    for d in db.other_personal_household_item_dropdowns.find({}):
        dropdowns[str(d["_id"])] = d.get("name", {}).get("en", "")

    households = []

    for i in db.other_personal_household_items.find({"status": 1}):
        households.append({
            "item_category": dropdowns.get(str(i.get("item_category")), ""),
            "item_name": i.get("item_name"),
            "locally_produced": i.get("locally_produced"),
            "yearly_expense": i.get("yearly_expenditure"),
            "usage_purpose": i.get("usage_purpose")
        })

    return make_json_safe({
        "individual_profiles": households,
        "supporting_metadata": dropdowns
    })

def household_items_llm_payload(data):
    return {
        "individual_profiles": data["individual_profiles"],
        "metadata": data["supporting_metadata"]
    }


# ============================================================
def business_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional rural enterprise analyst.

USER QUESTION:
{question}

VILLAGE BUSINESS DATA (authoritative, complete):
{readable}


Write a professional analysis using EXACTLY this structure:

Sphere: Business & Commercial Activities
Indicator: Village Enterprise Structure & Performance

Village-level values
<paragraph>

Enterprise-level business profile (cleaned & human-readable)
<paragraphs>

Global / bioregional benchmark
<paragraph>

Calculation logic
<paragraph>

Gap analysis
<paragraph>

Recommendations
<paragraph>

Rules:
- No bullet points
- No lists
- No technical references
- Use natural policy language
"""
def agriculture_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI.

USER QUESTION:
{question}

VILLAGE AGRICULTURE & COMMUNITY DATA:
{readable}

Write the response EXACTLY in this order:

Sphere: Food, Agriculture & Community Systems
Indicator: Crop-wise Consumption & Community Infrastructure

Single-crop details
For EACH crop, write ONE paragraph covering:
crop name, consumption type, total quantity, market purchase,
neighbour exchange, self-grown quantity.

Village-level values
After ALL crops, write ONE paragraph summarising
overall consumption and dependency pattern.

Community officer systems
Write ONE paragraph covering recycling and broadband.

Calculation logic
Gap analysis
Recommendations

Rules:
- NO bullet points
- NO lists
- NO IDs
- Crop paragraphs MUST come first
"""
def forestry_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional forestry, commons, and natural resource systems analyst.

USER QUESTION:
{question}

VILLAGE FORESTRY DATA (authoritative, cleaned):
{readable}

Write the response EXACTLY in this order:

Sphere: Forestry, Commons & Natural Resource Systems
Indicator: Household Forest Use, Resource Extraction & Ecological Conditions

Household-level forestry profiles
Write ONE paragraph per household describing:
type of forest accessed, land under forest cover, presence of community forests,
timber harvested, other forest products collected, quantities, purposes of use,
unfulfilled forest-related needs, and urgency of those needs.

Village-level forestry values
After all households, write ONE paragraph summarising:
overall forest dependence, timber extraction intensity, reliance on non-timber
forest products, community forest presence, and pressure on forest resources.

Forest condition & governance context
Write ONE paragraph using forestry officer data covering:
types of forests accessible, total accessible area, forest condition,
presence of flora and fauna, forest fire incidents, wildlife conflict,
illegal activities, and administrative observations.

Global / bioregional benchmark
Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
- Fully human-readable
- Use formal policy and sustainability language
"""

def demographics_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional demographic and social systems analyst.

USER QUESTION:
{question}

VILLAGE DEMOGRAPHIC DATA (authoritative, cleaned):
{readable}

Write the response EXACTLY in this order:

Sphere: Population, Health, Social Structure & Human Capital
Indicator: Individual Demographic Attributes & Village Social Conditions

Individual demographic profiles
Write ONE paragraph per individual covering health, emotional well-being,
language ability, occupation, income security, education, skills,
hobbies, social participation, and aspirations.

Village-level values
After all individuals, write ONE paragraph summarising overall
health patterns, disability presence, economic inclusion,
education levels, skills distribution, and social well-being.

Settlement-level characteristics
Write ONE paragraph using village officer data, including housing types,
land context, population growth, local language, and living conditions.
Mention house images as visual documentation.

Civic priorities
Write ONE paragraph explaining voting issues influencing villagers.

Global / bioregional benchmark
Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
- Fully human-readable
"""
def energy_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional energy systems and infrastructure analyst.

USER QUESTION:
{question}

VILLAGE ENERGY DATA (authoritative, cleaned):
{readable}

Write the response EXACTLY in this order:

Sphere: Energy & Infrastructure Systems
Indicator: Household Energy Access, Consumption & Village Supply Structure

Individual energy profiles
Write ONE paragraph per household covering electricity access,
consumption levels, expenditure, stability, petrol usage,
fuel sources, microgrid presence, installation cost,
and energy sufficiency.

Village-level energy values
After all individuals, write ONE paragraph summarising
overall electricity access, fuel dependency, energy costs,
stability issues, and sufficiency patterns.

Energy infrastructure & supply systems
Write ONE paragraph using energy officer data covering
renewable availability, grid composition, microgrid share,
distribution methods, capacity, and distance to fuel pumps.

Global / bioregional benchmark
Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
- Fully human-readable
"""

def fisheries_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
Sphere: Fisheries, Feeds & Aquatic Food Systems
Indicator: Fish Production, Feed Dependency & Economic Outcomes

USER QUESTION:
{question}

VILLAGE DATA:
{readable}

Individual system profiles
Write ONE paragraph per fishery covering feed type, number of fishes,
production output, self-consumption, sales, wastage, income and costs.

Village-level values
Summarise overall fish production, feed usage, income generation
and dependency patterns.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No IDs
- Fully human-readable
"""
def housing_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
Sphere: Housing, Settlement & Built Environment
Indicator: Housing Stock, Amenities & Expansion Needs

USER QUESTION:
{question}

VILLAGE HOUSING DATA:
{readable}

Individual housing profiles
Write ONE paragraph per house covering structure, size, floors,
amenities, equipment, furnishing, renovation/expansion status,
and visual documentation.

Village-level values
Summarise housing adequacy, expansion demand,
land pressure, and infrastructure sufficiency.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No Object IDs
- Images referenced textually
"""
def hunting_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
Sphere: Wildlife Use, Hunting & Subsistence Systems
Indicator: Hunting Intensity, Meat Use & Economic Role

USER QUESTION:
{question}

VILLAGE DATA:
{readable}

Individual hunting records
Write ONE paragraph per animal covering hunting scale,
meat distribution, income, wastage and processing.

Village-level values
Summarise wildlife dependence, food security role,
economic contribution and sustainability risk.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No Object IDs
"""
def mobility_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
Sphere: Mobility, Transport & Connectivity
Indicator: Household Mobility Access & Village Transport Infrastructure

USER QUESTION:
{question}

VILLAGE MOBILITY DATA:
{readable}

Household mobility usage
Write ONE paragraph per household covering
vehicle types, distances travelled, travel purposes and frequency.

Household mobility needs
Write ONE paragraph summarising unmet mobility needs,
public transport access, and vehicle requirements.

Village-level transport infrastructure
Write ONE paragraph using officer data covering
road connectivity, conditions, safety issues and healthcare access.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
"""

def land_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional land-use and rural planning analyst.

USER QUESTION:
{question}

VILLAGE LAND DATA (authoritative, cleaned):
{readable}

Write the response EXACTLY in this order:

Sphere: Land, Housing & Spatial Use
Indicator: Household Land Ownership, Utilisation & Village Distribution

Individual landholding profiles
Write ONE paragraph per household describing:
land location, total area with unit, year of purchase,
current land use, purposes of utilisation, and area allocation.

Village-level land values
After all households, write ONE paragraph summarising:
total land availability, utilised vs unutilised land,
forest cover, grassland, fallow land, and private ownership.

Land requirements & unmet demand
Write ONE paragraph covering households requiring land,
required area, urgency, and stated purpose.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
- Convert all measurements into readable units
- Fully human-readable policy language
"""

def water_llm_prompt(question, data):
    readable = compact_summary(data)


    return f"""
You are OmniVillage AI, a professional water systems and public health analyst.

USER QUESTION:
{question}

VILLAGE WATER DATA (authoritative, cleaned):
{readable}

Write the response EXACTLY in this order:

Sphere: Water Resources & Sanitation Systems
Indicator: Household Water Access, Quality & Village Supply Infrastructure

Household-level water profiles
Write ONE paragraph per household covering:
water use type, source, quality, yearly consumption, expenditure,
presence of scarcity, severity, and water recycling practices.

Village-level water values
After all households, write ONE paragraph summarising:
overall water availability, dependence on sources,
extent of scarcity, and expenditure burden.

Water infrastructure & sanitation context
Write ONE paragraph using water officer data covering:
available water sources, their condition, sustainable supply,
sewage treatment, storage capacity, and household coverage.

Global / bioregional benchmark
Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
- Fully human-readable
"""
def other_household_items_llm_prompt(question, data):
    readable = compact_summary(data)
    return f"""
Sphere: Household Goods & Personal Items
Indicator: Household Item Ownership & Expenditure

USER QUESTION:
{question}

VILLAGE DATA:
{readable}

Individual household items
Write one paragraph per household.

Village-level values
Summarise expenditure and dependency.

Calculation logic
Gap analysis
Recommendations

Rules:
- No bullet points
- No lists
- No Object IDs
"""



# ============================================================
# ROUTES
# ============================================================
@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    try:
        body = request.get_json(silent=True) or {}
        question = body.get("message", "").strip()

        if not question:
            return jsonify({"reply": "Please ask a valid question."})

        domain = detect_domain(question)

        if domain == "agriculture":
            data = agriculture_knowledge()
            payload = agriculture_llm_payload(data)
            payload_text = json.dumps(make_json_safe(payload), indent=2)
            prompt = agriculture_llm_prompt(question, payload_text)

        elif domain == "energy":
            data = energy_knowledge()
            payload = energy_llm_payload(data)
            payload_text = json.dumps(make_json_safe(payload), indent=2)
            prompt = energy_llm_prompt(question, payload_text)

        elif domain == "demographics":
            data = demographics_knowledge()
            prompt = demographics_llm_prompt(question, make_json_safe(data))

        elif domain == "land":
            data = land_knowledge()
            prompt = land_llm_prompt(question, make_json_safe(data))

        elif domain == "hunting":
            data = hunting_knowledge()
            prompt = hunting_llm_prompt(question, make_json_safe(data))

        elif domain == "business":
            data = business_knowledge_cached()
            payload = business_llm_payload(data)
            payload_text = json.dumps(make_json_safe(payload), indent=2)
            prompt = business_llm_prompt(question, payload_text)

        elif domain == "fisheries":
            data = fisheries_knowledge()
            prompt = fisheries_llm_prompt(question, make_json_safe(data))

        elif domain == "forestry":
            data = forestry_knowledge()
            prompt = forestry_llm_prompt(question, make_json_safe(data))

        elif domain == "housing":
            data = housing_knowledge()
            prompt = housing_llm_prompt(question, make_json_safe(data))

        elif domain == "mobility":
            data = mobility_knowledge()
            prompt = mobility_llm_prompt(question, make_json_safe(data))

        elif domain == "water":
            data = water_knowledge()
            prompt = water_llm_prompt(question, make_json_safe(data))

        elif domain == "household_items":
            data = other_household_items_knowledge()
            prompt = other_household_items_llm_prompt(question, make_json_safe(data))

        else:
            return jsonify({
                "reply": (
                    "Supported domains:\n"
                    "• Business\n• Agriculture\n• Energy\n• Demographics\n"
                    "• Land\n• Hunting\n• Fisheries\n• Forestry\n"
                    "• Housing\n• Mobility\n• Water"
                )
            })

        response = llm.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.3,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )

        return jsonify({"reply": response.output_text})


    except RateLimitError:
        return jsonify({
            "reply": "System under load. Please retry shortly."
        }), 200

    except Exception as e:
        return jsonify({
            "reply": "Internal server error",
            "error": str(e)
        }), 500
