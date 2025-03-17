"""
Wecalcu – Advanced Calculatie App
----------------------------------
Deze applicatie berekent kosten en genereert offertes met behulp van traditionele 
berekeningen, scenarioanalyse en machine learning voorspellingen. De code is 
gedocumenteerd en uitgebreid met extra interactieve dashboards, realtime simulaties 
en een toekomstgerichte trendanalyse.

Auteur: Wessel
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime, timedelta
import logging
import copy
import io
import math
import json
import re
import hashlib
import requests
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from typing import Any, Dict, List
from sklearn.linear_model import LinearRegression  # Voor geavanceerde ML-functionaliteit

# ----- streamlit wide mode -----
st.set_page_config(layout="wide")

# ====================
#   FORCE REFRESH
# ====================
def force_refresh():
    """
    Forceer een harde pagina-herlaad via meta-refresh, 
    ter vervanging van st.experimental_rerun().
    """
    st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)


# ----- Centrale Configuratie -----
CONFIG = {
    "POPPLER_PATH": r"C:\Users\wesse\progs\poppler\Release-24.08.0-0\poppler-24.08.0\Library\bin",
    "TESSERACT_PATH": r"D:\Program Files (x86)\code\release-24-08-0-0\tesseract",
    "TRAINING_DATASET_PATH": "training_dataset.json",
    "DEFAULT_ACTIVITEIT_FACTORS": {
        "Werkvoorbereiding": 1.0,
        "Lasersnijden": 1.0,
        "Zetten": 1.0,
        "Walsen": 1.0,
        "Hechten": 1.0,
        "Lassen": 0.2,
        "Assembleren": 1.0,
        "Verpakken": 1.0,
        "Controle": 0.5
    },
    "DEFAULT_MATERIALS": {
        "Staal": {"price_per_kg": 0.95, "density": 7850},
        "Gegalvaniseerd": {"price_per_kg": 0.95, "density": 7850},
        "Aluminium": {"price_per_kg": 3.60, "density": 2700},
        "RVS304": {"price_per_kg": 3.05, "density": 7950},
        "RVS316": {"price_per_kg": 4.50, "density": 7950},
        "Corten": {"price_per_kg": 1.10, "density": 7850},
    },
    "PROFILE_TYPES": ["Koker", "Hoeklijn", "Strip", "Buis"],
    "TREATMENT_PRICES": {
        "Poedercoaten": {"price_per_unit": 25.50, "basis": "m²"},
        "Inwendig Spuiten vinyl": {"price_per_unit": 11.00, "basis": "m²"},
        "Inwendig Spuiten epoxy": {"price_per_unit": 15.00, "basis": "m²"},
        "Uitwendig Spuiten polycoat": {"price_per_unit": 18.00, "basis": "m²"},
        "Beitsen/Passiveren": {"price_per_unit": 1.30, "basis": "kg"},
        "Verzinken": {"price_per_unit": 1.60, "basis": "kg"},
        "Stralen": {"price_per_unit": 10.00, "basis": "m²"},
    },
    "TREATMENT_SHOW_COLOR": ["Poedercoaten", "Inwendig Spuiten vinyl", "Inwendig Spuiten epoxy", "Uitwendig Spuiten polycoat"]
}

# ----- Stel paden in -----
pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_PATH"]

# ----- Logging configuratie -----
logging.basicConfig(
    filename='wecalcu.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ----- Database Setup met SQLAlchemy -----
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///wecalcu.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----- Database Models -----
class Material(Base):
    __tablename__ = "materials"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    price_per_kg = Column(Float)
    density = Column(Integer)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    price = Column(Float)

class Klant(Base):
    __tablename__ = "klanten"
    id = Column(Integer, primary_key=True, index=True)
    naam = Column(String, unique=True, index=True)
    adres = Column(String)
    contact = Column(String)
    margin = Column(Float, default=20.0)

class DBPlate(Base):
    __tablename__ = "db_plates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    length = Column(Integer)  # mm
    width = Column(Integer)   # mm
    thickness = Column(Float) # mm

class DBProfile(Base):
    __tablename__ = "db_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)
    length = Column(Integer)  # mm
    buiten_diameter = Column(Integer, nullable=True)
    binnen_diameter = Column(Integer, nullable=True)
    breedte = Column(Integer, nullable=True)
    hoogte = Column(Integer, nullable=True)
    dikte = Column(Float, nullable=True)

class DBTreatment(Base):
    __tablename__ = "db_treatments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    basis = Column(String)
    price_per_unit = Column(Float)

class DBSpecialItem(Base):
    __tablename__ = "db_special_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    price = Column(Float)
    default_quantity = Column(Integer)

class DBIsolation(Base):
    __tablename__ = "db_isolatie"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    default_area = Column(Float)
    price_per_m2 = Column(Float)

class DBMesh(Base):
    __tablename__ = "db_mesh"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    default_area = Column(Float)
    price_per_m2 = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----- Helper functies voor veilige conversie -----
def safe_int(x, default=0):
    try:
        if pd.isnull(x):
            return default
        return int(x)
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        if pd.isnull(x):
            return default
        return float(x)
    except Exception:
        return default

# ----- Baseline conversiefactoren ophalen -----
def fetch_baseline_factors():
    url = "https://example.com/baseline_factors.json"  # Pas indien nodig aan
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            st.info("Baseline conversiefactoren succesvol opgehaald.")
            return data
        else:
            st.warning("Kon de baseline conversiefactoren niet ophalen (statuscode niet 200).")
            return CONFIG["DEFAULT_ACTIVITEIT_FACTORS"]
    except Exception as e:
        st.warning(f"Fout bij ophalen van baseline conversiefactoren: {e}. Gebruik default waarden.")
        return CONFIG["DEFAULT_ACTIVITEIT_FACTORS"]

# ----- Constante definities -----
RAL_COLORS_HEX = {
    # [... GROOT RAL-kleuroverzicht ongewijzigd ...]
}

DEFAULT_MATERIALS = CONFIG["DEFAULT_MATERIALS"]
PROFILE_TYPES = CONFIG["PROFILE_TYPES"]
TREATMENT_PRICES = CONFIG["TREATMENT_PRICES"]
TREATMENT_SHOW_COLOR = CONFIG["TREATMENT_SHOW_COLOR"]

# ----- Database Interaction Functions -----
def load_materials_from_db():
    db = SessionLocal()
    materials = {}
    try:
        for mat in db.query(Material).all():
            materials[mat.name] = {"price_per_kg": mat.price_per_kg, "density": mat.density}
    finally:
        db.close()
    return {**DEFAULT_MATERIALS, **materials}

def load_products_from_db():
    db = SessionLocal()
    products = {}
    try:
        for prod in db.query(Product).all():
            products[prod.name] = {"beschrijving": prod.description, "price": prod.price}
    finally:
        db.close()
    return products

def load_profiles_from_db():
    db = SessionLocal()
    profiles = []
    try:
        for p in db.query(DBProfile).all():
            profiles.append(p.name)
    finally:
        db.close()
    return profiles

def load_treatments_from_db():
    db = SessionLocal()
    treatments = []
    try:
        for t in db.query(DBTreatment).all():
            treatments.append(t.name)
    finally:
        db.close()
    return treatments

def load_special_items_from_db():
    db = SessionLocal()
    items = []
    try:
        for item in db.query(DBSpecialItem).all():
            items.append({
                "name": item.name,
                "description": item.description,
                "price": item.price,
                "default_quantity": item.default_quantity
            })
    finally:
        db.close()
    return items

def load_isolatie_from_db():
    db = SessionLocal()
    items = []
    try:
        for item in db.query(DBIsolation).all():
            items.append({
                "name": item.name,
                "default_area": item.default_area,
                "price_per_m2": item.price_per_m2
            })
    finally:
        db.close()
    return items

def load_mesh_from_db():
    db = SessionLocal()
    items = []
    try:
        for item in db.query(DBMesh).all():
            items.append({
                "name": item.name,
                "default_area": item.default_area,
                "price_per_m2": item.price_per_m2
            })
    finally:
        db.close()
    return items

def load_dbplates_from_db():
    db = SessionLocal()
    plates = []
    try:
        for p in db.query(DBPlate).all():
            plates.append({"name": p.name, "length": p.length, "width": p.width, "thickness": p.thickness})
    finally:
        db.close()
    return plates

def save_material_to_db(name, price_per_kg, density):
    db = SessionLocal()
    try:
        material = db.query(Material).filter(Material.name == name).first()
        if material:
            material.price_per_kg = price_per_kg
            material.density = density
        else:
            material = Material(name=name, price_per_kg=price_per_kg, density=density)
            db.add(material)
        db.commit()
    finally:
        db.close()

def save_product_to_db(name, description, price):
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.name == name).first()
        if product:
            product.description = description
            product.price = price
        else:
            product = Product(name=name, description=description, price=price)
            db.add(product)
        db.commit()
    finally:
        db.close()

def save_klant_to_db(naam, adres, contact, margin):
    db = SessionLocal()
    try:
        klant = db.query(Klant).filter(Klant.naam == naam).first()
        if klant:
            klant.adres = adres
            klant.contact = contact
            klant.margin = margin
        else:
            klant = Klant(naam=naam, adres=adres, contact=contact, margin=margin)
            db.add(klant)
        db.commit()
    finally:
        db.close()

def delete_klant_from_db(klant_id):
    db = SessionLocal()
    try:
        klant = db.query(Klant).filter(Klant.id == klant_id).first()
        if klant:
            db.delete(klant)
            db.commit()
    finally:
        db.close()

# ----- Session Initialisatie -----
def init_session():
    if "calc_data" not in st.session_state:
        st.session_state["calc_data"] = {
            "date": date.today().isoformat(),
            "geldigheidsduur": "30 dagen",
            "klant_naam": "",
            "klant_adres": "",
            "klant_contact": "",
            "comments": "",
            "marge_type": "Winstmarge (%)",
            "margin_percentage": 20.0,
            "storage_percentage": 10.0,
            "vat_percentage": 21.0,
            "uurloon": 0.0,
            "kilometers": 0.0,
            "kosten_per_kilometer": 0.0,
            "total_net_cost": 0.0,
            "storage_cost": 0.0,
            "total_internal_cost": 0.0,
            "total_revenue_excl_vat": 0.0,
            "vat_amount": 0.0,
            "total": 0.0,
            "total_profit": 0.0,
            "kostprijs": 0.0,
            "subtotal_vor_btw": 0.0,
            "total_weight_all": 0.0,
            "total_area": 0.0,
            "component_details": [],
            "original_calc_data": None,
            "scenarios": {},
            "global_margin": 0
        }
    if "calc_history" not in st.session_state:
        st.session_state["calc_history"] = []
    if "num_items" not in st.session_state:
        st.session_state["num_items"] = 1
    if "database_materials" not in st.session_state:
        st.session_state["database_materials"] = load_materials_from_db()
    if "database_products" not in st.session_state:
        st.session_state["database_products"] = load_products_from_db()
    if "db_profiles" not in st.session_state:
        st.session_state["db_profiles"] = load_profiles_from_db()
    if "db_treatments" not in st.session_state:
        st.session_state["db_treatments"] = load_treatments_from_db()
    if "db_special_items" not in st.session_state:
        st.session_state["db_special_items"] = load_special_items_from_db()
    if "db_isolatie" not in st.session_state:
        st.session_state["db_isolatie"] = load_isolatie_from_db()
    if "db_mesh" not in st.session_state:
        st.session_state["db_mesh"] = load_mesh_from_db()
    if "baseline_factors" not in st.session_state:
        st.session_state["baseline_factors"] = fetch_baseline_factors()
    if "budget_activiteiten" not in st.session_state:
        st.session_state["budget_activiteiten"] = st.session_state["baseline_factors"].copy()
        
init_session()

# ----- Extra Helper Functies -----
def get_data_editor():
    if hasattr(st, "data_editor"):
        return st.data_editor
    elif hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor
    else:
        st.warning("Upgrade naar een nieuwere versie van Streamlit!")
        return st.dataframe

data_editor = get_data_editor()

def select_ral_color(key, default="RAL 9001"):
    ral_keys = list(RAL_COLORS_HEX.keys())
    if default not in ral_keys:
        default = ral_keys[0]
    selected = st.selectbox("Kies een RAL kleur:", ral_keys, index=ral_keys.index(default), key=key)
    color_hex = RAL_COLORS_HEX[selected]
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <div style="width: 40px; height: 25px; background-color: {color_hex}; border: 1px solid #000; margin-right: 10px;"></div>
            <span>{selected}</span>
        </div>
        """, unsafe_allow_html=True)
    return color_hex

def extract_text_from_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text

def detect_lines(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    return lines

def extract_measurements(text, lines):
    try:
        length_matches = re.findall(r"lengte[:\s]+(\d+)", text, re.IGNORECASE)
        width_matches = re.findall(r"breedte[:\s]+(\d+)", text, re.IGNORECASE)
        length = float(length_matches[0]) if length_matches else 0
        width = float(width_matches[0]) if width_matches else 0
    except Exception as e:
        st.error(f"Fout bij het extraheren van afmetingen: {e}")
        length, width = 0, 0
    num_lines = len(lines) if lines is not None else 0
    return {"raw_text": text, "number_of_lines": num_lines, "length": length, "width": width}

def get_user_feedback(extracted_data):
    st.subheader("Controleer en pas de geëxtraheerde gegevens aan")
    corrected_data = {}
    for key, value in extracted_data.items():
        corrected_value = st.text_input(f"{key}", value=str(value), key=f"feedback_{key}")
        try:
            if key in ["length", "width", "number_of_lines"]:
                corrected_value = float(corrected_value)
        except Exception:
            pass
        corrected_data[key] = corrected_value
    if st.button("Opslaan correcties", key="save_feedback"):
        st.success("Correcties opgeslagen!")
        return corrected_data
    return extracted_data

def update_training_dataset(original_data, corrected_data, dataset_path=CONFIG["TRAINING_DATASET_PATH"]):
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        dataset = []
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": original_data,
        "corrected": corrected_data
    }
    dataset.append(record)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
    st.info("Training dataset bijgewerkt. Deze data kan later gebruikt worden voor hertraining.")

@st.cache_data(show_spinner=False)
def pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, poppler_path=CONFIG["POPPLER_PATH"])
        return images
    except Exception as e:
        st.error(f"Fout bij het converteren van PDF naar afbeeldingen: {e}")
        return []

# ----- Simulation Functions -----
def simulate_predictive_pricing(current_cost):
    np.random.seed(42)
    noise = np.random.normal(loc=0, scale=current_cost * 0.05)
    predicted_price = current_cost + noise
    return round(predicted_price, 2)

def simulate_cost_forecasting(initial_cost, days=30):
    np.random.seed(1)
    forecasts = []
    cost = initial_cost
    for day in range(days):
        change = np.random.normal(loc=0, scale=cost * 0.02)
        cost = max(cost + change, 0)
        forecasts.append(round(cost, 2))
    return forecasts

def detect_anomaly(cost_value, lower_bound, upper_bound):
    return cost_value < lower_bound or cost_value > upper_bound

def simulate_future_trends(initial_cost, months=120, monthly_growth=0.5, monthly_volatility=1.0):
    """
    Simuleer toekomstige kosten over een periode van 'months'
    met een gegeven maandelijks groeipercentage en volatiliteit.
    """
    np.random.seed(42)
    trend = []
    cost = initial_cost
    for month in range(months):
        growth_effect = cost * (monthly_growth / 100)
        noise = np.random.normal(0, cost * (monthly_volatility / 100))
        cost = max(cost + growth_effect + noise, 0)
        trend.append(round(cost, 2))
    return trend

def ml_predictive_analysis(hist_months, base_cost, growth, volatility):
    """
    Genereer historische kostendata over 'hist_months' en train een lineair regressiemodel.
    Voorspel vervolgens kosten voor de volgende 60 maanden.
    """
    np.random.seed(42)
    months = np.arange(1, int(hist_months)+1).reshape(-1, 1)
    costs = []
    cost = base_cost
    for _ in range(int(hist_months)):
        growth_effect = cost * (growth / 100)
        noise = np.random.normal(0, cost * (volatility / 100))
        cost = max(cost + growth_effect + noise, 0)
        costs.append(cost)
    model = LinearRegression()
    model.fit(months, costs)
    future_months = np.arange(int(hist_months)+1, int(hist_months)+61).reshape(-1, 1)
    predictions = model.predict(future_months)
    return months.flatten(), costs, future_months.flatten(), predictions, model

# ----- Calculation Functions -----
# ----- Calculation Functions -----
def perform_calculations():
    """Voer de kernberekeningen uit en werk de globale calculatiegegevens bij."""
    cd = st.session_state["calc_data"]
    comps = cd["component_details"]
    while len(comps) < st.session_state["num_items"]:
        comps.append({})
    all_materials = {**st.session_state["database_materials"], **DEFAULT_MATERIALS}
    cd["total_net_cost"] = 0.0
    cd["storage_cost"] = 0.0
    cd["total_internal_cost"] = 0.0
    cd["total_revenue_excl_vat"] = 0.0
    cd["vat_amount"] = 0.0
    cd["total"] = 0.0
    cd["total_profit"] = 0.0
    cd["subtotal_vor_btw"] = 0.0
    cd["total_weight_all"] = 0.0
    cd["total_area"] = 0.0

    for comp in comps[:st.session_state["num_items"]]:
        quantity = comp.get("quantity", 1)
        comp["budget_per_stuk"] = round(comp.get("uren_per_stuk", 0.0) * cd.get("uurloon", 0.0), 2)
        comp["total_budget"] = round(comp["budget_per_stuk"] * quantity, 2)

        comp["total_area"] = 0.0
        comp["total_weight"] = 0.0
        comp["material_cost"] = 0.0
        comp["profile_cost"] = 0.0
        comp["treatment_cost"] = 0.0
        comp["special_items_cost"] = 0.0
        comp["isolatie_cost"] = 0.0
        comp["gaas_cost"] = 0.0
        comp["product_cost"] = 0.0

        # Platen
        for plate in comp.get("plates", []):
            length = plate.get("length", 0)
            width = plate.get("width", 0)
            thickness = plate.get("thickness", 0)
            aantal = plate.get("aantal", 1)
            area = (length * width) / 1e6
            comp["total_area"] += area * aantal
            materiaal = comp.get("materiaal", "Staal")
            density = all_materials.get(materiaal, {"density": 7850})["density"]
            volume = area * (thickness / 1000)
            comp["total_weight"] += volume * density * aantal
            price_per_kg = all_materials.get(materiaal, {"price_per_kg": 0.0})["price_per_kg"]
            volume_plate = (length * width * thickness) / 1e9
            massa = volume_plate * density
            cost_plate = massa * price_per_kg
            if plate.get("apply_waste", False):
                waste_pct = plate.get("material_waste_percentage", 0.0)
                cost_plate *= (1 + waste_pct / 100.0)
            comp["material_cost"] += cost_plate * aantal
            if plate.get("lasersnijden", False):
                laser_cost_per_m2 = plate.get("laser_cost_per_m2", 0.0)
                comp["material_cost"] += laser_cost_per_m2 * area * aantal

        # Profielen
        for profile in comp.get("profiles", []):
            length = profile.get("length", 0)
            aantal = profile.get("aantal", 1)
            if profile.get("type") == "Buis":
                outer_d = profile.get("buiten_diameter", 0)
                inner_d = profile.get("binnen_diameter", 0)
                if outer_d > 0 and length > 0:
                    cross_section = (math.pi / 4) * (((outer_d / 1000) ** 2) - ((inner_d / 1000) ** 2))
                    volume = cross_section * (length / 1000)
                else:
                    volume = 0
            else:
                breedte = profile.get("breedte", 0)
                hoogte = profile.get("hoogte", 0)
                dikte = profile.get("dikte", 0)
                if breedte > 0 and hoogte > 0 and dikte > 0 and length > 0:
                    outer_area = (breedte / 1000) * (hoogte / 1000)
                    inner_area = ((breedte - 2 * dikte) / 1000) * ((hoogte - 2 * dikte) / 1000) if (breedte > 2 * dikte and hoogte > 2 * dikte) else 0
                    cross_section = outer_area - inner_area
                    volume = cross_section * (length / 1000)
                else:
                    volume = 0
            materiaal = comp.get("materiaal", "Staal")
            density = all_materials.get(materiaal, {"density": 7850})["density"]
            weight_profile = volume * density
            profile["weight"] = round(weight_profile, 2)
            price_per_kg = all_materials.get(materiaal, {"price_per_kg": 0.0})["price_per_kg"]
            cost_profile = weight_profile * price_per_kg
            profile["cost"] = round(cost_profile, 2)
            profile["cost_total"] = round(cost_profile * aantal, 2)
            comp["total_weight"] += weight_profile * aantal
            comp["profile_cost"] += profile["cost_total"]

        # Behandelingen
        for treatment in comp.get("treatments", []):
            if treatment.get("selected", "Handmatig invoeren") != "Handmatig invoeren":
                basis = TREATMENT_PRICES[treatment["selected"]]["basis"]
                price = TREATMENT_PRICES[treatment["selected"]]["price_per_unit"]
                treatment["basis"] = basis
                treatment["price_per_unit"] = price
            if treatment.get("selected") not in TREATMENT_SHOW_COLOR:
                treatment["kleur"] = ""
            if treatment.get("basis", "") == "m²":
                comp["treatment_cost"] += treatment.get("price_per_unit", 0.0) * comp["total_area"]
            elif treatment.get("basis", "") == "kg":
                comp["treatment_cost"] += treatment.get("price_per_unit", 0.0) * comp["total_weight"]

        # Speciale Items
        comp["special_items_cost"] = round(sum(si.get("price", 0.0) * si.get("quantity", 1) for si in comp.get("special_items", [])) * quantity, 2)

        # Isolatie
        comp["isolatie_cost"] = round(sum(iso.get("price_per_m2", 0.0) * iso.get("area_m2", 0.0) for iso in comp.get("isolatie", [])) * quantity, 2)

        # Gaas
        comp["gaas_cost"] = round(sum(g.get("price_per_m2", 0.0) * g.get("area_m2", 0.0) for g in comp.get("gaas", [])) * quantity, 2)

        # Producten
        comp["product_cost"] = round(sum(prod.get("price", 0.0) * prod.get("quantity", 1) for prod in comp.get("producten", [])) * quantity, 2)

        comp["total_area"] *= quantity
        comp["total_weight"] *= quantity
        cd["total_area"] += comp["total_area"]
        cd["total_weight_all"] += comp["total_weight"]

        comp["material_cost"] *= quantity

        comp_cost = (comp["material_cost"] + comp["profile_cost"] + comp["treatment_cost"] +
                     comp["special_items_cost"] + comp["isolatie_cost"] + comp["gaas_cost"] +
                     comp["product_cost"] + comp["total_budget"])
        comp["net_cost_component"] = round(comp_cost, 2)
        cd["total_net_cost"] += comp["net_cost_component"]

    logging.info("Berekeningen uitgevoerd. Totale Nettokost: %s", cd["total_net_cost"])

def calculate_storage_costs():
    cd = st.session_state["calc_data"]
    comps = cd["component_details"]
    storage_base_cost = 0.0
    for comp in comps[:st.session_state["num_items"]]:
        storage_base_cost += comp["material_cost"]
        for treatment in comp.get("treatments", []):
            if treatment.get("basis", "") == "m²":
                storage_base_cost += treatment.get("price_per_unit", 0.0) * comp["total_area"]
            elif treatment.get("basis", "") == "kg":
                storage_base_cost += treatment.get("price_per_unit", 0.0) * comp["total_weight"]
        for prod in comp.get("producten", []):
            if prod.get("include_storage", True):
                storage_base_cost += prod.get("price", 0.0) * prod.get("quantity", 1)
    cd["storage_cost"] = round(storage_base_cost * (cd.get("storage_percentage", 10.0) / 100.0), 2)

def finalize_calculations():
    cd = st.session_state["calc_data"]
    calculate_storage_costs()
    kilometer_cost = cd.get("kilometers", 0.0) * cd.get("kosten_per_kilometer", 0.0)
    cd["total_internal_cost"] = round(cd["total_net_cost"] + cd["storage_cost"] + kilometer_cost, 2)
    
    if cd["marge_type"] == "Winstmarge (%)":
        selling_price = cd["total_internal_cost"] * (1 + cd["margin_percentage"] / 100.0)
        if selling_price != 0:
            computed_global_margin = ((selling_price - cd["total_internal_cost"]) / selling_price) * 100
        else:
            computed_global_margin = 0
    else:  # Boekhoudelijke Marge (%)
        if cd["total_internal_cost"] != 0:
            selling_price = cd["total_internal_cost"] / (1 - cd["margin_percentage"] / 100.0)
        else:
            selling_price = 0
        computed_global_margin = cd["margin_percentage"]
    
    cd["total_revenue_excl_vat"] = round(selling_price, 2)
    cd["vat_amount"] = round(cd["total_revenue_excl_vat"] * (cd.get("vat_percentage", 21.0) / 100.0), 2)
    cd["total"] = round(cd["total_revenue_excl_vat"] + cd["vat_amount"], 2)
    cd["total_profit"] = round(cd["total_revenue_excl_vat"] - cd["total_internal_cost"], 2)
    cd["kostprijs"] = cd["total_internal_cost"]
    cd["subtotal_vor_btw"] = cd["total_internal_cost"]

    cd["global_margin"] = computed_global_margin

    logging.info("Finale berekeningen voltooid. Globale margin: %s", cd["global_margin"])

def export_to_excel():
    cd = st.session_state["calc_data"]
    comps = cd.get("component_details", [])
    if st.button("Exporteer naar Excel"):
        try:
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            general_data = {
                "Datum": [cd.get("date", "")],
                "Geldigheidsduur": [cd.get("geldigheidsduur", "")],
                "Klantnaam": [cd.get("klant_naam", "")],
                "Klantadres": [cd.get("klant_adres", "")],
                "Klant Contact": [cd.get("klant_contact", "")],
                "Opmerkingen": [cd.get("comments", "")],
                "Marge Type": [cd.get("marge_type", "Winstmarge (%)")],
                "Margin (%)": [cd.get("margin_percentage", 20.0)],
                "Opslag (%)": [cd.get("storage_percentage", 10.0)],
                "BTW (%)": [cd.get("vat_percentage", 21.0)],
                "Uurloon (EUR)": [cd.get("uurloon", 0.0)],
                "Kilometers": [cd.get("kilometers", 0.0)],
                "Kosten per Kilometer (EUR)": [cd.get("kosten_per_kilometer", 0.0)]
            }
            df_general = pd.DataFrame(general_data).fillna("")
            df_general.to_excel(writer, sheet_name='Algemene Gegevens', index=False)
            all_components = []
            for i, comp in enumerate(comps, start=1):
                comp_data = {
                    "Posnummer": i,
                    "Omschrijving": comp.get("omschrijving", ""),
                    "Materiaal": comp.get("materiaal", ""),
                    "Aantal": comp.get("quantity", 1),
                    "Uren per Stuk": comp.get("uren_per_stuk", 0.0),
                    "Budget per Stuk (EUR)": comp.get("budget_per_stuk", 0.0),
                    "Totale Budget (EUR)": comp.get("total_budget", 0.0),
                    "Nettokost (EUR)": comp.get("net_cost_component", 0.0),
                    "Materiaal Oppervlakte (m²)": comp.get("total_area", 0.0),
                    "Geschat Gewicht (kg)": comp.get("total_weight", 0.0)
                }
                if cd["total_internal_cost"] > 0:
                    margin_component = (cd["total_revenue_excl_vat"] - cd["total_internal_cost"]) * (comp.get("net_cost_component", 0.0) / cd["total_internal_cost"])
                else:
                    margin_component = 0
                selling_price_pos = round(comp.get("net_cost_component", 0.0) + margin_component, 2)
                selling_price_per_stuk = round(selling_price_pos / comp.get("quantity", 1), 2)
                comp_data["Verkoopprijs per Posnummer (€)"] = selling_price_pos
                comp_data["Verkoopprijs per Stuk (€)"] = selling_price_per_stuk
                all_components.append(comp_data)
            df_components = pd.DataFrame(all_components).fillna("")
            df_components.to_excel(writer, sheet_name='Componenten', index=False)
            if st.session_state["database_materials"]:
                mat_list = []
                for mat, data in st.session_state["database_materials"].items():
                    mat_list.append({
                        "Materiaal": mat,
                        "Prijs per kg (EUR)": data.get("price_per_kg", 0.0),
                        "Dichtheid (kg/m³)": data.get("density", 0)
                    })
                df_db_mat = pd.DataFrame(mat_list).fillna("")
                df_db_mat.to_excel(writer, sheet_name='Extra Materialen', index=False)
            writer.close()
            processed_data = output.getvalue()
            st.download_button(
                label="Download Excel",
                data=processed_data,
                file_name='wecalcu_calculatie.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            logging.info("Excel-bestand succesvol geëxporteerd.")
        except Exception as e:
            st.error(f"Fout bij het exporteren naar Excel: {e}")
            logging.exception("Fout bij het exporteren naar Excel:")
    if st.button("Export PDF (Coming Soon)"):
        st.info("PDF-export is in ontwikkeling.")

def import_from_excel():
    cd = st.session_state["calc_data"]
    uploaded_file = st.file_uploader("Importeer Calculatie Excel", type=["xlsx"], key="excel_import")
    if uploaded_file is not None:
        try:
            xl = pd.ExcelFile(uploaded_file)
            if 'Algemene Gegevens' in xl.sheet_names:
                df_general = xl.parse('Algemene Gegevens')
                required_columns = ["Datum", "Geldigheidsduur", "Klantnaam", "Klantadres", "Klant Contact", "Opmerkingen", "Winstmarge (%)", "Opslag (%)", "BTW (%)", "Uurloon (EUR)", "Kilometers", "Kosten per Kilometer (EUR)"]
                if all(col in df_general.columns for col in required_columns):
                    cd["date"] = df_general.at[0, "Datum"]
                    cd["geldigheidsduur"] = df_general.at[0, "Geldigheidsduur"]
                    cd["klant_naam"] = df_general.at[0, "Klantnaam"]
                    cd["klant_adres"] = df_general.at[0, "Klantadres"]
                    cd["klant_contact"] = df_general.at[0, "Klant Contact"]
                    cd["comments"] = df_general.at[0, "Opmerkingen"]
                    cd["margin_percentage"] = safe_float(df_general.at[0, "Winstmarge (%)"], 20.0)
                    cd["storage_percentage"] = safe_float(df_general.at[0, "Opslag (%)"], 10.0)
                    cd["vat_percentage"] = safe_float(df_general.at[0, "BTW (%)"], 21.0)
                    cd["uurloon"] = safe_float(df_general.at[0, "Uurloon (EUR)"], 0.0)
                    cd["kilometers"] = safe_float(df_general.at[0, "Kilometers"], 0.0)
                    cd["kosten_per_kilometer"] = safe_float(df_general.at[0, "Kosten per Kilometer (EUR)"], 0.0)
                else:
                    missing_cols = [col for col in required_columns if col not in df_general.columns]
                    st.warning(f"Ontbrekende kolommen in 'Algemene Gegevens': {', '.join(missing_cols)}")
                    logging.warning(f"Ontbrekende kolommen in 'Algemene Gegevens': {missing_cols}")
            else:
                st.warning("Sheet 'Algemene Gegevens' ontbreekt in het Excel-bestand.")
                logging.warning("Sheet 'Algemene Gegevens' ontbreekt in het Excel-bestand.")
            if 'Componenten' in xl.sheet_names:
                df_components = xl.parse('Componenten')
                cd["component_details"] = []
                for _, row in df_components.iterrows():
                    comp = {
                        "omschrijving": row.get("Omschrijving", ""),
                        "materiaal": row.get("Materiaal", ""),
                        "quantity": safe_int(row.get("Aantal", 1), 1),
                        "uren_per_stuk": safe_float(row.get("Uren per Stuk", 0.0), 0.0),
                        "budget_per_stuk": safe_float(row.get("Budget per Stuk (EUR)", 0.0), 0.0),
                        "total_budget": safe_float(row.get("Totale Budget (EUR)", 0.0), 0.0),
                        "net_cost_component": safe_float(row.get("Nettokost (EUR)", 0.0), 0.0),
                        "total_area": safe_float(row.get("Materiaal Oppervlakte (m²)", 0.0), 0.0),
                        "total_weight": safe_float(row.get("Geschat Gewicht (kg)", 0.0), 0.0),
                        "plates": [],
                        "profiles": [],
                        "treatments": [],
                        "special_items": [],
                        "isolatie": [],
                        "gaas": [],
                        "producten": [],
                        "comments": ""
                    }
                    cd["component_details"].append(comp)
            else:
                st.warning("Sheet 'Componenten' ontbreekt in het Excel-bestand.")
                logging.warning("Sheet 'Componenten' ontbreekt in het Excel-bestand.")
            if "Extra Materialen" in xl.sheet_names:
                df_materials = xl.parse("Extra Materialen")
                for _, row in df_materials.iterrows():
                    mat = row.get("Materiaal", "")
                    if mat and not pd.isna(mat):
                        st.session_state["database_materials"][mat] = {
                            "price_per_kg": safe_float(row.get("Prijs per kg (EUR)", 0.0), 0.0),
                            "density": safe_int(row.get("Dichtheid (kg/m³)", 0), 0)
                        }
            else:
                st.info("Sheet 'Extra Materialen' niet gevonden. Geen database-materialen geïmporteerd.")
            st.success("Excel-bestand succesvol geïmporteerd.")
            logging.info("Excel-bestand succesvol geïmporteerd.")
            perform_calculations()
            finalize_calculations()
        except Exception as e:
            st.error(f"Fout bij het importeren van Excel: {e}")
            logging.exception("Fout bij het importeren van Excel:")

def get_full_profile(profile_name):
    db = SessionLocal()
    profile = db.query(DBProfile).filter(DBProfile.name == profile_name).first()
    db.close()
    if profile:
        return {
            "type": profile.type,
            "length": profile.length,
            "buiten_diameter": profile.buiten_diameter,
            "binnen_diameter": profile.binnen_diameter,
            "breedte": profile.breedte,
            "hoogte": profile.hoogte,
            "dikte": profile.dikte,
            "aantal": 1
        }
    else:
        return {}

def get_full_treatment(treatment_name):
    db = SessionLocal()
    treatment = db.query(DBTreatment).filter(DBTreatment.name == treatment_name).first()
    db.close()
    if treatment:
        return {"selected": treatment.name, "basis": treatment.basis, "price_per_unit": treatment.price_per_unit}
    else:
        if treatment_name in TREATMENT_PRICES:
            return {"selected": treatment_name, "basis": TREATMENT_PRICES[treatment_name]["basis"],
                    "price_per_unit": TREATMENT_PRICES[treatment_name]["price_per_unit"]}
        else:
            return {"selected": treatment_name, "basis": "", "price_per_unit": 0.0}
        
# ----- Belangrijke functies -----
def save_calculation_snapshot():
    """Sla een snapshot van de huidige berekening op."""
    snapshot = copy.deepcopy(st.session_state["calc_data"])
    snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["calc_history"].append(snapshot)
    st.success(f"Berekening opgeslagen op {snapshot['timestamp']}.")

def compare_snapshots():
    """Vergelijk twee opgeslagen berekenings-snapshots."""
    if len(st.session_state["calc_history"]) < 2:
        st.info("Niet genoeg historische snapshots voor vergelijking.")
        return
    timestamps = [snap["timestamp"] for snap in st.session_state["calc_history"]]
    snap1 = st.selectbox("Selecteer eerste snapshot:", timestamps, key="snap1")
    snap2 = st.selectbox("Selecteer tweede snapshot:", timestamps, key="snap2")
    s1 = next((snap for snap in st.session_state["calc_history"] if snap["timestamp"] == snap1), None)
    s2 = next((snap for snap in st.session_state["calc_history"] if snap["timestamp"] == snap2), None)
    if s1 and s2:
        metrics = {
            "Totale Nettokost (€)": ("total_net_cost", s1.get("total_net_cost", 0.0), s2.get("total_net_cost", 0.0)),
            "Opslag (€)": ("storage_cost", s1.get("storage_cost", 0.0), s2.get("storage_cost", 0.0)),
            "Totale Interne Kosten (€)": ("total_internal_cost", s1.get("total_internal_cost", 0.0), s2.get("total_internal_cost", 0.0)),
            "Totale Revenu Excl BTW (€)": ("total_revenue_excl_vat", s1.get("total_revenue_excl_vat", 0.0), s2.get("total_revenue_excl_vat", 0.0)),
            "BTW (€)": ("vat_amount", s1.get("vat_amount", 0.0), s2.get("vat_amount", 0.0)),
            "Totale Kosten Incl BTW (€)": ("total", s1.get("total", 0.0), s2.get("total", 0.0)),
            "Totale Winst (€)": ("total_profit", s1.get("total_profit", 0.0), s2.get("total_profit", 0.0))
        }
        comp_table = []
        for label, (key, orig_val, new_val) in metrics.items():
            comp_table.append({
                "Metric": label,
                "Origineel": f"€ {round(orig_val, 2):,.2f}",
                "Aangepast": f"€ {round(new_val, 2):,.2f}",
                "Verschil": f"€ {round(new_val - orig_val, 2):,.2f}"
            })
        df_comp = pd.DataFrame(comp_table)
        st.dataframe(df_comp.style.highlight_max(axis=0))
    else:
        st.info("Kies twee geldige snapshots voor vergelijking.")

def simulate_calculation(sim_params):
    """Voer een simulatie uit door de berekeningsparameters tijdelijk aan te passen."""
    sim_cd = copy.deepcopy(st.session_state["calc_data"])
    sim_cd["margin_percentage"] *= sim_params.get("winstmarge", 1.0)
    sim_cd["storage_percentage"] *= sim_params.get("opslag", 1.0)
    sim_cd["uurloon"] *= sim_params.get("uurloon", 1.0)
    sim_cd["kilometers"] *= sim_params.get("kilometers", 1.0)
    sim_cd["kosten_per_kilometer"] *= sim_params.get("kosten_per_kilometer", 1.0)
    for comp in sim_cd["component_details"]:
        for prod in comp.get("producten", []):
            prod["price"] *= sim_params.get("product_multiplier", 1.0)
    original_cd = copy.deepcopy(st.session_state["calc_data"])
    st.session_state["calc_data"] = sim_cd
    perform_calculations()
    finalize_calculations()
    sim_result = copy.deepcopy(st.session_state["calc_data"])
    st.session_state["calc_data"] = original_cd
    return sim_result

def self_learning_pipeline(pdf_path):
    """Voer OCR en lijndetectie uit op een PDF en verzamel feedback."""
    st.info("PDF wordt verwerkt... Dit kan enkele ogenblikken duren.")
    with st.spinner("PDF converteren naar afbeeldingen..."):
        images = pdf_to_images(pdf_path)
    if not images:
        st.error("Geen afbeeldingen gevonden in de PDF.")
        return
    if len(images) > 1:
        page_num = st.selectbox("Selecteer de pagina die je wilt verwerken:",
                                 list(range(1, len(images)+1)),
                                 format_func=lambda x: f"Pagina {x}",
                                 help="Kies de pagina uit de PDF die je wilt analyseren.")
    else:
        page_num = 1
    image = images[page_num - 1]
    st.image(image, caption=f"Geüploade PDF-tekening - Pagina {page_num}", use_container_width=True)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    st.image(processed_image, caption="Voorbewerkt beeld voor OCR", use_container_width=True)
    raw_text = extract_text_from_image(image)
    st.text_area("OCR Uitvoer", value=raw_text, height=150,
                 help="Dit is de ruwe tekst die door de OCR is verkregen.")
    lines = detect_lines(image)
    st.write(f"Aantal gedetecteerde lijnen: {len(lines) if lines is not None else 0}")
    extracted_data = extract_measurements(raw_text, lines)
    st.write("Automatisch geëxtraheerde data:", extracted_data)
    corrected_data = get_user_feedback(extracted_data)
    if corrected_data != extracted_data:
        update_training_dataset(extracted_data, corrected_data)
    try:
        length = float(corrected_data.get("length", 0))
        width = float(corrected_data.get("width", 0))
        area = length * width
    except Exception as e:
        st.error(f"Fout bij berekening: {e}")
        area = 0
    st.success(f"Berekening voltooid: Oppervlakte = {area} mm²")
    st.info("Self-learning module: hertraining is ingepland (placeholder voor toekomstige ontwikkeling).")


# ----- Pagina's -----
def page_ml_predictive():
    st.title("ML Predictive Analysis en Toekomstige Trend Simulatie")
    st.markdown("""
Pas de parameters aan voor zowel Trend Simulatie als ML Predictive Analysis.
    """)
    with st.expander("ML & Trend Simulatie"):
        tabs = st.tabs(["Trend Simulatie", "ML Predictive Analysis"])
        with tabs[0]:
            st.subheader("Trend Simulatie Instellingen")
            initial_cost = st.number_input("Startkost (€):", value=1000.0, min_value=0.0, step=50.0)
            monthly_growth = st.slider("Maandelijkse groeipercentage (%)", 0.0, 5.0, 0.5, 0.1)
            monthly_volatility = st.slider("Maandelijkse volatiliteit (%)", 0.0, 10.0, 1.0, 0.1)
            simulation_months = st.number_input("Simulatieperiode (maanden):", value=120, min_value=1, step=1)
            if st.button("Simuleer Trend", key="simulate_trend_ml"):
                trend = simulate_future_trends(initial_cost, months=simulation_months,
                                               monthly_growth=monthly_growth, monthly_volatility=monthly_volatility)
                months_list = list(range(1, simulation_months+1))
                fig_trend = px.line(x=months_list, y=trend, labels={'x': 'Maanden', 'y': 'Gesimuleerde Kosten (€)'},
                                    title="Toekomstige Trend Simulatie")
                st.plotly_chart(fig_trend, use_container_width=True)
                st.write(f"Laatste gesimuleerde kost: € {trend[-1]:.2f}")
        with tabs[1]:
            st.subheader("ML Predictive Analysis Instellingen")
            hist_months = st.number_input("Aantal historische maanden:", value=60, min_value=1, step=1)
            ml_base_cost = st.number_input("Basis kost voor historische data (€):", value=1000.0, min_value=0.0, step=50.0)
            ml_growth = st.slider("Gemiddelde maandelijkse groei historische data (%)", 0.0, 5.0, 0.5, 0.1)
            ml_volatility = st.slider("Historische volatiliteit (%)", 0.0, 10.0, 1.0, 0.1)
            if st.button("Voer ML Predictive Analysis uit", key="ml_predict"):
                hist_months_arr, hist_costs, future_months, predictions, model = ml_predictive_analysis(
                    hist_months, ml_base_cost, ml_growth, ml_volatility)
                fig_ml = px.scatter(x=hist_months_arr, y=hist_costs, labels={'x': 'Maanden', 'y': 'Kosten (€)'},
                                    title="Historische Data en ML Voorspelling")
                fig_ml.add_scatter(x=future_months, y=predictions, mode='lines', name='ML Voorspelling')
                st.plotly_chart(fig_ml, use_container_width=True)
                st.write(f"Lineaire regressie parameters: Slope = {model.coef_[0]:.2f}, Intercept = {model.intercept_:.2f}")

def page_budget_uren():
    st.title("Advanced Budget Uren Calculator")
    st.markdown("Bereken een dynamisch urenbudget per afdeling en bekijk interactieve grafieken.")
    
    if "budget_table" not in st.session_state:
        st.session_state["budget_table"] = pd.DataFrame({
            "Afdeling": ["Werkvoorbereiding", "Montage", "Kwaliteitscontrole"],
            "Conversiefactor (uren per eenheid)": [1.0, 1.2, 0.5],
            "Eenheden": [0, 0, 0]
        })
    
    st.markdown("### Bewerk Afdelingen")
    edited_table = st.data_editor(st.session_state["budget_table"], num_rows="dynamic", key="budget_editor")
    st.session_state["budget_table"] = edited_table.copy()
    
    st.markdown("### Basis Berekening")
    edited_table["Budget Uren"] = edited_table["Conversiefactor (uren per eenheid)"] * edited_table["Eenheden"]
    totaal_uren = edited_table["Budget Uren"].sum()
    gemiddelde_uren = edited_table["Budget Uren"].mean()
    max_uren = edited_table["Budget Uren"].max()
    min_uren = edited_table["Budget Uren"].min()
    
    st.markdown(f"**Totaal Budget Uren:** {totaal_uren:.2f} uur")
    st.markdown(f"**Gemiddelde Uren per Afdeling:** {gemiddelde_uren:.2f} uur")
    st.markdown(f"**Max Uren in een Afdeling:** {max_uren:.2f} uur")
    st.markdown(f"**Min Uren in een Afdeling:** {min_uren:.2f} uur")
    
    st.dataframe(edited_table)
    
    st.markdown("---")
    st.markdown("### Scenario Analyse")
    st.markdown("Pas onderstaande slider aan om een scenario te simuleren waarbij de conversiefactoren worden aangepast. Hiermee kun je zien hoe het totale urenbudget verandert.")
    
    overall_multiplier = st.slider("Algemene Conversiefactor Multiplier", 0.5, 1.5, 1.0, 0.01, key="overall_multiplier")
    scenario_table = edited_table.copy()
    scenario_table["Scenario Conversiefactor"] = scenario_table["Conversiefactor (uren per eenheid)"] * overall_multiplier
    scenario_table["Scenario Budget Uren"] = scenario_table["Scenario Conversiefactor"] * scenario_table["Eenheden"]
    totaal_scenario = scenario_table["Scenario Budget Uren"].sum()
    
    st.markdown(f"**Totaal Budget Uren (Scenario):** {totaal_scenario:.2f} uur")
    
    comparison_df = scenario_table[["Afdeling", "Budget Uren", "Scenario Budget Uren"]].set_index("Afdeling")
    
    fig_bar = px.bar(
        comparison_df.reset_index(),
        x="Afdeling",
        y=["Budget Uren", "Scenario Budget Uren"],
        barmode="group",
        title="Budget Uren per Afdeling: Basis vs Scenario"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    fig_pie = px.pie(
        comparison_df.reset_index(),
        names="Afdeling",
        values="Scenario Budget Uren",
        title="Verdeling van Scenario Budget Uren per Afdeling"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("### Vergelijking Basis vs Scenario")
    comparison_df["Verschil (uur)"] = comparison_df["Scenario Budget Uren"] - comparison_df["Budget Uren"]
    st.dataframe(comparison_df.reset_index())

def page_calculatie():
    cd = st.session_state["calc_data"]
    readonly = st.session_state.get("role", "viewer") not in ["admin", "editor"]

    st.header("1. Excel Export en Import")
    export_to_excel()
    import_from_excel()
    st.markdown("---")

    st.header("2. Berekeningsinformatie Invoeren")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        cd["date"] = st.date_input("Datum", value=pd.to_datetime(cd.get("date", date.today())).date(), disabled=readonly, help="Selecteer de datum van de berekening.")
        cd["geldigheidsduur"] = st.text_input("Geldigheidsduur (bijv. '30 dagen')", value=cd.get("geldigheidsduur", "30 dagen"), disabled=readonly, help="Hoe lang is de berekening geldig?")
    with col_info2:
        cd["klant_naam"] = st.text_input("Klantnaam", value=cd.get("klant_naam", ""), disabled=readonly, help="Voer de naam van de klant in.")
        cd["klant_adres"] = st.text_area("Klantadres", value=cd.get("klant_adres", ""), disabled=readonly, help="Voer het adres van de klant in.")
        cd["klant_contact"] = st.text_input("Klant Contact", value=cd.get("klant_contact", ""), disabled=readonly, help="Voer het contactnummer of e-mailadres van de klant in.")
    st.text_area("Opmerkingen", value=cd.get("comments", ""), height=100, key="comments_field", disabled=readonly, help="Eventuele aanvullende opmerkingen.")
    st.markdown("---")

    st.header("3. Invoeren Marge, Opslag, Uurloon en Kilometers")
    default_marge_type = st.session_state["calc_data"].get("marge_type", "Winstmarge (%)")
    default_index = 0 if default_marge_type == "Winstmarge (%)" else 1

    selected_marge = st.radio("Selecteer margetype (intern)",
                            ["Winstmarge (%)", "Boekhoudelijke Marge (%)"],
                            index=default_index,
                            key="marge_type_radio",
                            disabled=readonly,
                            help="Bij 'Winstmarge (%)' wordt de verkoopprijs berekend als Kosten x (1 + marge/100). Bij 'Boekhoudelijke Marge (%)' wordt de verkoopprijs berekend als Kosten / (1 - marge/100).")
    st.session_state["calc_data"]["marge_type"] = selected_marge
    col_margin, col_opslag, col_uurloon = st.columns(3)
    with col_margin:
        cd["margin_percentage"] = st.number_input("Margin (%) (intern)", min_value=0.0, max_value=100.0,
                                                  value=float(cd.get("margin_percentage", 20.0)), step=1.0,
                                                  key="margin_input", format="%.2f", disabled=readonly,
                                                  help="Geef de marge in procenten op.")
    with col_opslag:
        cd["storage_percentage"] = st.number_input("Opslag (%)", min_value=0.0, max_value=100.0,
                                                    value=float(cd.get("storage_percentage", 10.0)), step=1.0,
                                                    key="opslag", format="%.2f", disabled=readonly,
                                                    help="Opslagpercentage toegepast op de nettokosten.")
    with col_uurloon:
        cd["uurloon"] = st.number_input("Uurloon (EUR):", min_value=0.0,
                                        value=float(cd.get("uurloon", 0.0)), step=0.10,
                                        key="uurloon", format="%.2f", disabled=readonly,
                                        help="Voer het uurloon in dat voor de berekeningen gebruikt wordt.")
    col_km, col_km_kosten = st.columns(2)
    with col_km:
        cd["kilometers"] = st.number_input("Aantal Kilometers:", min_value=0.0,
                                            value=float(cd.get("kilometers", 0.0)), step=1.0,
                                            key="kilometers", format="%.2f", disabled=readonly,
                                            help="Aantal gereden kilometers.")
    with col_km_kosten:
        cd["kosten_per_kilometer"] = st.number_input("Kosten per Kilometer (EUR):", min_value=0.0,
                                                      value=float(cd.get("kosten_per_kilometer", 0.0)), step=0.10,
                                                      key="kosten_per_kilometer", format="%.2f", disabled=readonly,
                                                      help="Kosten per kilometer.")
    st.markdown("---")

    st.header("4. Posnummers Invoeren")
    if not readonly and st.button("Sla Berekening op"):
        save_calculation_snapshot()
    num_items = st.number_input("Aantal Posnummers", min_value=1, max_value=100,
                                 value=st.session_state.get("num_items", 1), step=1,
                                 key="num_items_input", disabled=readonly)
    st.session_state["num_items"] = num_items
    comps = st.session_state["calc_data"]["component_details"]
    while len(comps) < num_items:
        comps.append({})
    if len(comps) > num_items:
        st.session_state["calc_data"]["component_details"] = comps[:num_items]
    all_materials = {**st.session_state["database_materials"], **DEFAULT_MATERIALS}
    materiaal_options = list(all_materials.keys())
    
    for i in range(num_items):
        comp = st.session_state["calc_data"]["component_details"][i]
        with st.expander(f"Posnummer {i+1} Invoer", expanded=False):
            st.markdown(f"### Algemene gegevens voor Posnummer {i+1}")
            col1, col2 = st.columns(2)
            with col1:
                comp["omschrijving"] = st.text_input("Omschrijving", value=comp.get("omschrijving", ""), key=f"omschrijving_{i}", disabled=readonly)
            with col2:
                selected_materiaal = st.selectbox("Materiaal", materiaal_options,
                                                  index=(materiaal_options.index(comp.get("materiaal", "Staal"))
                                                         if comp.get("materiaal", "Staal") in materiaal_options else 0),
                                                  key=f"materiaal_{i}", disabled=readonly)
                comp["materiaal"] = selected_materiaal
                price_per_kg = all_materials[selected_materiaal]["price_per_kg"]
                st.markdown(f"**Prijs per kg Materiaal:** € {price_per_kg:.2f}/kg")
            col3, col4 = st.columns(2)
            with col3:
                comp["quantity"] = st.number_input("Aantal", min_value=1, value=int(comp.get("quantity", 1)), step=1, key=f"quantity_{i}", disabled=readonly)
            with col4:
                comp["uren_per_stuk"] = st.number_input("Uren per Stuk", min_value=0.0, value=float(comp.get("uren_per_stuk", 0.0)), step=0.1, key=f"uren_per_stuk_{i}", disabled=readonly, help="Voer het aantal uren per stuk in.")
            comp["comments"] = st.text_area("Commentaar", value=comp.get("comments", ""), key=f"comp_comments_{i}", disabled=readonly)
            budget_per_stuk = comp.get("uren_per_stuk", 0.0) * cd.get("uurloon", 0.0)
            comp["budget_per_stuk"] = round(budget_per_stuk, 2)
            total_budget = round(budget_per_stuk * comp.get("quantity", 1), 2)
            comp["total_budget"] = total_budget
            st.markdown(f"**Budget per Stuk (€):** € {comp.get('budget_per_stuk', 0.0):,.2f}")
            st.markdown(f"**Totale Budget (€):** € {total_budget:,.2f}")

            tabs = st.tabs(["Platen", "Profielen", "Behandelingen", "Speciale Items", "Isolatie", "Gaas", "Producten"])
            
            with tabs[0]:
                st.markdown("#### Platen")
                input_mode = st.radio("Kies invoermethode voor Platen", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_platen_{i}")
                if input_mode == "Kies uit database":
                    dbplates = load_dbplates_from_db()
                    if dbplates:
                        plate_options = [p["name"] for p in dbplates]
                        selected_plate = st.selectbox("Selecteer een plaat", plate_options, key=f"db_plate_select_{i}")
                        plate_data = next((p for p in dbplates if p["name"] == selected_plate), None)
                        if plate_data:
                            if not comp.get("plates"):
                                comp.setdefault("plates", []).append({})
                            current_plate = comp["plates"][-1]
                            current_plate["length"] = plate_data["length"]
                            current_plate["width"] = plate_data["width"]
                            current_plate["thickness"] = plate_data["thickness"]
                            st.info("Plaatgegevens ingeladen vanuit de database.")
                    else:
                        st.info("Geen platen gevonden in de database.")
                else:
                    col_plate_btns = st.columns(2)
                    with col_plate_btns[0]:
                        if st.button("Voeg plaat toe", key=f"add_plate_{i}"):
                            comp.setdefault("plates", []).append({})
                    with col_plate_btns[1]:
                        if st.button("Verwijder laatste plaat", key=f"rem_plate_{i}"):
                            if comp.get("plates"):
                                comp["plates"].pop()
                                st.success("Laatst toegevoegde plaat verwijderd.")
                    for j, pl in enumerate(comp.get("plates", [])):
                        st.markdown(f"**Plaat {j+1}**")
                        cols_plate = st.columns(6)
                        with cols_plate[0]:
                            pl["length"] = int(st.number_input("Lengte (mm):", min_value=0, value=int(pl.get("length", 0)), step=1, key=f"plate_len_{i}_{j}", disabled=readonly))
                        with cols_plate[1]:
                            pl["width"] = int(st.number_input("Breedte (mm):", min_value=0, value=int(pl.get("width", 0)), step=1, key=f"plate_w_{i}_{j}", disabled=readonly))
                        with cols_plate[2]:
                            pl["thickness"] = st.number_input("Dikte (mm):", min_value=0.0, value=float(pl.get("thickness", 0.0)), step=0.1, key=f"plate_thick_{i}_{j}", format="%.2f", disabled=readonly)
                        with cols_plate[3]:
                            pl["aantal"] = st.number_input("Aantal:", min_value=1, value=int(pl.get("aantal", 1)), step=1, key=f"plate_amt_{i}_{j}", disabled=readonly)
                        with cols_plate[4]:
                            pl["lasersnijden"] = st.checkbox("Lasersnijden", value=pl.get("lasersnijden", False), key=f"plate_laser_{i}_{j}", disabled=readonly)
                            if pl["lasersnijden"]:
                                pl["laser_cost_per_m2"] = st.number_input("Kosten per m² (EUR):", min_value=0.0, value=float(pl.get("laser_cost_per_m2", 0.0)), step=0.10, key=f"plate_laser_cost_{i}_{j}", format="%.2f", disabled=readonly)
                        with cols_plate[5]:
                            pl["apply_waste"] = st.checkbox("Afval %", value=pl.get("apply_waste", False), key=f"plate_apply_waste_{i}_{j}", disabled=readonly)
                            if pl["apply_waste"]:
                                pl["material_waste_percentage"] = st.number_input("Afval (%):", min_value=0.0, max_value=100.0, value=float(pl.get("material_waste_percentage", 0.0)), step=1.0, key=f"plate_waste_pct_{i}_{j}", format="%.2f", disabled=readonly)
                        plate_area = (pl.get("length", 0) * pl.get("width", 0)) / 1e6 * pl.get("aantal", 1)
                        st.markdown(f"*Aantal: {pl.get('aantal', 1)} | Oppervlakte: {plate_area:.2f} m²*")
            
            with tabs[1]:
                st.markdown("#### Profielen")
                input_mode = st.radio("Kies invoermethode voor Profielen", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_profielen_{i}")
                if input_mode == "Kies uit database":
                    dbprofiles = st.session_state["db_profiles"]
                    if dbprofiles:
                        selected_profile = st.selectbox("Selecteer een profiel", dbprofiles, key=f"db_profile_select_{i}")
                        profile_data = get_full_profile(selected_profile)
                        if profile_data:
                            if not comp.get("profiles"):
                                comp.setdefault("profiles", []).append({})
                            comp["profiles"][-1] = profile_data
                            st.info("Profielgegevens ingeladen vanuit de database.")
                    else:
                        st.info("Geen profielen gevonden in de database.")
                else:
                    col_prof_btns = st.columns(2)
                    with col_prof_btns[0]:
                        if st.button("Voeg profiel toe", key=f"add_profile_{i}"):
                            comp.setdefault("profiles", []).append({})
                    with col_prof_btns[1]:
                        if st.button("Verwijder laatste profiel", key=f"rem_profile_{i}"):
                            if comp.get("profiles"):
                                comp["profiles"].pop()
                                st.success("Laatst toegevoegde profiel verwijderd.")
                    if not readonly:
                        extra_profile = st.selectbox("Selecteer profiel uit database", st.session_state["db_profiles"], key=f"db_profile_extra_{i}")
                        if extra_profile:
                            profile_data = get_full_profile(extra_profile)
                            if profile_data:
                                comp.setdefault("profiles", []).append(profile_data)
                    for j, prof in enumerate(comp.get("profiles", [])):
                        st.markdown(f"**Profiel {j+1}**")
                        cols = st.columns(4)
                        prof["type"] = cols[0].selectbox("Type", PROFILE_TYPES, index=(PROFILE_TYPES.index(prof.get("type", "Buis")) if prof.get("type", "Buis") in PROFILE_TYPES else 0), key=f"profile_type_{i}_{j}", disabled=readonly)
                        prof["length"] = int(cols[1].number_input("Lengte (mm):", min_value=0, value=int(prof.get("length", 0)), step=1, key=f"profile_length_{i}_{j}", disabled=readonly))
                        prof["aantal"] = int(cols[2].number_input("Aantal:", min_value=1, value=int(prof.get("aantal", 1)), step=1, key=f"profile_aantal_{i}_{j}", disabled=readonly))
                        if prof["type"] == "Buis":
                            cols_diams = st.columns(2)
                            prof["buiten_diameter"] = int(cols_diams[0].number_input("Buiten Diameter (mm):", min_value=0, value=int(prof.get("buiten_diameter", 0)), step=1, key=f"profile_outer_{i}_{j}", disabled=readonly))
                            prof["binnen_diameter"] = int(cols_diams[1].number_input("Binnen Diameter (mm):", min_value=0, value=int(prof.get("binnen_diameter", 0)), step=1, key=f"profile_inner_{i}_{j}", disabled=readonly))
                        else:
                            cols_dims = st.columns(3)
                            prof["breedte"] = int(cols_dims[0].number_input("Breedte (mm):", min_value=0, value=int(prof.get("breedte", 0)), step=1, key=f"profile_width_{i}_{j}", disabled=readonly))
                            prof["hoogte"] = int(cols_dims[1].number_input("Hoogte (mm):", min_value=0, value=int(prof.get("hoogte", 0)), step=1, key=f"profile_height_{i}_{j}", disabled=readonly))
                            prof["dikte"] = cols_dims[2].number_input("Dikte (mm):", min_value=0.0, value=float(prof.get("dikte", 0.0)), step=0.1, key=f"profile_thickness_{i}_{j}", format="%.2f", disabled=readonly)
            
            with tabs[2]:
                st.markdown("#### Behandelingen")
                input_mode = st.radio("Kies invoermethode voor Behandelingen", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_behandelingen_{i}")
                if input_mode == "Kies uit database":
                    db_treatments = st.session_state["db_treatments"]
                    if db_treatments:
                        selected_treatment = st.selectbox("Selecteer een behandeling", db_treatments, key=f"db_treatment_select_{i}")
                        treatment_data = get_full_treatment(selected_treatment)
                        if treatment_data:
                            if not comp.get("treatments"):
                                comp.setdefault("treatments", []).append({})
                            comp["treatments"][-1] = treatment_data
                            st.info("Behandeling ingeladen vanuit de database.")
                    else:
                        st.info("Geen behandelingen gevonden in de database.")
                else:
                    col_treat_btns = st.columns(2)
                    with col_treat_btns[0]:
                        if st.button("Voeg behandeling toe", key=f"add_treatment_{i}"):
                            comp.setdefault("treatments", []).append({})
                    with col_treat_btns[1]:
                        if st.button("Verwijder laatste behandeling", key=f"rem_treatment_{i}"):
                            if comp.get("treatments"):
                                comp["treatments"].pop()
                                st.success("Laatst toegevoegde behandeling verwijderd.")
                    if not readonly:
                        extra_treat = st.selectbox("Selecteer behandeling uit database", st.session_state["db_treatments"], key=f"db_treatment_extra_{i}")
                        if extra_treat:
                            treatment_data = get_full_treatment(extra_treat)
                            if treatment_data:
                                comp.setdefault("treatments", []).append(treatment_data)
                    for j, treat in enumerate(comp.get("treatments", [])):
                        st.markdown(f"**Behandeling {j+1}**")
                        cols = st.columns(4)
                        treatment_options = ["Handmatig invoeren"] + list(TREATMENT_PRICES.keys())
                        treat["selected"] = cols[0].selectbox("Selecteer", options=treatment_options, index=treatment_options.index(treat.get("selected", "Handmatig invoeren")), key=f"treatment_select_{i}_{j}", disabled=readonly)
                        if treat["selected"] == "Handmatig invoeren":
                            treat["basis"] = cols[1].selectbox("Basis", ["m²", "kg"], index=0 if treat.get("basis", "m²")=="m²" else 1, key=f"treatment_basis_{i}_{j}", disabled=readonly)
                            treat["price_per_unit"] = cols[2].number_input("Prijs per eenheid (EUR):", min_value=0.0, value=float(treat.get("price_per_unit", 0.0)), step=0.10, key=f"treatment_price_{i}_{j}", format="%.2f", disabled=readonly)
                        else:
                            basis = TREATMENT_PRICES[treat["selected"]]["basis"]
                            price = TREATMENT_PRICES[treat["selected"]]["price_per_unit"]
                            treat["basis"] = basis
                            treat["price_per_unit"] = price
                            cols[1].write(f"Basis: {basis}")
                            cols[2].write(f"Prijs: € {price:.2f}")
                        if treat["selected"] in TREATMENT_SHOW_COLOR:
                            treat["kleur"] = select_ral_color(key=f"treatment_color_{i}_{j}", default=treat.get("kleur", "RAL 9001"))
                        else:
                            treat["kleur"] = ""
            
            with tabs[3]:
                st.markdown("#### Speciale Items")
                input_mode = st.radio("Kies invoermethode voor Speciale Items", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_special_{i}")
                if input_mode == "Kies uit database":
                    db_special_items = load_special_items_from_db()
                    if db_special_items:
                        special_options = [item["name"] for item in db_special_items]
                        selected_special = st.selectbox("Selecteer een speciaal item", special_options, key=f"db_special_select_{i}")
                        special_data = next((item for item in db_special_items if item["name"] == selected_special), None)
                        if special_data:
                            if not comp.get("special_items"):
                                comp.setdefault("special_items", []).append({})
                            comp["special_items"][-1] = special_data
                            st.info("Special item ingeladen vanuit de database.")
                    else:
                        st.info("Geen speciale items gevonden in de database.")
                else:
                    col_spec_btns = st.columns(2)
                    with col_spec_btns[0]:
                        if st.button("Voeg speciale item toe", key=f"add_special_{i}"):
                            comp.setdefault("special_items", []).append({})
                    with col_spec_btns[1]:
                        if st.button("Verwijder laatste speciale item", key=f"rem_special_{i}"):
                            if comp.get("special_items"):
                                comp["special_items"].pop()
                                st.success("Laatst toegevoegde speciale item verwijderd.")
                    for j, spec in enumerate(comp.get("special_items", [])):
                        st.markdown(f"**Special Item {j+1}**")
                        cols = st.columns(3)
                        spec["description"] = cols[0].text_input("Omschrijving", value=spec.get("description", ""), key=f"special_desc_{i}_{j}", disabled=readonly)
                        spec["price"] = cols[1].number_input("Prijs (EUR):", min_value=0.0, value=float(spec.get("price", 0.0)), step=0.10, key=f"special_price_{i}_{j}", format="%.2f", disabled=readonly)
                        spec["quantity"] = int(cols[2].number_input("Aantal", min_value=1, value=int(spec.get("quantity", 1)), step=1, key=f"special_qty_{i}_{j}", disabled=readonly))
            
            with tabs[4]:
                st.markdown("#### Isolatie")
                input_mode = st.radio("Kies invoermethode voor Isolatie", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_isolatie_{i}")
                if input_mode == "Kies uit database":
                    db_isolatie = load_isolatie_from_db()
                    if db_isolatie:
                        isolatie_options = [item["name"] for item in db_isolatie]
                        selected_iso = st.selectbox("Selecteer isolatie", isolatie_options, key=f"db_isolatie_select_{i}")
                        iso_data = next((item for item in db_isolatie if item["name"] == selected_iso), None)
                        if iso_data:
                            if not comp.get("isolatie"):
                                comp.setdefault("isolatie", []).append({})
                            comp["isolatie"][-1] = iso_data
                            st.info("Isolatie ingeladen vanuit de database.")
                    else:
                        st.info("Geen isolatie-items gevonden in de database.")
                else:
                    col_iso_btns = st.columns(2)
                    with col_iso_btns[0]:
                        if st.button("Voeg isolatie toe", key=f"add_isolatie_{i}"):
                            comp.setdefault("isolatie", []).append({})
                    with col_iso_btns[1]:
                        if st.button("Verwijder laatste isolatie", key=f"rem_isolatie_{i}"):
                            if comp.get("isolatie"):
                                comp["isolatie"].pop()
                                st.success("Laatst toegevoegde isolatie verwijderd.")
                    for j, iso in enumerate(comp.get("isolatie", [])):
                        st.markdown(f"**Isolatie {j+1}**")
                        cols = st.columns(3)
                        iso["name"] = cols[0].text_input("Naam", value=iso.get("name", ""), key=f"isolatie_name_{i}_{j}", disabled=readonly)
                        iso["area_m2"] = cols[1].number_input("Oppervlakte (m²):", min_value=0.0, value=float(iso.get("area_m2", 0.0)), step=0.1, key=f"isolatie_area_{i}_{j}", format="%.2f", disabled=readonly)
                        iso["price_per_m2"] = cols[2].number_input("Prijs per m² (EUR):", min_value=0.0, value=float(iso.get("price_per_m2", 0.0)), step=0.10, key=f"isolatie_price_{i}_{j}", format="%.2f", disabled=readonly)
            
            with tabs[5]:
                st.markdown("#### Gaas")
                input_mode = st.radio("Kies invoermethode voor Gaas", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_gaas_{i}")
                if input_mode == "Kies uit database":
                    db_mesh = load_mesh_from_db()
                    if db_mesh:
                        mesh_options = [item["name"] for item in db_mesh]
                        selected_mesh = st.selectbox("Selecteer gaas", mesh_options, key=f"db_mesh_select_{i}")
                        mesh_data = next((item for item in db_mesh if item["name"] == selected_mesh), None)
                        if mesh_data:
                            if not comp.get("gaas"):
                                comp.setdefault("gaas", []).append({})
                            comp["gaas"][-1] = mesh_data
                            st.info("Gaas ingeladen vanuit de database.")
                    else:
                        st.info("Geen gaas-items gevonden in de database.")
                else:
                    col_gaas_btns = st.columns(2)
                    with col_gaas_btns[0]:
                        if st.button("Voeg gaas toe", key=f"add_gaas_{i}"):
                            comp.setdefault("gaas", []).append({})
                    with col_gaas_btns[1]:
                        if st.button("Verwijder laatste gaas", key=f"rem_gaas_{i}"):
                            if comp.get("gaas"):
                                comp["gaas"].pop()
                                st.success("Laatst toegevoegde gaas verwijderd.")
                    for j, gaas in enumerate(comp.get("gaas", [])):
                        st.markdown(f"**Gaas {j+1}**")
                        cols = st.columns(3)
                        gaas["name"] = cols[0].text_input("Naam", value=gaas.get("name", ""), key=f"gaas_name_{i}_{j}", disabled=readonly)
                        gaas["area_m2"] = cols[1].number_input("Oppervlakte (m²):", min_value=0.0, value=float(gaas.get("area_m2", 0.0)), step=0.1, key=f"gaas_area_{i}_{j}", format="%.2f", disabled=readonly)
                        gaas["price_per_m2"] = cols[2].number_input("Prijs per m² (EUR):", min_value=0.0, value=float(gaas.get("price_per_m2", 0.0)), step=0.10, key=f"gaas_price_{i}_{j}", format="%.2f", disabled=readonly)
            
            with tabs[6]:
                st.markdown("#### Producten")
                input_mode = st.radio("Kies invoermethode voor Producten", ["Handmatige invoer", "Kies uit database"], key=f"input_mode_producten_{i}")
                if input_mode == "Kies uit database":
                    db_products = list(st.session_state["database_products"].keys())
                    if db_products:
                        selected_product = st.selectbox("Selecteer een product", db_products, key=f"db_product_select_{i}")
                        product_data = st.session_state["database_products"].get(selected_product, {})
                        if product_data:
                            if not comp.get("producten"):
                                comp.setdefault("producten", []).append({})
                            comp["producten"][-1] = {
                                "name": selected_product,
                                "price": product_data.get("price", 0.0),
                                "quantity": 1,
                                "include_storage": True
                            }
                            st.info("Productgegevens ingeladen vanuit de database.")
                    else:
                        st.info("Geen producten gevonden in de database.")
                else:
                    col_prod_btns = st.columns(2)
                    with col_prod_btns[0]:
                        if st.button("Voeg product toe", key=f"add_product_{i}"):
                            comp.setdefault("producten", []).append({})
                    with col_prod_btns[1]:
                        if st.button("Verwijder laatste product", key=f"rem_product_{i}"):
                            if comp.get("producten"):
                                comp["producten"].pop()
                                st.success("Laatst toegevoegde product verwijderd.")
                            else:
                                st.warning("Er zijn geen producten om te verwijderen.")
                    for j, prod in enumerate(comp.get("producten", [])):
                        st.markdown(f"**Product {j+1}**")
                        cols = st.columns(3)
                        prod["name"] = cols[0].text_input("Productnaam", value=prod.get("name", ""), key=f"prod_name_{i}_{j}", disabled=readonly)
                        prod["price"] = cols[1].number_input("Prijs (EUR):", min_value=0.0, value=float(prod.get("price", 0.0)), step=0.10, key=f"prod_price_{i}_{j}", format="%.2f", disabled=readonly)
                        prod["quantity"] = int(cols[2].number_input("Aantal:", min_value=1, value=int(prod.get("quantity", 1)), step=1, key=f"prod_qty_{i}_{j}", disabled=readonly))
                        prod["include_storage"] = st.checkbox("Opnemen in opslag", value=prod.get("include_storage", True), key=f"prod_storage_{i}_{j}", disabled=readonly)
            st.markdown("")
    
    perform_calculations()
    finalize_calculations()
    with st.expander("Totale Kosten"):
        st.markdown("### Overzicht Kosten")
        cd = st.session_state["calc_data"]  # Ensure cd is defined here
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("Nettokostprijs", f"€ {cd.get('total_net_cost', 0.0):,.2f}")
        col2.metric("Opslag kosten", f"€ {cd.get('storage_cost', 0.0):,.2f}")
        col3.metric("Totale interne kosten", f"€ {cd.get('total_internal_cost', 0.0):,.2f}")
        col4.metric("Verkoopprijs (excl. BTW)", f"€ {cd.get('total_revenue_excl_vat', 0.0):,.2f}")
        col5.metric("BTW-bedrag", f"€ {cd.get('vat_amount', 0.0):,.2f}")
        col6.metric("Verkoopprijs (incl. BTW)", f"€ {cd.get('total', 0.0):,.2f}")
        col7.metric("Winst", f"€ {cd.get('total_profit', 0.0):,.2f}")
        col8.metric(f"Globale {cd['marge_type']}", f"{round(cd.get('global_margin', 0), 2)}%")
        
        st.markdown("________________________________________")
        st.markdown("#### Kosten Breakdown per Categorie")
        comps = cd.get("component_details", [])
        platen_total       = sum(comp.get("material_cost", 0.0) for comp in comps)
        profielen_total    = sum(comp.get("profile_cost", 0.0) for comp in comps)
        behandelingen_total = sum(comp.get("treatment_cost", 0.0) for comp in comps)
        speciale_items_total = sum(comp.get("special_items_cost", 0.0) for comp in comps)
        isolatie_total     = sum(comp.get("isolatie_cost", 0.0) for comp in comps)
        gaas_total         = sum(comp.get("gaas_cost", 0.0) for comp in comps)
        producten_total    = sum(comp.get("product_cost", 0.0) for comp in comps)
        
        st.markdown(f"**Platen:** € {platen_total:,.2f}")
        st.markdown(f"**Profielen:** € {profielen_total:,.2f}")
        st.markdown(f"**Behandelingen:** € {behandelingen_total:,.2f}")
        st.markdown(f"**Speciale Items:** € {speciale_items_total:,.2f}")
        st.markdown(f"**Isolatie:** € {isolatie_total:,.2f}")
        st.markdown(f"**Gaas:** € {gaas_total:,.2f}")
        st.markdown(f"**Producten:** € {producten_total:,.2f}")
    st.markdown("________________________________________")
    st.markdown("## Overzicht per Posnummer")
    pos_summary = []
    for idx, comp in enumerate(cd["component_details"][:st.session_state.get("num_items", 1)], start=1):
        base_net_cost = comp.get("net_cost_component", 0.0)
        # Bepaal het aandeel van deze component in de totale nettokost
        storage_share = (base_net_cost / cd["total_net_cost"]) * cd["storage_cost"] if cd["total_net_cost"] > 0 else 0
        adjusted_net_cost = round(base_net_cost + storage_share, 2)
        if cd.get("total_internal_cost", 0) > 0:
            margin_component = (cd["total_revenue_excl_vat"] - cd["total_internal_cost"]) * (adjusted_net_cost / cd["total_internal_cost"])
        else:
            margin_component = 0
        selling_price_pos = round(adjusted_net_cost + margin_component, 2)
        selling_price_per_stuk = round(selling_price_pos / comp.get("quantity", 1), 2)


        pos_summary.append({
            "Posnummer": idx,
            "Omschrijving": comp.get("omschrijving", ""),
            "Materiaal": comp.get("materiaal", ""),
            "Aantal": comp.get("quantity", 1),
            "Nettokost (€)": net_cost,
            "Verkoopprijs per Posnummer (€)": selling_price_pos,
            "Verkoopprijs per Stuk (€)": selling_price_per_stuk,
            "Boekhoudelijke Marge (%)": pos_margin_percentage
        })

    df_pos_summary = pd.DataFrame(pos_summary)
    st.dataframe(df_pos_summary.style.format({
        "Nettokost (€)": "{:.2f}",
        "Verkoopprijs per Posnummer (€)": "{:.2f}",
        "Verkoopprijs per Stuk (€)": "{:.2f}",
        "Boekhoudelijke Marge (%)": "{:.2f}"
    }), use_container_width=True)
    
    st.markdown("________________________________________")
    st.markdown("## Totale Offerte")
    st.markdown(f"**Totale Offerte (excl. BTW):** € {cd.get('total_revenue_excl_vat', 0.0):,.2f}")
    st.markdown(f"**BTW:** € {cd.get('vat_amount', 0.0):,.2f}")
    st.markdown(f"**Totale Offerte (incl. BTW):** € {cd.get('total', 0.0):,.2f}")
    st.markdown(f"**Globale {cd['marge_type']}:** {round(cd.get('global_margin', 0), 2)}%")

def page_offerte():
    cd = st.session_state["calc_data"]
    st.title("Offerte Overzicht")
    
    if not cd.get("component_details"):
        st.warning("Geen posnummers gevonden.")
        return

    st.markdown("### Algemene Offerte Informatie")
    st.markdown(f"**Datum:** {cd.get('date', '')}")
    st.markdown(f"**Geldigheidsduur:** {cd.get('geldigheidsduur', '')}")
    st.markdown(f"**Klantnaam:** {cd.get('klant_naam', '')}")
    st.markdown(f"**Klantadres:** {cd.get('klant_adres', '')}")
    st.markdown(f"**Klant Contact:** {cd.get('klant_contact', '')}")
    if cd.get("comments"):
        st.markdown(f"**Opmerkingen:** {cd.get('comments', '')}")
    st.markdown("---")
    st.markdown("### Overzicht per Posnummer")
    
    treatments_with_color = [
        "Poedercoaten", 
        "Inwendig Spuiten vinyl", 
        "Inwendig Spuiten epoxy", 
        "Uitwendig Spuiten polycoat"
    ]
    default_ral = "RAL 9024"
    
    rows = []
    for idx, comp in enumerate(cd.get("component_details", []), start=1):
    base_net_cost = comp.get("net_cost_component", 0.0)
    storage_share = (base_net_cost / cd["total_net_cost"]) * cd["storage_cost"] if cd["total_net_cost"] > 0 else 0
    adjusted_net_cost = round(base_net_cost + storage_share, 2)
    if cd.get("total_internal_cost", 0) > 0:
        margin_component = ((cd["total_revenue_excl_vat"] - cd["total_internal_cost"]) *
                            (adjusted_net_cost / cd["total_internal_cost"]))
    else:
        margin_component = 0
    selling_price_pos = round(adjusted_net_cost + margin_component, 2)
    selling_price_per_stuk = round(selling_price_pos / comp.get("quantity", 1), 2)
        
        if comp.get("treatments"):
            treatments_list = []
            for t in comp.get("treatments", []):
                treatment_name = t.get("selected", "Onbekend")
                if treatment_name in treatments_with_color:
                    ral_name = t.get("ral_name", "").strip()
                    if not ral_name:
                        kleur_val = t.get("kleur", "").strip()
                        for key, value in RAL_COLORS_HEX.items():
                            if value.lower() == kleur_val.lower():
                                ral_name = key
                                break
                    if not ral_name:
                        ral_name = default_ral
                    treatments_list.append(f"{treatment_name} (RAL: {ral_name})")
                else:
                    treatments_list.append(treatment_name)
            treatment_str = ", ".join(treatments_list)
        else:
            treatment_str = ""
        
        rows.append({
            "Posnummer": idx,
            "Omschrijving": comp.get("omschrijving", ""),
            "Aantal": comp.get("quantity", 1),
            "Prijs per Posnummer": f"€ {selling_price_pos:,.2f}",
            "Prijs per Stuk": f"€ {selling_price_per_stuk:,.2f}",
            "Nabehandelingen": treatment_str
        })
    
    df = pd.DataFrame(rows)
    st.table(df)
    
    st.markdown("### Totale Offerte")
    st.markdown(f"**Totale Offerte (excl. BTW):** € {cd.get('total_revenue_excl_vat', 0.0):,.2f}")
    st.markdown(f"**BTW (21%):** € {cd.get('vat_amount', 0.0):,.2f}")
    st.markdown(f"**Totale Offerte (incl. BTW):** € {cd.get('total', 0.0):,.2f}")

def compare_snapshots():
    if len(st.session_state["calc_history"]) < 2:
        st.info("Niet genoeg historische snapshots voor vergelijking.")
        return
    timestamps = [snap["timestamp"] for snap in st.session_state["calc_history"]]
    # Use multiselect to choose exactly two snapshots
    selected = st.multiselect("Selecteer twee snapshots voor vergelijking:", timestamps, default=timestamps[:2])
    if len(selected) != 2:
        st.info("Selecteer precies twee snapshots.")
        return
    s1 = next((snap for snap in st.session_state["calc_history"] if snap["timestamp"] == selected[0]), None)
    s2 = next((snap for snap in st.session_state["calc_history"] if snap["timestamp"] == selected[1]), None)
    if s1 and s2:
        metrics = {
            "Totale Nettokost (€)": ("total_net_cost", s1.get("total_net_cost", 0.0), s2.get("total_net_cost", 0.0)),
            "Opslag (€)": ("storage_cost", s1.get("storage_cost", 0.0), s2.get("storage_cost", 0.0)),
            "Totale Interne Kosten (€)": ("total_internal_cost", s1.get("total_internal_cost", 0.0), s2.get("total_internal_cost", 0.0)),
            "Totale Revenu Excl BTW (€)": ("total_revenue_excl_vat", s1.get("total_revenue_excl_vat", 0.0), s2.get("total_revenue_excl_vat", 0.0)),
            "BTW (€)": ("vat_amount", s1.get("vat_amount", 0.0), s2.get("vat_amount", 0.0)),
            "Totale Kosten Incl BTW (€)": ("total", s1.get("total", 0.0), s2.get("total", 0.0)),
            "Totale Winst (€)": ("total_profit", s1.get("total_profit", 0.0), s2.get("total_profit", 0.0))
        }
        comp_table = []
        for label, (key, orig_val, new_val) in metrics.items():
            comp_table.append({
                "Metric": label,
                "Origineel": f"€ {round(orig_val, 2):,.2f}",
                "Aangepast": f"€ {round(new_val, 2):,.2f}",
                "Verschil": f"€ {round(new_val - orig_val, 2):,.2f}"
            })
        df_comp = pd.DataFrame(comp_table)
        st.dataframe(df_comp.style.highlight_max(axis=0))
    else:
        st.info("Kies twee geldige snapshots voor vergelijking.")


def page_geavanceerde_rapportage():
    st.title("Geavanceerde Rapportage en Visualisatie")
    st.markdown("Bekijk hier een uitgebreid dashboard met meerdere analyses.")
    cd = st.session_state["calc_data"]  # Define cd at the start

    with st.expander("Dashboard & KPI Overzichten"):
        st.markdown("**KPI's:**")
        st.metric("Totale Nettokost", f"€ {cd.get('total_net_cost', 0.0):,.2f}")
        st.metric("Totale Winst", f"€ {cd.get('total_profit', 0.0):,.2f}")
        st.metric("Globale Marge (%)", f"{round(cd.get('global_margin', 0), 2)}%")
        st.markdown("Deze KPI's bieden een snel overzicht van de kernresultaten.")
        # Additional suggestion: add a line chart if historical trend data exists.
        if "historical_trend" in cd:
            st.markdown("#### Historische Trend")
            fig_line = px.line(x=cd["historical_trend"]["months"], y=cd["historical_trend"]["values"],
                               labels={'x': 'Maanden', 'y': 'Waarde'},
                               title="Historische Kosten Trend")
            st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("Kostenverdeling & Trends"):
        comps = cd.get("component_details", [])
        if not comps:
            st.warning("Geen posnummers gevonden. Voer eerst een calculatie uit.")
        else:
            cost_categories = {
                "Materiaalkosten": round(sum(comp.get("material_cost", 0.0) for comp in comps), 2),
                "Behandelingen": round(sum(comp.get("treatment_cost", 0.0) for comp in comps), 2),
                "Speciale Items": round(sum(comp.get("special_items_cost", 0.0) for comp in comps), 2),
                "Isolatie": round(sum(comp.get("isolatie_cost", 0.0) for comp in comps), 2),
                "Gaas": round(sum(comp.get("gaas_cost", 0.0) for comp in comps), 2),
                "Producten": round(sum(comp.get("product_cost", 0.0) for comp in comps), 2),
                "Kilometer Kosten": round(cd.get("kilometers", 0.0) * cd.get("kosten_per_kilometer", 0.0), 2)
            }
            fig_pie = px.pie(names=list(cost_categories.keys()), values=list(cost_categories.values()),
                             title='Kostenverdeling per Categorie')
            st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("Interactie & Filters"):
        df_costs = pd.DataFrame([{
            "Posnummer": idx + 1,
            "Nettokost": comp.get("net_cost_component", 0.0)
        } for idx, comp in enumerate(cd.get("component_details", []))])
        threshold = st.slider("Toon posnummers met Nettokost boven:", 0.0, max(df_costs["Nettokost"]) + 100, 0.0, 10.0)
        df_filtered = df_costs[df_costs["Nettokost"] >= threshold]
        st.dataframe(df_filtered)

    with st.expander("Historische Berekeningen & Snapshot Comparison"):
        if st.session_state["calc_history"]:
            history_df = pd.DataFrame([
                {"Timestamp": snap.get("timestamp", ""),
                 "Totale Nettokost (€)": snap.get("total_net_cost", 0.0),
                 "Totale Winst (€)": snap.get("total_profit", 0.0)}
                for snap in st.session_state["calc_history"]
            ])
            st.dataframe(history_df)
        else:
            st.info("Geen historische berekeningen beschikbaar.")
        # Call the modified snapshot comparison function using the new multiselect approach
        compare_snapshots()

def page_scenario_analyse():
    st.title("Scenario Analyse en What‑If Modellering")
    st.markdown(
        """
        Pas de onderstaande parameters aan om de impact op de berekeningen te simuleren.
        Wijzig de multipliers om te zien hoe uw kosten en winst veranderen in verschillende scenario's.
        """
    )
    cd = st.session_state["calc_data"]

    # Save original snapshot if not already saved.
    if cd.get("original_calc_data") is None:
        cd["original_calc_data"] = copy.deepcopy(cd)
        cd["original_calc_data"]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------
    # 1. Multi-Variable Simulation
    # -------------------------------
    st.subheader("1. Multi-Variable Simulatie")
    st.markdown(
        """
        Pas de volgende multipliers aan om het effect op de berekeningen te simuleren:
        
        - **Winstmarge Multiplier:** Past de winstmarge aan.
        - **Opslag Multiplier:** Wijzigt de opslagkosten.
        - **Uurloon Multiplier:** Verandert het uurloon.
        - **Kilometers Multiplier:** Verandert het aantal kilometers.
        - **Kosten per Kilometer Multiplier:** Wijzigt de kosten per kilometer.
        - **Product Prijs Multiplier:** Verandert de productprijzen.
        
        Klik op *Simuleer Multi-Variable* om de resultaten te vergelijken met de originele berekening.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        winstmarge_mult = st.slider("Winstmarge Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_winstmarge")
        opslag_mult = st.slider("Opslag Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_opslag")
        uurloon_mult = st.slider("Uurloon Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_uurloon")
    with col2:
        kilometers_mult = st.slider("Kilometers Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_kilometers")
        kosten_km_mult = st.slider("Kosten per Kilometer Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_kmk")
        product_mult = st.slider("Product Prijs Multiplier", 0.9, 1.1, 1.0, 0.01, key="sim_product")
    
    if st.button("Simuleer Multi-Variable"):
        sim_params = {
            "winstmarge": winstmarge_mult,
            "opslag": opslag_mult,
            "uurloon": uurloon_mult,
            "kilometers": kilometers_mult,
            "kosten_per_kilometer": kosten_km_mult,
            "product_multiplier": product_mult
        }
        sim_result = simulate_calculation(sim_params)
        st.markdown("#### Vergelijking: Origineel vs Simulatie")
        compare_table = []
        keys = ["total_net_cost", "storage_cost", "total_internal_cost", "total_revenue_excl_vat", "vat_amount", "total", "total_profit"]
        labels = {
            "total_net_cost": "Totale Nettokost (€)",
            "storage_cost": "Opslag (€)",
            "total_internal_cost": "Totale Interne Kosten (€)",
            "total_revenue_excl_vat": "Totale Revenu Excl BTW (€)",
            "vat_amount": "BTW (€)",
            "total": "Totale Kosten Incl BTW (€)",
            "total_profit": "Totale Winst (€)"
        }
        for k in keys:
            orig_val = cd.get(k, 0.0)
            sim_val = sim_result.get(k, 0.0)
            compare_table.append({
                "Metric": labels.get(k, k),
                "Origineel": f"€ {round(orig_val, 2):,.2f}",
                "Simulatie": f"€ {round(sim_val, 2):,.2f}",
                "Verschil": f"€ {round(sim_val - orig_val, 2):,.2f}"
            })
        df_compare = pd.DataFrame(compare_table)
        st.dataframe(df_compare.style.highlight_max(axis=0))

    st.markdown("---")
    
    # -------------------------------
    # 2. Monte Carlo Simulation
    # -------------------------------
    st.subheader("2. Monte Carlo Simulatie")
    st.markdown(
        """
        Voer een Monte Carlo simulatie uit om de kansverdeling van de totale winst te onderzoeken.
        Geef hieronder het aantal simulaties op en klik op *Voer Monte Carlo Simulatie uit*.
        """
    )
    mc_runs = st.number_input("Aantal simulaties", min_value=10, max_value=1000, value=100, step=10, key="mc_runs")
    if st.button("Voer Monte Carlo Simulatie uit"):
        results = []
        for _ in range(int(mc_runs)):
            rand_params = {
                "winstmarge": winstmarge_mult * np.random.normal(1, 0.01),
                "opslag": opslag_mult * np.random.normal(1, 0.01),
                "uurloon": uurloon_mult * np.random.normal(1, 0.01),
                "kilometers": kilometers_mult * np.random.normal(1, 0.01),
                "kosten_per_kilometer": kosten_km_mult * np.random.normal(1, 0.01),
                "product_multiplier": product_mult * np.random.normal(1, 0.01)
            }
            sim_data = simulate_calculation(rand_params)
            results.append(sim_data.get("total_profit", 0.0))
        fig_hist = px.histogram(x=results, nbins=30, labels={'x': 'Totale Winst (€)', 'y': 'Frequentie'},
                                title="Monte Carlo Simulatie: Verdeling Totale Winst")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown(f"**Gemiddelde Totale Winst:** € {np.mean(results):,.2f}")

    st.markdown("---")
    
    # -------------------------------
    # 3. Tornado Diagram (Sensitivity Analysis)
    # -------------------------------
    st.subheader("3. Tornado Diagram (Sensitiviteitsanalyse)")
    st.markdown(
        """
        Bekijk welke parameters de grootste invloed hebben op de totale winst.
        De onderstaande diagram toont de verandering in winst bij een ±5% variatie in elk van de multipliers.
        """
    )
    base_profit = cd.get("total_profit", 0.0)
    sensitivity = {}
    variation = 0.05  # ±5% variation
    multipliers = {
        "winstmarge": winstmarge_mult,
        "opslag": opslag_mult,
        "uurloon": uurloon_mult,
        "kilometers": kilometers_mult,
        "kosten_per_kilometer": kosten_km_mult,
        "product_multiplier": product_mult
    }
    for key, base_val in multipliers.items():
        params_up = {k: multipliers[k] for k in multipliers}
        params_down = {k: multipliers[k] for k in multipliers}
        params_up[key] = base_val * (1 + variation)
        params_down[key] = base_val * (1 - variation)
        profit_up = simulate_calculation(params_up).get("total_profit", 0.0)
        profit_down = simulate_calculation(params_down).get("total_profit", 0.0)
        sensitivity[key] = max(abs(profit_up - base_profit), abs(profit_down - base_profit))
    tornado_data = pd.DataFrame({
        "Parameter": list(sensitivity.keys()),
        "Impact op Winst (€)": list(sensitivity.values())
    }).sort_values(by="Impact op Winst (€)", ascending=True)
    fig_tornado = px.bar(tornado_data, x="Impact op Winst (€)", y="Parameter", orientation='h',
                         title="Tornado Diagram: Impact per Parameter",
                         labels={"Impact op Winst (€)": "Verandering in Totale Winst (€)"})
    st.plotly_chart(fig_tornado, use_container_width=True)


def page_pdf_calculatie():
    st.title("Calculatie vanaf PDF-tekening")
    st.markdown(
        """
        Upload hier een PDF-tekening. Deze module converteert de PDF naar afbeeldingen, voert OCR en lijndetectie uit,
        en laat je de geëxtraheerde afmetingen controleren. Feedback wordt opgeslagen voor toekomstige hertraining.
        """
    )
    
    pdf_file = st.file_uploader("Upload een PDF-tekening", type=["pdf"], key="pdf_upload")
    
    if pdf_file is not None:
        try:
            file_bytes = pdf_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            
            # Process only if a new file is uploaded
            if st.session_state.get("last_pdf_hash", "") != file_hash:
                st.session_state["last_pdf_hash"] = file_hash
                # Save the file locally
                with open("uploaded.pdf", "wb") as f:
                    f.write(file_bytes)
                st.info("Nieuwe PDF ontvangen. Data wordt geanalyseerd...")
                
                # Convert PDF to images and display a preview of the first page
                images = pdf_to_images("uploaded.pdf")
                if images:
                    st.image(images[0], caption="Voorbeeld: Pagina 1", use_container_width=True)
                else:
                    st.warning("Kon geen afbeeldingen genereren uit de PDF.")
                
                # Run the self-learning pipeline (OCR and measurement extraction)
                self_learning_pipeline("uploaded.pdf")
                st.success("PDF verwerking voltooid!")
            else:
                st.info("Deze PDF is al verwerkt. Upload een andere PDF om te updaten.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden tijdens het verwerken van de PDF: {e}")
            logging.exception("Fout bij het verwerken van de PDF:")

# =====================
#   page_database()
# =====================
def page_database():
    st.title("Product Database - Advanced Management")
    st.markdown("Hier kun je database-items toevoegen, bewerken en bekijken.")

    tabs = st.tabs([
        "Materialen", "Producten", "Klanten", "Platen", 
        "Profielen", "Behandelingen", "Speciale Items", 
        "Isolatie", "Gaas"
    ])

    # [ Tab 0: Materialen ]
    with tabs[0]:
        st.header("Materialen")
        db = SessionLocal()
        materialen = db.query(Material).all()
        db.close()
        if materialen:
            st.subheader("Bestaande Materialen")
            for mat in materialen:
                st.write(f"**Naam:** {mat.name} | **Prijs per kg:** €{mat.price_per_kg:.2f} | **Dichtheid:** {mat.density} kg/m³")
                # Hier geen admin-check, kun je zelf toevoegen als nodig
        else:
            st.info("Geen materialen gevonden.")

        st.subheader("Voeg nieuw materiaal toe")
        with st.form(key="form_materialen"):
            col1, col2 = st.columns(2)
            with col1:
                nieuw_mat = st.text_input("Naam")
            with col2:
                prijs_per_kg = st.number_input("Prijs per kg (EUR)", min_value=0.0, value=0.95, step=0.1, format="%.2f")
            density = st.number_input("Dichtheid (kg/m³)", min_value=0, value=7850, step=1)
            submit_material = st.form_submit_button("Materiaal toevoegen")
            if submit_material:
                if nieuw_mat.strip() == "":
                    st.error("Voer een geldige naam in.")
                else:
                    save_material_to_db(nieuw_mat.strip(), prijs_per_kg, density)
                    st.success(f"Materiaal '{nieuw_mat}' toegevoegd!")
                    # VERVANG experimental_rerun
                    force_refresh()  # <--- GEBRUIK NU force_refresh()

    # [ Tab 1: Producten ]
    with tabs[1]:
        st.header("Producten")
        db = SessionLocal()
        producten = db.query(Product).all()
        db.close()
        if producten:
            st.subheader("Bestaande Producten")
            for prod in producten:
                st.write(f"**Naam:** {prod.name} | **Beschrijving:** {prod.description} | **Prijs:** €{prod.price:.2f}")
        else:
            st.info("Geen producten gevonden.")

        st.subheader("Voeg nieuw product toe")
        with st.form(key="form_producten"):
            naam_prod = st.text_input("Productnaam")
            beschrijving = st.text_area("Beschrijving")
            prijs = st.number_input("Prijs (EUR)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            submit_product = st.form_submit_button("Product toevoegen")
            if submit_product:
                if naam_prod.strip() == "":
                    st.error("Voer een geldige productnaam in.")
                else:
                    save_product_to_db(naam_prod.strip(), beschrijving, prijs)
                    st.success(f"Product '{naam_prod}' toegevoegd!")
                    force_refresh()  # <--- GEBRUIK force_refresh()

    # [ Tab 2: Klanten ]
    with tabs[2]:
        st.header("Klanten")
        klanten_placeholder = st.empty()

        def render_klanten():
            with klanten_placeholder.container():
                db = SessionLocal()
                klanten = db.query(Klant).all()
                db.close()
                if klanten:
                    st.subheader("Bestaande Klanten")
                    for klant in klanten:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**Naam:** {klant.naam} | **Adres:** {klant.adres} | **Contact:** {klant.contact} | **Standaard Marge:** {klant.margin:.2f}%")
                        with col2:
                            if st.button("Verwijder", key=f"delete_{klant.id}"):
                                delete_klant_from_db(klant.id)
                                st.success(f"Klant '{klant.naam}' verwijderd!")
                                klanten_placeholder.empty()
                                render_klanten()
                else:
                    st.info("Geen klanten gevonden.")

        render_klanten()
        st.subheader("Voeg nieuwe klant toe")
        with st.form(key="form_klanten"):
            naam_klant = st.text_input("Klantnaam")
            adres = st.text_area("Adres")
            contact = st.text_input("Contactgegevens")
            marge = st.number_input("Standaard Marge (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.2f")
            submit_klant = st.form_submit_button("Klant toevoegen")
            if submit_klant:
                if naam_klant.strip() == "":
                    st.error("Voer een geldige klantnaam in.")
                else:
                    save_klant_to_db(naam_klant.strip(), adres, contact, marge)
                    st.success(f"Klant '{naam_klant}' toegevoegd!")
                    klanten_placeholder.empty()
                    render_klanten()

    # [ Tab 3: Platen ]
    with tabs[3]:
        st.header("Platen")
        db = SessionLocal()
        dbplates = db.query(DBPlate).all()
        db.close()
        if dbplates:
            st.subheader("Bestaande Platen")
            for plate in dbplates:
                st.write(f"**Naam:** {plate.name} | **Lengte:** {plate.length} mm | **Breedte:** {plate.width} mm | **Dikte:** {plate.thickness} mm")
        else:
            st.info("Geen platen gevonden.")

        st.subheader("Voeg nieuwe plaat toe")
        with st.form(key="form_platen"):
            plate_name = st.text_input("Naam van de plaat")
            lengte = st.number_input("Lengte (mm)", min_value=0, value=1000, step=1)
            breedte = st.number_input("Breedte (mm)", min_value=0, value=500, step=1)
            dikte = st.number_input("Dikte (mm)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
            submit_plate = st.form_submit_button("Plaat toevoegen")
            if submit_plate:
                if plate_name.strip() == "":
                    st.error("Voer een geldige naam in voor de plaat.")
                else:
                    new_plate = DBPlate(name=plate_name.strip(), length=lengte, width=breedte, thickness=dikte)
                    db = SessionLocal()
                    db.add(new_plate)
                    db.commit()
                    db.close()
                    st.success(f"Plaat '{plate_name}' toegevoegd!")
                    force_refresh()  # <---

    # [ Tab 4: Profielen ]
    with tabs[4]:
        st.header("Profielen")
        db = SessionLocal()
        profiles = db.query(DBProfile).all()
        db.close()
        if profiles:
            st.subheader("Bestaande Profielen")
            for prof in profiles:
                st.write(f"**Naam:** {prof.name} | **Type:** {prof.type} | **Lengte:** {prof.length} mm")
        else:
            st.info("Geen profielen gevonden.")

        st.subheader("Voeg nieuw profiel toe")
        with st.form(key="form_profielen"):
            naam_prof = st.text_input("Naam van het profiel")
            type_prof = st.selectbox("Type", PROFILE_TYPES)
            lengte_prof = st.number_input("Lengte (mm)", min_value=0, value=1000, step=1)
            buiten_diameter = st.number_input("Buiten Diameter (mm) (optioneel)", min_value=0, value=0, step=1)
            binnen_diameter = st.number_input("Binnen Diameter (mm) (optioneel)", min_value=0, value=0, step=1)
            breedte = st.number_input("Breedte (mm) (optioneel)", min_value=0, value=0, step=1)
            hoogte = st.number_input("Hoogte (mm) (optioneel)", min_value=0, value=0, step=1)
            dikte = st.number_input("Dikte (mm) (optioneel)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            submit_prof = st.form_submit_button("Profiel toevoegen")
            if submit_prof:
                if naam_prof.strip() == "":
                    st.error("Voer een geldige naam in voor het profiel.")
                else:
                    new_prof = DBProfile(
                        name=naam_prof.strip(),
                        type=type_prof,
                        length=lengte_prof,
                        buiten_diameter=buiten_diameter if buiten_diameter > 0 else None,
                        binnen_diameter=binnen_diameter if binnen_diameter > 0 else None,
                        breedte=breedte if breedte > 0 else None,
                        hoogte=hoogte if hoogte > 0 else None,
                        dikte=dikte if dikte > 0 else None
                    )
                    db = SessionLocal()
                    db.add(new_prof)
                    db.commit()
                    db.close()
                    st.success(f"Profiel '{naam_prof}' toegevoegd!")
                    force_refresh()

    # [ Tab 5: Behandelingen ]
    with tabs[5]:
        st.header("Behandelingen")
        db = SessionLocal()
        treatments = db.query(DBTreatment).all()
        db.close()
        if treatments:
            st.subheader("Bestaande Behandelingen")
            for treat in treatments:
                st.write(f"**Naam:** {treat.name} | **Basis:** {treat.basis} | **Prijs per eenheid:** €{treat.price_per_unit:.2f}")
        else:
            st.info("Geen behandelingen gevonden.")

        st.subheader("Voeg nieuwe behandeling toe")
        with st.form(key="form_behandelingen"):
            naam_treat = st.text_input("Naam van de behandeling")
            basis = st.selectbox("Basis", ["m²", "kg"])
            prijs_per_unit = st.number_input("Prijs per eenheid (EUR)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            submit_treat = st.form_submit_button("Behandeling toevoegen")
            if submit_treat:
                if naam_treat.strip() == "":
                    st.error("Voer een geldige naam in.")
                else:
                    new_treat = DBTreatment(name=naam_treat.strip(), basis=basis, price_per_unit=prijs_per_unit)
                    db = SessionLocal()
                    db.add(new_treat)
                    db.commit()
                    db.close()
                    st.success(f"Behandeling '{naam_treat}' toegevoegd!")
                    force_refresh()

    # [ Tab 6: Speciale Items ]
    with tabs[6]:
        st.header("Speciale Items")
        db = SessionLocal()
        special_items = db.query(DBSpecialItem).all()
        db.close()
        if special_items:
            st.subheader("Bestaande Speciale Items")
            for item in special_items:
                st.write(f"**Naam:** {item.name} | **Beschrijving:** {item.description} | **Prijs:** €{item.price:.2f} | **Standaard hoeveelheid:** {item.default_quantity}")
        else:
            st.info("Geen speciale items gevonden.")

        st.subheader("Voeg nieuw speciaal item toe")
        with st.form(key="form_speciale_items"):
            naam_item = st.text_input("Naam van het speciale item")
            beschrijving_item = st.text_area("Beschrijving")
            prijs_item = st.number_input("Prijs (EUR)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            default_qty = st.number_input("Standaard hoeveelheid", min_value=1, value=1, step=1)
            submit_item = st.form_submit_button("Special item toevoegen")
            if submit_item:
                if naam_item.strip() == "":
                    st.error("Voer een geldige naam in.")
                else:
                    new_item = DBSpecialItem(
                        name=naam_item.strip(),
                        description=beschrijving_item,
                        price=prijs_item,
                        default_quantity=default_qty
                    )
                    db = SessionLocal()
                    db.add(new_item)
                    db.commit()
                    db.close()
                    st.success(f"Special item '{naam_item}' toegevoegd!")
                    force_refresh()

    # [ Tab 7: Isolatie ]
    with tabs[7]:
        st.header("Isolatie")
        db = SessionLocal()
        isolatie_items = db.query(DBIsolation).all()
        db.close()
        if isolatie_items:
            st.subheader("Bestaande Isolatie-items")
            for iso in isolatie_items:
                st.write(f"**Naam:** {iso.name} | **Standaard oppervlakte:** {iso.default_area} m² | **Prijs per m²:** €{iso.price_per_m2:.2f}")
        else:
            st.info("Geen isolatie-items gevonden.")

        st.subheader("Voeg nieuw isolatie-item toe")
        with st.form(key="form_isolatie"):
            naam_iso = st.text_input("Naam van isolatie")
            standaard_oppervlakte = st.number_input("Standaard oppervlakte (m²)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            prijs_m2 = st.number_input("Prijs per m² (EUR)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            submit_iso = st.form_submit_button("Isolatie-item toevoegen")
            if submit_iso:
                if naam_iso.strip() == "":
                    st.error("Voer een geldige naam in.")
                else:
                    new_iso = DBIsolation(name=naam_iso.strip(), default_area=standaard_oppervlakte, price_per_m2=prijs_m2)
                    db = SessionLocal()
                    db.add(new_iso)
                    db.commit()
                    db.close()
                    st.success(f"Isolatie-item '{naam_iso}' toegevoegd!")
                    force_refresh()

    # [ Tab 8: Gaas ]
    with tabs[8]:
        st.header("Gaas")
        db = SessionLocal()
        mesh_items = db.query(DBMesh).all()
        db.close()
        if mesh_items:
            st.subheader("Bestaande Gaas-items")
            for mesh in mesh_items:
                st.write(f"**Naam:** {mesh.name} | **Standaard oppervlakte:** {mesh.default_area} m² | **Prijs per m²:** €{mesh.price_per_m2:.2f}")
        else:
            st.info("Geen gaas-items gevonden.")

        st.subheader("Voeg nieuw gaas-item toe")
        with st.form(key="form_mesh"):
            naam_mesh = st.text_input("Naam van gaas")
            standaard_oppervlakte_mesh = st.number_input("Standaard oppervlakte (m²)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            prijs_m2_mesh = st.number_input("Prijs per m² (EUR)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            submit_mesh = st.form_submit_button("Gaas-item toevoegen")
            if submit_mesh:
                if naam_mesh.strip() == "":
                    st.error("Voer een geldige naam in.")
                else:
                    new_mesh = DBMesh(name=naam_mesh.strip(), default_area=standaard_oppervlakte_mesh, price_per_m2=prijs_m2_mesh)
                    db = SessionLocal()
                    db.add(new_mesh)
                    db.commit()
                    db.close()
                    st.success(f"Gaas-item '{naam_mesh}' toegevoegd!")
                    force_refresh()

# =====================
#   reset_app
# =====================
def reset_app():
    if st.session_state.get("role", "viewer") == "admin":
        if st.button("Bevestig Reset"):
            st.session_state["calc_data"] = {
                "date": date.today().isoformat(),
                "geldigheidsduur": "30 dagen",
                "klant_naam": "",
                "klant_adres": "",
                "klant_contact": "",
                "comments": "",
                "marge_type": "Winstmarge (%)",
                "margin_percentage": 20.0,
                "storage_percentage": 10.0,
                "vat_percentage": 21.0,
                "uurloon": 0.0,
                "kilometers": 0.0,
                "kosten_per_kilometer": 0.0,
                "total_net_cost": 0.0,
                "storage_cost": 0.0,
                "total_internal_cost": 0.0,
                "total_revenue_excl_vat": 0.0,
                "vat_amount": 0.0,
                "total": 0.0,
                "total_profit": 0.0,
                "kostprijs": 0.0,
                "subtotal_vor_btw": 0.0,
                "total_weight_all": 0.0,
                "total_area": 0.0,
                "component_details": [],
                "original_calc_data": None,
                "scenarios": {},
                "global_margin": 0
            }
            st.session_state["num_items"] = 1
            st.session_state["calc_history"] = []
            try:
                # VERVANG experimental_rerun
                force_refresh()  # <--- 
            except AttributeError:
                st.warning("Refresh de pagina handmatig om de reset door te voeren.")
    else:
        st.error("Je hebt geen toestemming om de applicatie te resetten.")

# =====================
#   main()
# =====================
def main():
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #888;
        }
        </style>
        """, unsafe_allow_html=True
    )
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
        st.session_state["username"] = ""
        st.session_state["role"] = ""
    if not st.session_state["authentication_status"]:
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Gebruikersnaam")
        password = st.sidebar.text_input("Wachtwoord", type="password")
        if st.sidebar.button("Inloggen"):
            users = {
                "admin": {"password": "adminpass", "role": "admin"},
                "editor": {"password": "editorpass", "role": "editor"},
                "viewer": {"password": "viewerpass", "role": "viewer"}
            }
            user = users.get(username)
            if user and password == user["password"]:
                st.session_state["authentication_status"] = True
                st.session_state["username"] = username
                st.session_state["role"] = user["role"]
                st.sidebar.success("Inloggen succesvol!")
                logging.info(f"User '{username}' logged in as {user['role']}.")
            else:
                st.sidebar.error("Gebruikersnaam of wachtwoord is incorrect.")
                logging.warning(f"Failed login attempt for username: '{username}'")
    else:
        st.sidebar.title("Wecalcu Menu")
        menu = [
            "Calculatie",
            "Offerte",
            "Geavanceerde Rapportage en Visualisatie",
            "Scenario Analyse en What‑If Modellering",
            "Budget Uren",
            "Database",
            "PDF Calculatie",
            "ML Predictive Analysis en Toekomstige Trend Simulatie"
        ]
        if st.session_state["role"] == "admin":
            menu.append("Reset App")
        choice = st.sidebar.radio("Navigatie:", menu)
        if choice == "Calculatie":
            page_calculatie()
        elif choice == "Offerte":
            page_offerte()
        elif choice == "Geavanceerde Rapportage en Visualisatie":
            page_geavanceerde_rapportage()
        elif choice == "Scenario Analyse en What‑If Modellering":
            page_scenario_analyse()
        elif choice == "Budget Uren":
            page_budget_uren()
        elif choice == "Database":
            page_database()
        elif choice == "PDF Calculatie":
            page_pdf_calculatie()
        elif choice == "ML Predictive Analysis en Toekomstige Trend Simulatie":
            page_ml_predictive()
        elif choice == "Reset App" and st.session_state["role"] == "admin":
            reset_app()
        if st.sidebar.button("Uitloggen"):
            logging.info(f"User '{st.session_state['username']}' logged out.")
            st.session_state["authentication_status"] = False
            st.session_state["username"] = ""
            st.session_state["role"] = ""
            st.sidebar.success("Je bent uitgelogd.")
            try:
                # st.experimental_rerun() -> vervang door:
                force_refresh()
            except AttributeError:
                st.warning("Refresh de pagina handmatig om uit te loggen.")
    st.markdown(
        """
        <div class="footer">
            <p>Made by Wessel 😊</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
