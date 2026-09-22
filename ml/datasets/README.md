# Mango Orchard Datasets (Phase 1)

## 1. Disease Detection Dataset Strategy

### Target Schema Classes (6 Total)
1. **Anthracnose** (*Colletotrichum gloeosporioides*)
2. **Bacterial Canker** (*Xanthomonas citri* pv. *mangiferaeindicae*)
3. **Powdery Mildew** (*Oidium mangiferae*)
4. **Sooty Mold** (*Capnodium mangiferae*)
5. **Leaf Spot** (*Pestalotiopsis mangiferae*) — *Supplemental Source Required*
6. **Healthy**

### MangoLeafBD Mapping Table
| Original MangoLeafBD Label | Target Schema Label | Action |
|---|---|---|
| `Anthracnose` | `Anthracnose` | Direct inclusion |
| `Bacterial Canker` | `Bacterial Canker` | Direct inclusion |
| `Powdery Mildew` | `Powdery Mildew` | Direct inclusion |
| `Sooty Mould` | `Sooty Mold` | Rename spelling (`Mould` -> `Mold`) |
| `Healthy` | `Healthy` | Direct inclusion |
| `Cutting Weevil` | — | **EXCLUDE** (Pest damage, not foliar disease) |
| `Die Back` | — | **EXCLUDE** (Stem/vascular symptom) |
| `Gall Midge` | — | **EXCLUDE** (Pest gall symptom) |
| *Not present* | `Leaf Spot` | **SUPPLEMENTAL SOURCING** (Roboflow / Kaggle) |

---

## 2. Pest Detection Dataset Strategy

| Pest Class | Confirmed Sample Size | Data Source | Readiness Status |
|---|---|---|---|
| **Fruit Fly** | 4,797 images | Roboflow: `school-vakkx/mango-fruit-fly` v4 | **CONFIRMED** |
| **Mealy Bug** | 881 images | Roboflow: `grape-n5yol/mealybug-b8wfq` v8 | **CONFIRMED** |
| **Scale Insect** | 809 images | Roboflow: `ralph-matthew-r-aquino/hemispherical-scale-insect` v1 | **CONFIRMED** |
| **Mango Hopper** | < 50 images | None identified | **WEAK** — Needs manual annotation |
| **Stem Borer** | < 50 images | None / IP102 transfer | **WEAK** — Needs manual annotation |

> [!CAUTION]
> **Policy Guardrail**:
> Do NOT train YOLOv8 on `Mango Hopper` or `Stem Borer` until $\ge 100$ labeled bounding-box images per class exist.
> Both classes are flagged as `UNRELIABLE_CLASSES` in `ml/pest_detection/inference.py`.

---

## 3. Running Acquisition Scripts

```bash
# Set up Kaggle credentials at ~/.kaggle/kaggle.json
python ml/datasets/phase1_disease.py

# Set up Roboflow API key
export ROBOFLOW_API_KEY="your_key"
python ml/datasets/phase1_pest.py

# Or run orchestrator:
python ml/datasets/download_datasets.py
```
