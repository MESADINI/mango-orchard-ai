# Orchard Health Index (OHI) Methodology

## 1. Overview & Core Philosophy
The **Orchard Health Index (OHI)** is a deterministic, rule-based mathematical scoring system designed to quantify the overall physiological health, stress level, and pest/pathogen vulnerability of a mango orchard or individual tree on a scale from **0 to 100**.

Unlike opaque deep-learning end-to-end regression models, OHI is:
1. **Deterministic & Auditable**: Every single point reduction from 100 can be traced directly to an empirical vision detection, severity metric, or weather risk factor.
2. **Configurable**: Weights are tunable per agro-climatic zone and mango cultivar (e.g., Alphonso, Banganapalli, Dasheri, Totapuri).
3. **Transparent**: Missing or placeholder parameters (such as soil nutrient readings) are explicitly disclosed.

---

## 2. Mathematical Formulation

The composite Orchard Health Index is defined as:

$$\text{OHI} = \max\left(0, \min\left(100, 100 - \Delta_{\text{total}}\right)\right)$$

Where the total penalty $\Delta_{\text{total}}$ is a linear combination of four domain-specific penalty factors:

$$\Delta_{\text{total}} = w_d \cdot S_d + w_p \cdot P_p + w_w \cdot W_v + w_n \cdot N_d$$

### Default Weight Distribution
| Component | Symbol | Default Weight ($w$) | Normalized Weight ($N_d$ defaulted) |
|---|---|---|---|
| **Disease Severity Penalty** | $S_d$ | $0.40$ | $0.45$ |
| **Pest Pressure Penalty** | $P_p$ | $0.35$ | $0.40$ |
| **Weather Vulnerability Index** | $W_v$ | $0.15$ | $0.15$ |
| **Nutrient Deficiency Penalty** | $N_d$ | $0.10$ | $0.00$ *(Disclosed Placeholder)* |

$$\sum w_i = 1.0$$

---

## 3. Sub-Component Scoring Models

### A. Disease Severity Penalty ($S_d \in [0, 100]$)
Given a detected disease $c \in \{\text{Anthracnose}, \text{Bacterial Canker}, \text{Powdery Mildew}, \text{Sooty Mold}, \text{Leaf Spot}, \text{Healthy}\}$:

$$S_d = \begin{cases} 
0, & \text{if } c = \text{Healthy} \\
\min\left(100, \left(\alpha_c \cdot 40\right) + \left(A_{\text{lesion}} \cdot 60\right)\right) \cdot \text{Conf}_c, & \text{otherwise}
\end{cases}$$

Where:
- $\text{Conf}_c \in [0, 1]$ is the EfficientNet-B3 model classification confidence.
- $A_{\text{lesion}} \in [0, 1]$ is the segmented diseased surface area fraction computed by the computer vision severity module.
- $\alpha_c$ is the pathogen virulence multiplier:
  - **Anthracnose** (*Colletotrichum gloeosporioides*): $\alpha = 1.00$ (High tissue necrosis and fruit drop risk)
  - **Bacterial Canker** (*Xanthomonas citri* pv. *mangiferaeindicae*): $\alpha = 0.95$ (Systemic vascular damage)
  - **Powdery Mildew** (*Oidium mangiferae*): $\alpha = 0.85$ (Inflorescence and young flush destruction)
  - **Leaf Spot** (*Pestalotiopsis mangiferae*): $\alpha = 0.70$ (Photosynthetic inhibition)
  - **Sooty Mold** (*Capnodium mangiferae*): $\alpha = 0.50$ (Superficial fungal coat following honeydew excretion)

### B. Pest Pressure Penalty ($P_p \in [0, 100]$)
Derived from YOLOv8 bounding box counts across canopy samples:

$$P_p = \min\left(100, \sum_{i=1}^{K} \beta_{k_i} \cdot \gamma_{k_i} \cdot \text{Conf}_{k_i}\right)$$

Where:
- $\beta_k$ is the pest severity impact factor:
  - **Fruit Fly** (*Bactrocera dorsalis*): $\beta = 18.0$ per detected individual
  - **Mealy Bug** (*Drosicha mangiferae*): $\beta = 12.0$ per cluster detection
  - **Scale Insect** (*Coccus viridis*): $\beta = 10.0$ per colony detection
  - **Mango Hopper** (*Amritodus atkinsoni*): $\beta = 15.0$ *(Flagged UNRELIABLE until $\ge 100$ annotated samples)*
  - **Stem Borer** (*Batocera rufomaculata*): $\beta = 35.0$ *(Flagged UNRELIABLE until $\ge 100$ annotated samples)*
- $\gamma_k$ is a canopy density attenuation factor.

### C. Weather Vulnerability Coefficient ($W_v \in [0, 100]$)
Fetched from the Open-Meteo API for the orchard coordinates:
- Relative Humidity ($\text{RH} > 80\%$): Elevates fungal germination risk.
- Temperature Window ($22^\circ\text{C} \le T \le 32^\circ\text{C}$): Optimal pathogen propagation range.
- Precipitation ($P > 2.0\text{ mm}$): Wash-off and spore splash transmission.

$$W_v = \left(0.45 \cdot f_{\text{RH}}(\text{RH})\right) + \left(0.35 \cdot f_T(T)\right) + \left(0.20 \cdot f_P(\text{Rain})\right)$$

### D. Disclosed Nutrient Deficiency Placeholder ($N_d$)
> **Policy Compliance**:
> Currently, multi-spectral soil or leaf nitrogen sensors are not integrated.
> - Default value: $N_d = 0$.
> - API Flag: `"nutrient_input_was_defaulted": true`.
> - UI Behavior: Displays amber notification badge: *"Nutrient deficiency analysis disabled (sensor input defaulted to 0)."*
> - Weight reallocation: The $0.10$ weight is proportionally reallocated to Disease Severity ($+0.05$) and Pest Pressure ($+0.05$).

---

## 4. Health Classification Tiers

| Score Range | Tier Label | Color Code | Action Required |
|---|---|---|---|
| **85 – 100** | **Optimal Health** | `#10B981` (Emerald) | Routine surveillance, cultural weed management. |
| **70 – 84** | **Mild Stress / Warning** | `#F59E0B` (Amber) | Prophylactic biocontrol, weather monitoring. |
| **50 – 69** | **High Infestation Risk** | `#F97316` (Orange) | Targeted organic/chemical fungicide or trap placement. |
| **0 – 49** | **Severe Outbreak / Critical** | `#EF4444` (Red) | Emergency systemic intervention per ICAR guidelines. |
