# Treatment sources and their verification

The app recommends only what is in `ml/forecasting/recommendations_db.json`. Each entry links to the exact page or
document it comes from (`citation_url`) and quotes it (`citation_quote`). All entries were checked against the live
sources on **2026-10-17**. Tests in `ml/forecasting/tests/test_recommendations.py` enforce deep links, quotes and the
banned-chemical list. The optional SQL seed (`database/seeds/`) is generated from the same file.

## Sources used

| Source | What it supports | Link |
|---|---|---|
| TNAU Agritech Portal, Crop Protection: Mango, Anthracnose | *P. fluorescens* spray | https://agritech.tnau.ac.in/crop_protection/mango_1.html |
| TNAU Agritech Portal, Powdery mildew | wettable sulphur 0.2%, timing | https://agritech.tnau.ac.in/crop_protection/mango_2.html |
| TNAU Agritech Portal, Sooty mould | insect control + starch spray recipe | https://agritech.tnau.ac.in/crop_protection/mango_7.html |
| TNAU Agritech Portal, Fruit fly | sanitation, ploughing | https://agritech.tnau.ac.in/crop_protection/mango/mango_8.html |
| TNAU Agritech Portal, Giant mealybug | trunk banding, *Cryptolaemus* release | https://agritech.tnau.ac.in/crop_protection/mango/mango_15.html |
| TNAU Agritech Portal, Scales | pruning | https://agritech.tnau.ac.in/crop_protection/mango/mango_14.html |
| National Horticulture Mission, *IPM Schedule for Mango Pests*, Extension Bulletin No. 1 (Horticulture Year 2012; PDF hosted by TNAU) | copper oxychloride, carbendazim, canker, hopper, stem borer, scale, methyl eugenol traps; disease-favourable weather used by the forecaster | https://agritech.tnau.ac.in/horticulture/pdf/tech_bulletin/national/IPM-Mango-Revised-Sept2011.pdf |
| CIB&RC / DPPQS, *List of pesticides which are banned, refused registration and restricted in use* (updated 31.07.2026) | regulatory check of every chemical | https://ppqs.gov.in/sites/default/files/list_of_pesticides_which_are_banned_refused_registration_and_restricted_in_use.pdf |

**Not verified:** whether each product's registered label claim includes mango. The CIB&RC "major uses" list was not
checked product by product. The app therefore tells the user to check the label and its pre-harvest interval. A
pre-harvest interval is stated only where the source gives one (carbendazim: last spray 15 days before harvest).

## Removed or corrected after verification

| Condition | Previous advice | Finding | Action |
|---|---|---|---|
| Fruit fly | Methyl eugenol + **malathion** traps; jaggery + **malathion** bait | Malathion is banned for use on mango (S.O. 4294(E), 3 Oct 2023) | Malathion removed; traps kept (10/ha, April–August, NHM) with a warning about the trap insecticide |
| Fruit fly | "10–15 traps/ha, 60 days before harvest" (ICAR-CISH) | Not found in the cited source; NHM gives 10 traps/ha during fruiting | Corrected |
| Scale insect | **Dimethoate** 30% EC 1.7 mL/L | Dimethoate is banned on fruits eaten raw (S.O. 4294(E), 2023); TNAU's alternative, phosphamidon, is banned (S.O. 3951(E), 2018) | Chemical removed; pruning + ant removal kept |
| Stem borer (chatbot) | **Dichlorvos** 76% EC injection | Dichlorvos is banned (S.O. 3951(E), 2018) | Replaced by NHM's kerosene/petrol plug and biocontrol |
| Mango hopper (chatbot) | Imidacloprid 17.8% SL 0.3 mL/L; thiamethoxam (ICAR-CISH) | Source not found; NHM gives imidacloprid 200 SL 0.25 mL/L, need-based | Corrected to NHM |
| Powdery mildew | Sulphur 2–3 g/L; hexaconazole; **dinocap** | TNAU/NHM give 0.2% (2 g/L); dinocap is banned (S.O. 4294(E), 2023); tridemorph (in both sources) is banned (2018) | Dose corrected; dinocap/tridemorph not suggested |
| Anthracnose | Copper oxychloride 0.3% cited to TNAU; azoxystrobin (ICAR-CISH) | TNAU's page gives *P. fluorescens*, not copper; azoxystrobin source not found | Copper re-cited to NHM; *P. fluorescens* added from TNAU; azoxystrobin removed |
| Bacterial canker | Streptocycline 100 ppm + COC 2.5 g/L (DPPQS) | Not found at the cited source; NHM gives 200–300 ppm Streptocycline or COC 0.3%. A phase-out of streptomycin + tetracycline in agriculture was ordered in draft | Copper oxychloride 0.3% kept (NHM); Streptocycline not suggested |
| Sooty mould | Starch 2% (TNAU Periyakulam) | TNAU page: 1 kg starch in 5 L, diluted to 20 L, after insect control; NHM: 2% | Re-cited to TNAU page with its recipe |
| Mealy bug | Band 30 cm wide, November–December | TNAU: 20 cm wide, 400 gauge, mid-December, 50 cm above ground | Corrected |

Advice for Leaf Spot is not given: it is not a model class (no licensed training images).
