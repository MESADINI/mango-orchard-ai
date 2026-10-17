-- AI Mango Orchard Health & Pest Intelligence
-- 001_seed_recommendations.sql: Verified ICAR/TNAU/DPPQS Treatment Protocols

INSERT INTO public.treatment_recommendations 
(target_condition, treatment_type, title, product_name, active_ingredient, dosage_rate, application_guidance, waiting_period_days, citation_source, citation_url)
VALUES
-- Anthracnose
(
    'Anthracnose',
    'Chemical',
    'Foliar Fungicide Spray (Blossom Emergence)',
    'Copper Oxychloride 50% WP (Blitox / Blue Copper)',
    'Copper Oxychloride 50% WP',
    '3.0 g per Litre of water (0.3%)',
    'Spray thoroughly at blossom emergence and repeat at 15-day intervals during wet or humid weather conditions to prevent panicle and twig blighting.',
    14,
    'TNAU Agritech Portal (Crop Protection: Diseases of Mango)',
    'https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_fruits_mango.html'
),
(
    'Anthracnose',
    'Chemical',
    'Systemic Fungicide Spray (Post-Flush)',
    'Azoxystrobin 23% SC (Amistar) or Carbendazim 50% WP (Bavistin)',
    'Azoxystrobin 23% SC / Carbendazim 50% WP',
    '1.0 mL/L (Azoxystrobin) or 1.0 g/L (Carbendazim)',
    'Apply during young flush emergence or fruit setting stage. Rotate systemic fungicides with copper-based protectants to prevent pathogen resistance development.',
    21,
    'ICAR-CISH Lucknow Advisory Monograph',
    'https://cish.icar.gov.in'
),
(
    'Anthracnose',
    'Biological',
    'Bio-control Foliar Application',
    'Trichoderma viride 1.0% WP or Pseudomonas fluorescens',
    'Trichoderma viride 10^8 CFU/g',
    '5.0 g per Litre of water',
    'Spray in early morning or late afternoon during humid spells. Enhances canopy antagonism against Colletotrichum gloeosporioides spores.',
    0,
    'TNAU Organic Crop Protection Guidelines',
    'https://agritech.tnau.ac.in'
),

-- Bacterial Canker
(
    'Bacterial Canker',
    'Chemical',
    'Antibiotic + Copper Combination Spray',
    'Streptocycline + Copper Oxychloride',
    'Streptomycin sulphate 90% + Tetracycline hydrochloride 10% + COC 50% WP',
    'Streptocycline 100 ppm (1 g in 10 L water) + COC 2.5 g/L',
    'Spray at 15-day intervals starting from appearance of angular dark water-soaked spots on leaves. Ensure complete coverage of twigs and fruits.',
    20,
    'DPPQS Standard Pest Management Guidelines; ICAR-CISH Bulletin',
    'https://ppqs.gov.in'
),
(
    'Bacterial Canker',
    'Cultural',
    'Sanitation & Bordeaux Paste Wound Sealing',
    'Bordeaux Paste (1:1:10)',
    'Copper sulphate 1 kg + Quicklime 1 kg in 10 L water',
    'Apply paste with brush directly to pruned branches',
    'Prune dead, cankered twigs 5 cm below the visible necrosis margin. Swab cut ends immediately with Bordeaux paste to block systemic vascular ingress.',
    0,
    'ICAR-CISH Package of Practices',
    'https://cish.icar.gov.in'
),

-- Powdery Mildew
(
    'Powdery Mildew',
    'Chemical',
    'Inorganic Sulphur Protective Spray',
    'Wettable Sulphur 80% WP (Sulfex)',
    'Sulphur 80% WP',
    '2.0 g to 3.0 g per Litre of water',
    'Apply first spray at initiation of flower panicle flush when cool nights and warm dry days favor conidial germination. Avoid spraying in direct hot sunlight (>35°C).',
    10,
    'TNAU Agritech Portal (Diseases of Mango)',
    'https://agritech.tnau.ac.in'
),
(
    'Powdery Mildew',
    'Chemical',
    'Systemic Triazole Eradicant Spray',
    'Hexaconazole 5% EC (Contaf) or Dinocap 48% EC (Karathane)',
    'Hexaconazole 5% EC',
    '1.0 mL per Litre of water',
    'Apply upon initial appearance of white superficial fungal colonies on rachis or flower petals. Provides rapid curactive action.',
    15,
    'DPPQS Major Uses of Registered Pesticides for Mango',
    'https://ppqs.gov.in'
),

-- Sooty Mold
(
    'Sooty Mold',
    'Cultural',
    'Starch Flaking Foliar Wash',
    'Boiled Maida / Cassava Starch Solution',
    'Starch 2.0% w/v',
    '20 g starch boiled in 1 Litre of water',
    'Spray the cooled 2% starch solution over leaves covered with black sooty mold. On drying, the starch membrane contracts, flakes off, and mechanically removes the mold crust.',
    0,
    'TNAU Horticultural College & Research Institute, Periyakulam',
    'https://agritech.tnau.ac.in'
),
(
    'Sooty Mold',
    'Chemical',
    'Sucking Insect Vector Suppression',
    'Fish Oil Rosin Soap (FORS) or Neem Oil 10,000 ppm',
    'Azadirachtin 1% EC',
    'FORS 25 g/L or Neem Oil 2.0 mL/L',
    'Sooty mold feeds on honeydew secreted by scale insects, hoppers, and mealybugs. Controlling insect vectors stops fungal proliferation.',
    3,
    'ICAR-CISH Entomological Circular #18',
    'https://cish.icar.gov.in'
),

-- Leaf Spot (not a model class: no licensed training data; kept for reference only)
(
    'Leaf Spot',
    'Chemical',
    'Dithiocarbamate Protective Foliar Spray',
    'Mancozeb 75% WP (Dithane M-45)',
    'Mancozeb 75% WP',
    '2.5 g per Litre of water',
    'Apply 2-3 sprays at 14-day intervals starting with leaf flush. Protects new foliage from Pestalotiopsis and Phoma spot complexes.',
    14,
    'ICAR-CISH Technical Monograph Series on Foliar Pathogens',
    'https://cish.icar.gov.in'
),

-- Fruit Fly
(
    'Fruit Fly',
    'Biological',
    'Para-Pheromone Lure Trapping',
    'Methyl Eugenol Trap (CISH / TNAU Design)',
    'Methyl Eugenol 1.0 mL + Malathion 50% EC 0.5 mL per dispenser',
    '10 to 15 traps per hectare',
    'Hang traps inside canopy 1.5 to 2.0 meters above ground level, 60 days prior to harvest maturity. Recharge dispenser every 30 days.',
    0,
    'ICAR-CISH IPM Package for Bactrocera dorsalis',
    'https://cish.icar.gov.in'
),
(
    'Fruit Fly',
    'Chemical',
    'Canopy Bait Spot Spray',
    'Protein Hydrolysate / Jaggery Bait Spray',
    'Jaggery 100 g + Malathion 50% EC 20 mL in 10 L water',
    'Coarse droplet spot spray on lower canopy leaves',
    'Apply in spots to under-surface of foliage once a week during fruiting stage. Do not spray directly on developing fruit clusters.',
    7,
    'TNAU Department of Agricultural Entomology Advisory',
    'https://agritech.tnau.ac.in'
),

-- Mealy Bug
(
    'Mealy Bug',
    'Cultural',
    'Trunk Polythene Barrier Banding',
    'Polythene Banding (400 gauge, 30 cm wide)',
    'Slippery Polyethylene Film + Grease band',
    '1 band per tree trunk 50 cm above ground level',
    'Wrap band tightly around trunk in November-December before first instar nymphs crawl up from orchard soil. Smear grease along upper edge.',
    0,
    'ICAR-CISH IPM Bulletin for Drosicha mangiferae',
    'https://cish.icar.gov.in'
),
(
    'Mealy Bug',
    'Chemical',
    'Insect Growth Regulator Foliar Spray',
    'Buprofezin 25% SC (Applaud)',
    'Buprofezin 25% SC',
    '1.5 mL per Litre of water',
    'Spray infested shoots and branches upon nymph appearance. Inhibits chitin synthesis and suppresses subsequent generations.',
    14,
    'DPPQS Approved Insecticide Schedule for Mango',
    'https://ppqs.gov.in'
),

-- Scale Insect
(
    'Scale Insect',
    'Chemical',
    'Systemic Organophosphate / Pyrethroid Spray',
    'Dimethoate 30% EC (Rogor) or Thiamethoxam 25% WG',
    'Dimethoate 30% EC 1.7 mL/L or Thiamethoxam 0.3 g/L',
    '1.7 mL per Litre of water',
    'Direct high-pressure spray to underside of leaves and young green stems where scales encrust.',
    21,
    'TNAU Agritech Portal (Crop Protection: Mango Pests)',
    'https://agritech.tnau.ac.in'
);
