-- AI Mango Orchard Health & Pest Intelligence
-- 001_initial_schema.sql: Core Schema & Row Level Security (RLS)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Farms Table
CREATE TABLE IF NOT EXISTS public.farms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    cultivar VARCHAR(100) DEFAULT 'Alphonso',
    acreage DECIMAL(8, 2) DEFAULT 5.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trees Table
CREATE TABLE IF NOT EXISTS public.trees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    tag_code VARCHAR(50) NOT NULL,
    row_number INTEGER,
    tree_number INTEGER,
    variety VARCHAR(100),
    planted_year INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scans Table
CREATE TABLE IF NOT EXISTS public.scans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farm_id UUID REFERENCES public.farms(id) ON DELETE SET NULL,
    tree_id UUID REFERENCES public.trees(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    scan_type VARCHAR(50) DEFAULT 'dual', -- 'disease', 'pest', 'dual'
    ohi_score DECIMAL(5, 2),
    nutrient_input_was_defaulted BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Disease Detections Table
CREATE TABLE IF NOT EXISTS public.disease_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID NOT NULL REFERENCES public.scans(id) ON DELETE CASCADE,
    disease_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    severity_percentage DECIMAL(5, 2) DEFAULT 0.0,
    gradcam_url TEXT,
    is_healthy BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pest Detections Table
CREATE TABLE IF NOT EXISTS public.pest_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID NOT NULL REFERENCES public.scans(id) ON DELETE CASCADE,
    pest_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    bbox_x1 DECIMAL(8, 2),
    bbox_y1 DECIMAL(8, 2),
    bbox_x2 DECIMAL(8, 2),
    bbox_y2 DECIMAL(8, 2),
    is_unreliable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Weather Records Table
CREATE TABLE IF NOT EXISTS public.weather_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    temperature DECIMAL(5, 2) NOT NULL,
    relative_humidity DECIMAL(5, 2) NOT NULL,
    precipitation DECIMAL(6, 2) NOT NULL,
    wind_speed DECIMAL(6, 2),
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Treatment Recommendations Table (Source-Cited)
CREATE TABLE IF NOT EXISTS public.treatment_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target_condition VARCHAR(100) NOT NULL, -- e.g. 'Anthracnose', 'Fruit Fly'
    treatment_type VARCHAR(50) NOT NULL, -- 'Chemical', 'Biological', 'Cultural'
    title VARCHAR(255) NOT NULL,
    product_name VARCHAR(255),
    active_ingredient VARCHAR(255),
    dosage_rate VARCHAR(100) NOT NULL,
    application_guidance TEXT NOT NULL,
    waiting_period_days INTEGER DEFAULT 0,
    citation_source VARCHAR(255) NOT NULL, -- 'ICAR-CISH', 'TNAU', 'DPPQS'
    citation_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bilingual Advisory Chat Sessions
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    farm_id UUID REFERENCES public.farms(id) ON DELETE SET NULL,
    language_code VARCHAR(10) DEFAULT 'en', -- 'en' or 'ta'
    title VARCHAR(255) DEFAULT 'New Consultation',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    language_code VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security (RLS)
ALTER TABLE public.farms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trees ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.disease_detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pest_detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weather_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.treatment_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can manage own farms" ON public.farms
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view farm trees" ON public.trees
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.farms WHERE farms.id = trees.farm_id AND farms.user_id = auth.uid())
    );

CREATE POLICY "Users can manage own scans" ON public.scans
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can read disease detections for own scans" ON public.disease_detections
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.scans WHERE scans.id = disease_detections.scan_id AND scans.user_id = auth.uid())
    );

CREATE POLICY "Users can read pest detections for own scans" ON public.pest_detections
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.scans WHERE scans.id = pest_detections.scan_id AND scans.user_id = auth.uid())
    );

CREATE POLICY "Public read-only for verified treatment recommendations" ON public.treatment_recommendations
    FOR SELECT TO authenticated, anon USING (true);

CREATE POLICY "Users can view weather records for their farms" ON public.weather_records
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.farms WHERE farms.id = weather_records.farm_id AND farms.user_id = auth.uid())
    );

CREATE POLICY "Users can manage own chat sessions" ON public.chat_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage messages in own chat sessions" ON public.chat_messages
    FOR ALL USING (
        EXISTS (SELECT 1 FROM public.chat_sessions WHERE chat_sessions.id = chat_messages.session_id AND chat_sessions.user_id = auth.uid())
    );
