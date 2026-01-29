-- CardioVoice Database Schema
-- For Med-Gemma Impact Challenge
-- PostgreSQL Database

-- Create database (run as postgres user)
-- createdb cardivoice_db

-- Connect and run this script
-- psql -d cardivoice_db -f schema.sql

-- ============================================================================
-- USERS TABLE (for future multi-user support)
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    role VARCHAR(20) DEFAULT 'physician',
    department VARCHAR(50) DEFAULT 'cardiology',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- ============================================================================
-- PATIENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    date_of_birth DATE,
    phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    emergency_contact VARCHAR(200),
    insurance_id VARCHAR(100),
    
    -- Medical information
    allergies TEXT,
    medications TEXT,
    medical_history TEXT,
    family_history TEXT,
    social_history TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    
    CONSTRAINT valid_gender CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say'))
);

-- ============================================================================
-- ENCOUNTERS TABLE (Outpatient Visits)
-- ============================================================================

CREATE TABLE IF NOT EXISTS encounters (
    id SERIAL PRIMARY KEY,
    encounter_id VARCHAR(50) UNIQUE NOT NULL,
    patient_id INTEGER REFERENCES patients(id),
    physician_id INTEGER REFERENCES users(id),
    
    -- Visit timing
    visit_date DATE NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_minutes INTEGER,
    
    -- Visit type
    visit_type VARCHAR(50) DEFAULT 'outpatient',
    template_type VARCHAR(20) DEFAULT 'general',
    
    -- Status
    status VARCHAR(20) DEFAULT 'in_progress',
    audio_quality VARCHAR(20) DEFAULT 'good',
    
    -- Audio file
    audio_file_path VARCHAR(500),
    audio_duration_seconds INTEGER,
    audio_sample_rate INTEGER DEFAULT 16000,
    
    -- Transcription
    raw_transcription TEXT,
    transcription_quality_score DECIMAL(3,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_status CHECK (status IN ('in_progress', 'draft', 'verified', 'final', 'cancelled')),
    CONSTRAINT valid_visit_type CHECK (visit_type IN ('outpatient', 'follow_up', 'consultation', 'emergency')),
    CONSTRAINT valid_template CHECK (template_type IN ('general', 'cardiology', 'internal_medicine', 'pediatrics'))
);

-- ============================================================================
-- OUTPATIENT RECORDS TABLE (Generated Medical Notes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS outpatient_records (
    id SERIAL PRIMARY KEY,
    encounter_id INTEGER REFERENCES encounters(id),
    patient_id INTEGER REFERENCES patients(id),
    physician_id INTEGER REFERENCES users(id),
    
    -- Patient demographics (snapshot at time of visit)
    patient_name VARCHAR(100),
    patient_age INTEGER,
    patient_gender VARCHAR(10),
    
    -- Structured note content
    chief_complaint TEXT,
    history_of_present_illness TEXT,
    past_medical_history TEXT,
    past_surgical_history TEXT,
    medications TEXT,
    allergies TEXT,
    family_history TEXT,
    social_history TEXT,
    review_of_systems TEXT,
    
    -- Physical examination
    general_appearance TEXT,
    vital_signs TEXT,
    cardiovascular_exam TEXT,
    respiratory_exam TEXT,
    abdominal_exam TEXT,
    extremity_exam TEXT,
    neurological_exam TEXT,
    other_exam_findings TEXT,
    
    -- Cardiovascular-specific
    ecg_findings TEXT,
    echocardiogram_findings TEXT,
    stress_test_results TEXT,
    
    -- Assessment and Plan
    assessment TEXT,
    problem_list TEXT,
    differential_diagnosis TEXT,
    plan TEXT,
    medications_prescribed TEXT,
    tests_ordered TEXT,
    procedures_performed TEXT,
    patient_instructions TEXT,
    follow_up_instructions TEXT,
    
    -- Additional notes
    physician_comments TEXT,
    ai_generated_content BOOLEAN DEFAULT TRUE,
    ai_confidence_score DECIMAL(3,2),
    
    -- Review and verification
    status VARCHAR(20) DEFAULT 'draft',
    doctor_verified BOOLEAN DEFAULT FALSE,
    verified_by INTEGER REFERENCES users(id),
    verified_at TIMESTAMP,
    doctor_notes TEXT,
    
    -- AI raw response (for debugging and improvement)
    raw_llm_response TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_at TIMESTAMP,
    
    CONSTRAINT valid_record_status CHECK (status IN ('draft', 'review', 'verified', 'final', 'modified'))
);

-- ============================================================================
-- TRANSCRIPTION CHUNKS TABLE (For real-time transcription)
-- ============================================================================

CREATE TABLE IF NOT EXISTS transcription_chunks (
    id SERIAL PRIMARY KEY,
    encounter_id INTEGER REFERENCES encounters(id),
    chunk_order INTEGER NOT NULL,
    audio_start_time DECIMAL(10,3),
    audio_end_time DECIMAL(10,3),
    transcribed_text TEXT,
    confidence_score DECIMAL(3,2),
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOCTOR ANNOTATIONS TABLE (Track edits and corrections)
-- ============================================================================

CREATE TABLE IF NOT EXISTS doctor_annotations (
    id SERIAL PRIMARY KEY,
    record_id INTEGER REFERENCES outpatient_records(id),
    field_name VARCHAR(50),
    original_ai_text TEXT,
    corrected_text TEXT,
    correction_reason TEXT,
    correction_type VARCHAR(20),
    time_spent_seconds INTEGER,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_correction_type CHECK (correction_type IN ('factual_error', 'omission', 'terminology', 'formatting', 'clarity', 'other'))
);

-- ============================================================================
-- AUDIO RECORDINGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS audio_recordings (
    id SERIAL PRIMARY KEY,
    encounter_id INTEGER REFERENCES encounters(id),
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    duration_seconds INTEGER,
    sample_rate INTEGER DEFAULT 16000,
    channels INTEGER DEFAULT 1,
    bits_per_sample INTEGER DEFAULT 16,
    audio_format VARCHAR(20) DEFAULT 'wav',
    compression VARCHAR(50),
    quality_score DECIMAL(3,2),
    noise_level DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_audio_format CHECK (audio_format IN ('wav', 'mp3', 'flac', 'm4a', 'webm'))
);

-- ============================================================================
-- SYSTEM LOGS TABLE (For debugging and audit)
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    log_message TEXT,
    component VARCHAR(50),
    encounter_id INTEGER REFERENCES encounters(id),
    user_id INTEGER REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_log_level CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

-- ============================================================================
-- INDEXES (For query performance)
-- ============================================================================

-- Patient indexes
CREATE INDEX idx_patients_patient_id ON patients(patient_id);
CREATE INDEX idx_patients_name ON patients(name);
CREATE INDEX idx_patients_dob ON patients(date_of_birth);

-- Encounter indexes
CREATE INDEX idx_encounters_patient ON encounters(patient_id);
CREATE INDEX idx_encounters_physician ON encounters(physician_id);
CREATE INDEX idx_encounters_date ON encounters(visit_date);
CREATE INDEX idx_encounters_status ON encounters(status);

-- Record indexes
CREATE INDEX idx_records_encounter ON outpatient_records(encounter_id);
CREATE INDEX idx_records_patient ON outpatient_records(patient_id);
CREATE INDEX idx_records_date ON outpatient_records(created_at);
CREATE INDEX idx_records_status ON outpatient_records(status);
CREATE INDEX idx_records_physician ON outpatient_records(physician_id);

-- Transcription indexes
CREATE INDEX idx_transcription_encounter ON transcription_chunks(encounter_id);
CREATE INDEX idx_transcription_order ON transcription_chunks(chunk_order);

-- Annotation indexes
CREATE INDEX idx_annotations_record ON doctor_annotations(record_id);
CREATE INDEX idx_annotations_field ON doctor_annotations(field_name);

-- Log indexes
CREATE INDEX idx_logs_level ON system_logs(log_level);
CREATE INDEX idx_logs_component ON system_logs(component);
CREATE INDEX idx_logs_created ON system_logs(created_at);

-- ============================================================================
-- VIEWS (For common queries)
-- ============================================================================

-- View: Today's encounters
CREATE OR REPLACE VIEW v_today_encounters AS
SELECT 
    e.*,
    p.name as patient_name,
    p.age,
    p.gender,
    u.username as physician_name
FROM encounters e
JOIN patients p ON e.patient_id = p.id
JOIN users u ON e.physician_id = u.id
WHERE e.visit_date = CURRENT_DATE;

-- View: Pending reviews
CREATE OR REPLACE VIEW v_pending_reviews AS
SELECT 
    orec.*,
    p.name as patient_name,
    u.username as physician_name
FROM outpatient_records orec
JOIN patients p ON orec.patient_id = p.id
JOIN users u ON orec.physician_id = u.id
WHERE orec.status = 'draft'
ORDER BY orec.created_at DESC;

-- View: Annotation summary by field
CREATE OR REPLACE VIEW v_annotation_summary AS
SELECT 
    record_id,
    field_name,
    COUNT(*) as correction_count,
    COUNT(DISTINCT correction_type) as unique_correction_types
FROM doctor_annotations
GROUP BY record_id, field_name;

-- ============================================================================
-- FUNCTIONS (Stored procedures)
-- ============================================================================

-- Function: Update timestamp on modification
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp update
CREATE TRIGGER update_patients_timestamp
    BEFORE UPDATE ON patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_encounters_timestamp
    BEFORE UPDATE ON encounters
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_records_timestamp
    BEFORE UPDATE ON outpatient_records
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function: Generate unique encounter ID
CREATE OR REPLACE FUNCTION generate_encounter_id()
RETURNS VARCHAR(50) AS $$
BEGIN
    RETURN 'ENC' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || 
           '-' || 
           LPAD(FLOOR(RANDOM() * 1000000)::TEXT, 6, '0');
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate AI correction rate
CREATE OR REPLACE FUNCTION calculate_correction_rate(p_record_id INTEGER)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    v_total_fields INTEGER;
    v_corrected_fields INTEGER;
BEGIN
    SELECT 
        COUNT(DISTINCT field_name),
        COUNT(DISTINCT CASE WHEN corrected_text IS NOT NULL THEN field_name END)
    INTO v_total_fields, v_corrected_fields
    FROM doctor_annotations
    WHERE record_id = p_record_id;
    
    IF v_total_fields = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN (v_corrected_fields::DECIMAL / v_total_fields) * 100;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA (For testing)
-- ============================================================================

-- Insert sample physician
INSERT INTO users (username, email, role, department)
VALUES ('dr_smith', 'smith@hospital.com', 'physician', 'cardiology')
ON CONFLICT (username) DO NOTHING;

-- Insert sample patient
INSERT INTO patients (patient_id, name, age, gender, allergies, medications)
VALUES ('P001', 'John Johnson', 58, 'male', 'Penicillin', 'Lisinopril 10mg, Atorvastatin 20mg')
ON CONFLICT (patient_id) DO NOTHING;

-- ============================================================================
-- GRANT PERMISSIONS (Adjust as needed)
-- ============================================================================

-- GRANT ALL PRIVILEGES ON DATABASE cardivoice_db TO cardivoice_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cardivoice_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cardivoice_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO cardivoice_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO cardivoice_user;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE users IS 'System users including physicians and staff';
COMMENT ON TABLE patients IS 'Patient demographic and medical history information';
COMMENT ON TABLE encounters IS 'Patient visit records with audio and transcription';
COMMENT ON TABLE outpatient_records IS 'AI-generated outpatient documentation';
COMMENT ON TABLE transcription_chunks IS 'Real-time transcription segments';
COMMENT ON TABLE doctor_annotations IS 'Physician corrections and annotations';
COMMENT ON TABLE audio_recordings IS 'Audio file metadata and storage';
COMMENT ON TABLE system_logs IS 'Application logs for debugging and audit';

COMMENT ON COLUMN outpatient_records.ai_confidence_score IS 'Confidence score from MedGemma (0-1)';
COMMENT ON COLUMN encounters.audio_quality IS 'Audio quality assessment (good/fair/poor)';
COMMENT ON COLUMN encounters.transcription_quality_score IS 'Transcription accuracy score (0-1)';
