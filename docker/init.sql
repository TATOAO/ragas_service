-- Initialize RAGAS database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database if it doesn't exist
-- (This is handled by the POSTGRES_DB environment variable)

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ragas TO ragas_user;

-- Connect to the ragas database
\c ragas;

-- Create tables (these will be created by SQLAlchemy, but we can add any custom setup here)
-- The actual table creation is handled by the FastAPI application using SQLAlchemy

-- Create indexes for better performance
-- (These will be created by SQLAlchemy, but we can add custom indexes here)

-- Insert default metrics data
INSERT INTO metrics (metric_id, name, description, metric_type, supported_sample_types, parameters, default_config, example_usage, is_active, created_at, updated_at) VALUES
(
    uuid_generate_v4(),
    'answer_relevancy',
    'Measures how relevant the answer is to the question',
    'llm_based',
    '["single_turn", "multi_turn"]',
    '{"llm_required": true, "embeddings_required": false}',
    '{"llm": "gpt-4o", "embeddings": null}',
    '{"sample": {"user_input": "What is the capital of France?", "response": "The capital of France is Paris."}, "expected_score": 0.9}',
    true,
    NOW(),
    NOW()
),
(
    uuid_generate_v4(),
    'context_precision',
    'Measures the precision of retrieved contexts',
    'embedding_based',
    '["single_turn", "multi_turn"]',
    '{"llm_required": false, "embeddings_required": true}',
    '{"llm": null, "embeddings": "text-embedding-3-small"}',
    '{"sample": {"user_input": "What is the capital of France?", "retrieved_contexts": ["Paris is the capital of France."]}, "expected_score": 0.8}',
    true,
    NOW(),
    NOW()
),
(
    uuid_generate_v4(),
    'faithfulness',
    'Measures how faithful the answer is to the provided context',
    'llm_based',
    '["single_turn", "multi_turn"]',
    '{"llm_required": true, "embeddings_required": false}',
    '{"llm": "gpt-4o", "embeddings": null}',
    '{"sample": {"user_input": "What is the capital of France?", "retrieved_contexts": ["Paris is the capital of France."], "response": "The capital of France is Paris."}, "expected_score": 0.85}',
    true,
    NOW(),
    NOW()
),
(
    uuid_generate_v4(),
    'context_recall',
    'Measures the recall of retrieved contexts',
    'embedding_based',
    '["single_turn", "multi_turn"]',
    '{"llm_required": false, "embeddings_required": true}',
    '{"llm": null, "embeddings": "text-embedding-3-small"}',
    '{"sample": {"user_input": "What is the capital of France?", "retrieved_contexts": ["Paris is the capital of France."], "reference_contexts": ["Paris is the capital and largest city of France."]}, "expected_score": 0.7}',
    true,
    NOW(),
    NOW()
)
ON CONFLICT (name) DO NOTHING;
