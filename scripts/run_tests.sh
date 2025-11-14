#!/bin/bash

# Run comprehensive test suite for Awetales Diarization System

echo "Starting Awetales Diarization System Test Suite..."

# Set environment
export TESTING=true
export MODEL_CACHE_DIR=./test_model_cache

# Create test directories
mkdir -p test_model_cache
mkdir -p test_output

echo "Running unit tests..."
pytest tests/core/ -v --cov=src.core --cov-report=html --cov-report=xml

echo "Running API tests..."
pytest tests/api/ -v --cov=src.api --cov-append --cov-report=html --cov-report=xml

echo "Running integration tests..."
pytest tests/integration/ -v --cov=src --cov-append --cov-report=html --cov-report=xml -m "integration"

echo "Running performance tests..."
pytest tests/performance/ -v --cov=src --cov-append --cov-report=html --cov-report=xml -m "benchmark"

echo "Generating coverage report..."
coverage report -m
coverage html

echo "Test suite completed!"