"""
Simple Model Performance Analysis Script for ParamBench Benchmark

This script analyzes the performance of various models on the ParamBench dataset,
generating only CSV outputs for accuracy, response time, and subject/type-wise metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHECKPOINTS_PATH = Path("checkpoints")
RESULTS_PATH = Path("results")
DATASET_CSV_PATH = Path("data/full-data.csv")

# Create results directory
RESULTS_PATH.mkdir(exist_ok=True)


def load_question_type_mapping():
    """Load question type mapping from dataset CSV"""
    logger.info("Loading question type mapping from CSV...")
    
    try:
        df_dataset = pd.read_csv(DATASET_CSV_PATH, encoding='utf-8')
        question_type_map = dict(zip(df_dataset['unique_question_id'], df_dataset['question_type']))
        
        logger.info(f"Loaded {len(question_type_map)} question type mappings")
        unique_types = df_dataset['question_type'].unique()
        logger.info(f"Available question types: {list(unique_types)}")
        
        return question_type_map
        
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        logger.warning("Type-wise analysis will be skipped.")
        return {}


def load_model_checkpoints():
    """Load all model checkpoint JSON files"""
    logger.info("Loading model checkpoints...")
    
    all_model_data = {}
    checkpoint_files = list(CHECKPOINTS_PATH.glob("*.json"))
    
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                model_name = data.get('model_name', checkpoint_file.stem)
                all_model_data[model_name] = data
                logger.debug(f"Loaded checkpoint: {model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to load {checkpoint_file}: {e}")
    
    logger.info(f"Loaded {len(all_model_data)} model checkpoints")
    return all_model_data


def calculate_model_metrics(model_data):
    """Calculate overall metrics for a model"""
    responses = model_data.get("responses", {})
    timestamps = model_data.get("timestamps", {})
    gen_stats = model_data.get("generation_stats", {})
    
    # Basic accuracy metrics
    correct_count = sum(r.get("is_correct", False) for r in responses.values())
    total_count = len(responses)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Time metrics
    avg_time = float(np.mean(list(timestamps.values()))) if timestamps else 0.0
    
    # Token metrics
    token_counts = [s.get("generated_tokens", 0) for s in gen_stats.values()]
    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "avg_time": avg_time,
        "avg_tokens": avg_tokens
    }


def process_model_summaries(all_model_data):
    """Process overall model summaries"""
    logger.info("Processing model summaries...")
    
    results_rows = []
    
    for model_name, model_data in all_model_data.items():
        metrics = calculate_model_metrics(model_data)
        
        results_rows.append({
            "Model": model_name,
            "Accuracy": round(metrics['accuracy'] * 100, 2),
            "Correct_Answers": metrics['correct_count'],
            "Total_Questions": metrics['total_count'],
            "Avg_Response_Time_Seconds": round(metrics['avg_time'], 2),
            "Avg_Tokens_Generated": round(metrics['avg_tokens'], 2),
        })
    
    df_results = pd.DataFrame(results_rows)
    df_results = df_results.sort_values("Accuracy", ascending=False)
    
    # Save summary CSV
    summary_csv_path = RESULTS_PATH / "model_performance_summary.csv"
    df_results.to_csv(summary_csv_path, index=False, encoding="utf-8")
    logger.info(f"Summary CSV saved at: {summary_csv_path}")
    
    return df_results


def process_type_wise_analysis(all_model_data, question_type_map):
    """Process question type-wise analysis"""
    if not question_type_map:
        logger.warning("Skipping type-wise analysis - no question type mapping available")
        return
    
    logger.info("Processing type-wise analysis...")
    
    combined_types = defaultdict(dict)
    
    for model_name, model_data in all_model_data.items():
        responses = model_data.get("responses", {})
        type_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for question_id, response in responses.items():
            question_type = question_type_map.get(question_id, "Unknown")
            type_metrics[question_type]['total'] += 1
            if response.get("is_correct", False):
                type_metrics[question_type]['correct'] += 1
        
        # Calculate accuracy percentages
        for q_type, metrics in type_metrics.items():
            if metrics['total'] > 0:
                accuracy = metrics['correct'] / metrics['total'] * 100
                combined_types[q_type][model_name] = round(accuracy, 1)
    
    # Create combined table
    df_combined_types = pd.DataFrame.from_dict(combined_types, orient="index").fillna(0)
    df_combined_types.index.name = "Question_Type"
    df_combined_types = df_combined_types.reset_index().sort_values("Question_Type")
    
    # Save combined results
    combined_csv_path = RESULTS_PATH / "type_wise_accuracy_all_models.csv"
    df_combined_types.to_csv(combined_csv_path, index=False, encoding="utf-8")
    logger.info(f"Type-wise CSV saved at: {combined_csv_path}")


def process_subject_wise_analysis(all_model_data):
    """Process subject-wise analysis"""
    logger.info("Processing subject-wise analysis...")
    
    combined_subjects = defaultdict(dict)
    
    for model_name, model_data in all_model_data.items():
        responses = model_data.get("responses", {})
        subject_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for response in responses.values():
            subject = response.get("subject", "Unknown")
            subject_metrics[subject]['total'] += 1
            if response.get("is_correct", False):
                subject_metrics[subject]['correct'] += 1
        
        # Calculate accuracy percentages
        for subject, metrics in subject_metrics.items():
            if metrics['total'] > 0:
                accuracy = metrics['correct'] / metrics['total'] * 100
                combined_subjects[subject][model_name] = round(accuracy, 1)
    
    # Create combined table
    df_combined = pd.DataFrame.from_dict(combined_subjects, orient="index").fillna(0)
    df_combined.index.name = "Subject"
    df_combined = df_combined.reset_index().sort_values("Subject")
    
    # Save combined results
    combined_csv_path = RESULTS_PATH / "subject_wise_accuracy_all_models.csv"
    df_combined.to_csv(combined_csv_path, index=False, encoding="utf-8")
    logger.info(f"Subject-wise CSV saved at: {combined_csv_path}")


def main():
    """Main entry point for the analysis script"""
    try:
        logger.info("Starting model performance analysis...")
        
        # Load data
        question_type_map = load_question_type_mapping()
        all_model_data = load_model_checkpoints()
        
        if not all_model_data:
            logger.error("No model checkpoints found!")
            return
        
        # Process and save summaries
        model_summaries = process_model_summaries(all_model_data)
        
        # Process category-wise metrics
        process_type_wise_analysis(all_model_data, question_type_map)
        process_subject_wise_analysis(all_model_data)
        
        logger.info("\n" + "="*50)
        logger.info("Analysis Summary:")
        logger.info("="*50)
        logger.info(f"✓ Processed {len(all_model_data)} models")
        logger.info(f"✓ Generated model performance summary")
        logger.info(f"✓ Generated aggregated type-wise analysis")
        logger.info(f"✓ Generated aggregated subject-wise analysis")
        logger.info(f"✓ All CSV files saved to: {RESULTS_PATH}")
        logger.info("="*50)
        
        # Print top 5 models by accuracy
        print("\nTop 5 Models by Accuracy:")
        print(model_summaries.head().to_string(index=False))
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
