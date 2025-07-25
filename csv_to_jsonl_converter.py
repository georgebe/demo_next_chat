#!/usr/bin/env python3
"""
CSV to JSONL Converter for Amazon Bedrock Evaluation Dataset

This script converts a CSV file with support case data to JSONL format
compatible with Amazon Bedrock evaluation datasets.

CSV Columns:
- case_id: Alphanumeric identifier
- prompt: Multi-line support conversation
- channel: Communication medium (email, call, etc.)
- flag: Default to 'test'
- completion: Support issue classification

Output JSONL format:
{"prompt": "single line prompt", "referenceResponse": "classification"}
"""

import csv
import json
import argparse
import sys
from pathlib import Path


def clean_multiline_text(text):
    """
    Convert multiline text to single line by:
    1. Replacing newlines with spaces
    2. Removing extra whitespace
    3. Preserving conversation structure
    """
    if not text:
        return ""
    
    # Replace various newline characters with space
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra whitespace while preserving single spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def convert_csv_to_jsonl(input_csv_path, output_jsonl_path):
    """
    Convert CSV file to JSONL format for Bedrock evaluation dataset.
    
    Args:
        input_csv_path (str): Path to input CSV file
        output_jsonl_path (str): Path to output JSONL file
    """
    
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
            # Use csv.DictReader to handle CSV parsing
            csv_reader = csv.DictReader(csv_file)
            
            # Verify required columns exist
            required_columns = ['case_id', 'prompt', 'channel', 'flag', 'completion']
            missing_columns = [col for col in required_columns if col not in csv_reader.fieldnames]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            converted_count = 0
            skipped_count = 0
            
            with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
                    try:
                        # Extract and clean the prompt
                        raw_prompt = row['prompt'].strip()
                        if not raw_prompt:
                            print(f"Warning: Empty prompt in row {row_num}, skipping...")
                            skipped_count += 1
                            continue
                        
                        # Clean multiline prompt to single line
                        cleaned_prompt = clean_multiline_text(raw_prompt)
                        
                        # Extract completion (reference response)
                        completion = row['completion'].strip()
                        if not completion:
                            print(f"Warning: Empty completion in row {row_num}, skipping...")
                            skipped_count += 1
                            continue
                        
                        # Create Bedrock evaluation format
                        bedrock_record = {
                            "prompt": cleaned_prompt,
                            "referenceResponse": completion
                        }
                        
                        # Write JSONL record
                        jsonl_file.write(json.dumps(bedrock_record, ensure_ascii=False) + '\n')
                        converted_count += 1
                        
                        # Progress indicator for large files
                        if converted_count % 100 == 0:
                            print(f"Processed {converted_count} records...")
                            
                    except Exception as e:
                        print(f"Error processing row {row_num}: {e}")
                        skipped_count += 1
                        continue
            
            print(f"\nConversion completed!")
            print(f"Successfully converted: {converted_count} records")
            print(f"Skipped records: {skipped_count}")
            print(f"Output file: {output_jsonl_path}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def validate_jsonl_output(jsonl_path, sample_size=5):
    """
    Validate the generated JSONL file and show sample records.
    
    Args:
        jsonl_path (str): Path to JSONL file
        sample_size (int): Number of sample records to display
    """
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        print(f"\nValidation Results:")
        print(f"Total records in JSONL: {len(lines)}")
        
        # Show sample records
        print(f"\nSample records (first {min(sample_size, len(lines))}):")
        print("-" * 80)
        
        for i, line in enumerate(lines[:sample_size]):
            try:
                record = json.loads(line.strip())
                print(f"Record {i+1}:")
                print(f"  Prompt: {record['prompt'][:100]}{'...' if len(record['prompt']) > 100 else ''}")
                print(f"  Reference Response: {record['referenceResponse']}")
                print()
            except json.JSONDecodeError as e:
                print(f"  Invalid JSON in line {i+1}: {e}")
                
    except Exception as e:
        print(f"Validation error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV support case data to JSONL format for Amazon Bedrock evaluation dataset"
    )
    parser.add_argument(
        "input_csv", 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Path to output JSONL file (default: input_filename.jsonl)",
        default=None
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate output JSONL file and show sample records"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=5,
        help="Number of sample records to show during validation (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_path = Path(args.input_csv)
        args.output = input_path.with_suffix('.jsonl')
    
    print(f"Converting CSV to JSONL...")
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    # Convert CSV to JSONL
    convert_csv_to_jsonl(args.input_csv, args.output)
    
    # Validate output if requested
    if args.validate:
        validate_jsonl_output(args.output, args.sample_size)


if __name__ == "__main__":
    main()
