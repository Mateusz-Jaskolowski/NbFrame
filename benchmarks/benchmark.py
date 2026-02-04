#!/usr/bin/env python3
# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

"""
NbFrame performance benchmarking module.

This module provides functions and a CLI to test different combinations of
batch sizes and worker counts to determine the optimal configuration for
processing sequences with NbFrame's batch prediction API.
"""

import time
import numpy as np
import pandas as pd
import gc
import json
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
from rich.console import Console
from rich.table import Table

from nbframe import predict_kink_probabilities

def process_batch(batch_sequences, batch_size):
    """Process a single batch of sequences."""
    try:
        # Process the batch using the high‑throughput batch prediction API
        batch_results = predict_kink_probabilities(
            sequences=batch_sequences,
            do_alignment=False,  # Skip alignment since sequences are already aligned
            verbose=False,  # Disable verbose output for benchmarking
            batch_size=batch_size  # Use the batch size parameter
        )
        return batch_results
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def run_benchmark(sequences, batch_size, num_workers, test_name="benchmark", benchmark_sample_size=50000):
    """Run a single benchmark with the given batch size and worker count"""
    print(f"\n=== Running benchmark: batch_size={batch_size}, workers={num_workers} ===")

    # Use a consistent sample size for all benchmarks
    # If sequences are already a sample, use them all
    if len(sequences) > benchmark_sample_size:
        # Use a fixed random seed for reproducibility
        # Note: We're regenerating the sample for each benchmark to avoid memory issues
        # with very large sequence datasets, but using the same random seed ensures
        # we get the same sequences each time
        import random
        random.seed(42)
        benchmark_indices = random.sample(range(len(sequences)), benchmark_sample_size)
        benchmark_sequences = [sequences[i] for i in benchmark_indices]
        print(f"Using a random sample of {benchmark_sample_size:,} sequences for this benchmark")
    else:
        benchmark_sequences = sequences
        print(f"Using all {len(sequences):,} sequences for this benchmark")

    # Calculate the number of batches
    total_sequences = len(benchmark_sequences)
    num_batches = int(np.ceil(total_sequences / batch_size))

    results = []
    processing_times = []

    start_time = time.time()

    # For single worker, process sequentially
    if num_workers == 1:
        # Process all batches with Rich progress
        with Progress(
            SpinnerColumn(),
            "[bold blue]Benchmark",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task(f"{test_name}", total=num_batches)
            for batch_idx in range(num_batches):
                # Calculate batch indices
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_sequences)

                # Get the current batch of sequences
                batch_sequences = benchmark_sequences[start_idx:end_idx]
                batch_size_actual = len(batch_sequences)

                # Process the batch using high‑throughput batch prediction
                batch_start_time = time.time()
                batch_results = predict_kink_probabilities(
                    sequences=batch_sequences,
                    do_alignment=False,  # Skip alignment since sequences are already aligned
                    verbose=False,  # Disable verbose output for benchmarking
                    batch_size=batch_size  # Use the batch size parameter
                )
                batch_time = time.time() - batch_start_time

                results.extend(batch_results)
                processing_times.append(batch_time)

                # Calculate and display current speed
                sequences_per_second = batch_size_actual / batch_time if batch_time > 0 else 0.0
                progress.console.print(f"Batch {batch_idx+1}/{num_batches}: {sequences_per_second:.2f} sequences/second")
                progress.advance(task_id, 1)

                # Force garbage collection after each batch
                gc.collect()
    else:
        # For multiple workers, use ProcessPoolExecutor to process batches in parallel
        batch_indexes = []
        batch_data = []

        # Prepare all batch data
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_sequences)
            batch_sequences = benchmark_sequences[start_idx:end_idx]
            batch_indexes.append((start_idx, end_idx))
            batch_data.append(batch_sequences)

        batch_start_time = time.time()

        # Use ProcessPoolExecutor to process batches in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            batch_futures = [executor.submit(process_batch, batch, batch_size) for batch in batch_data]

            with Progress(
                SpinnerColumn(),
                "[bold blue]Benchmark",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task(f"{test_name}", total=len(batch_futures))

                # Collect results as they complete
                for i, future in enumerate(as_completed(batch_futures)):
                    batch_result = future.result()
                    results.extend(batch_result)
                    progress.advance(task_id, 1)

                    # Calculate batch time (approximate since batches run in parallel)
                    if i == len(batch_futures) - 1:
                        total_batch_time = time.time() - batch_start_time
                        processing_times.append(total_batch_time)

        # Force garbage collection
        gc.collect()

    total_time = time.time() - start_time
    sequences_processed = len(results)

    # For multiple workers, calculate the total processing time differently
    if num_workers == 1:
        total_processing_time = sum(processing_times)
    else:
        total_processing_time = processing_times[0] if processing_times else 0

    # Calculate average speed
    avg_speed = sequences_processed / total_time if total_time > 0 else 0

    # Calculate throughput metrics
    benchmark_result = {
        "batch_size": batch_size,
        "workers": num_workers,
        "sequences_processed": sequences_processed,
        "total_processing_time": total_processing_time,
        "total_wall_time": total_time,
        "sequences_per_second": avg_speed,
        "speed_per_worker": avg_speed / num_workers if num_workers > 0 else 0,
        "overhead_time": total_time - total_processing_time,
        "overhead_percentage": ((total_time - total_processing_time) / total_time) * 100 if total_time > 0 else 0
    }

    console = Console()
    table = Table(title=f"Benchmark Results: {test_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Workers", str(num_workers))
    table.add_row("Sequences Processed", f"{sequences_processed:,}")
    table.add_row("Average Speed", f"{avg_speed:.2f} seq/s")
    table.add_row("Speed per Worker", f"{(avg_speed / num_workers):.2f} seq/s/worker")
    table.add_row("Overhead", f"{benchmark_result['overhead_percentage']:.2f}%")
    console.print(table)

    return benchmark_result

def run_benchmarks_from_config(
    input_path: str,
    sample_size: int,
    benchmark_size: int,
    batch_sizes: list[int],
    worker_counts_opt: str,
    output_json: str,
    prioritize: str,
):
    """
    Run a grid of benchmark configurations using the provided settings.

    Designed to be called from the Typer CLI and from Python code.
    Assumes ``batch_sizes`` is already a list of positive integers.
    """

    # Basic validation of batch sizes
    if not batch_sizes:
        print("Error: at least one batch size must be provided")
        return
    if not all(isinstance(size, int) and size > 0 for size in batch_sizes):
        print("Error: batch_sizes must be a list of positive integers")
        return

    # Process worker counts
    if worker_counts_opt.lower() == 'auto':
        # Auto-generate worker counts based on system CPUs
        max_cpus = multiprocessing.cpu_count()
        if max_cpus <= 4:
            worker_counts = [1, max_cpus]
        elif max_cpus <= 8:
            worker_counts = [1, max_cpus//2, max_cpus]
        else:
            # Test more combinations on higher-core systems
            worker_counts = [1, 2, 4, max_cpus//2, max_cpus-2, max_cpus]
            # Remove duplicates and sort
            worker_counts = sorted(list(set(worker_counts)))
    else:
        try:
            worker_counts = [int(count) for count in worker_counts_opt.split(',')]
        except ValueError:
            print("Error: worker_counts must be comma-separated integers or 'auto'")
            return

    # Sort worker counts in descending order to start with highest parallelism
    worker_counts = sorted(worker_counts, reverse=True)
    print(f"Testing worker counts in descending order: {worker_counts}")

    # Load the sample data
    print(f"Loading sample data from {input_path}...")
    try:
        # Read only the sample_size rows
        sequence_database = pd.read_parquet(input_path)

        if sample_size < len(sequence_database):
            # Take a random sample for more representative benchmarking
            sequence_database = sequence_database.sample(n=sample_size, random_state=42)
            print(f"Random sample of {sample_size:,} sequences selected for benchmarking")
        else:
            print(f"Using all {len(sequence_database):,} available sequences for benchmarking")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    # Validate that required column exists
    if 'aligned_sequence' not in sequence_database.columns:
        print("Error: Column 'aligned_sequence' not found in database")
        return

    # Get sequences as list for processing
    sequences = sequence_database['aligned_sequence'].tolist()
    print(f"Loaded {len(sequences):,} sequences for benchmarking")

    # Run all combinations of batch sizes and worker counts
    all_results = []

    # Create a timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # First, run a warmup to load models and data into memory
    print("\n=== Running warmup ===")
    warmup_batch = sequences[:min(1000, len(sequences))]
    _ = predict_kink_probabilities(
        sequences=warmup_batch,
        do_alignment=False,
        verbose=False,
    )
    print("Warmup complete")

    # Run all combinations of batch sizes and worker counts
    if prioritize == 'batches':
        # First loop through batch sizes, then for each batch size, test all worker counts
        print("\n>>> Prioritizing batch sizes (testing all worker counts for each batch size) <<<")
        for batch_size in batch_sizes:
            print(f"\n>>> Testing batch size: {batch_size} <<<\n")
            for num_workers in worker_counts:
                test_name = f"b{batch_size}_w{num_workers}"
                try:
                    result = run_benchmark(sequences, batch_size, num_workers, test_name, benchmark_size)
                    result["timestamp"] = timestamp
                    all_results.append(result)

                    # Save incremental results after each benchmark
                    with open(output_json, 'w') as f:
                        json.dump(all_results, f, indent=2)
                except Exception as e:
                    print(f"Error during benchmark {test_name}: {e}")
    else:
        # First loop through worker counts, then for each worker count, test all batch sizes
        print("\n>>> Prioritizing worker counts (testing all batch sizes for each worker count) <<<")
        for num_workers in worker_counts:
            print(f"\n>>> Testing worker count: {num_workers} <<<\n")
            for batch_size in batch_sizes:
                test_name = f"b{batch_size}_w{num_workers}"
                try:
                    result = run_benchmark(sequences, batch_size, num_workers, test_name, benchmark_size)
                    result["timestamp"] = timestamp
                    all_results.append(result)

                    # Save incremental results after each benchmark
                    with open(output_json, 'w') as f:
                        json.dump(all_results, f, indent=2)
                except Exception as e:
                    print(f"Error during benchmark {test_name}: {e}")

    # Find the best configuration
    if all_results:
        # Sort by sequences per second (highest first)
        sorted_results = sorted(all_results, key=lambda x: x["sequences_per_second"], reverse=True)
        best_result = sorted_results[0]

        summary_console = Console()
        summary_console.print("\n[bold]BENCHMARK SUMMARY[/]")
        best_table = Table(title="Best Configuration")
        best_table.add_column("Batch Size")
        best_table.add_column("Workers")
        best_table.add_column("Speed (seq/s)")
        best_table.add_row(str(best_result['batch_size']), str(best_result['workers']), f"{best_result['sequences_per_second']:.2f}")
        summary_console.print(best_table)

        top_table = Table(title="Top 3 Configurations")
        top_table.add_column("#")
        top_table.add_column("Batch Size")
        top_table.add_column("Workers")
        top_table.add_column("Speed (seq/s)")
        for i, result in enumerate(sorted_results[:min(3, len(sorted_results))]):
            top_table.add_row(str(i+1), str(result['batch_size']), str(result['workers']), f"{result['sequences_per_second']:.2f}")
        summary_console.print(top_table)

        total_dataset_size = len(sequence_database)
        estimated_time_hours = total_dataset_size / best_result['sequences_per_second'] / 3600
        summary_console.print(f"\nWith the best configuration, processing all {total_dataset_size:,} sequences would take ~{estimated_time_hours:.2f} hours")

        full_dataset_size = 22_000_000
        estimated_full_time_hours = full_dataset_size / best_result['sequences_per_second'] / 3600
        summary_console.print(f"Estimated time for 22 million sequences: {estimated_full_time_hours:.2f} hours")

        summary_console.print(f"\nFull benchmark results saved to {output_json}")
    else:
        print("No successful benchmarks completed")

    return all_results