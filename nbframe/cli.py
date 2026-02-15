# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

import json
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .sequence_predictor import (
    predict_kink_probability,
    predict_from_fasta,
    classify_sequence as classify_sequence_api,
    classify_sequences,
)
from .sequence_config import (
    DEFAULT_SEQ_KINKED_THRESHOLD,
    DEFAULT_SEQ_EXTENDED_THRESHOLD,
    UNCERTAINTY_EXPLANATION,
    assign_label,
)
from .structure_config import (
    DEFAULT_STRUCT_KINKED_THRESHOLD,
    DEFAULT_STRUCT_EXTENDED_THRESHOLD,
    STRUCT_UNCERTAINTY_EXPLANATION,
)
from .structure_classifier import (
    classify_all_nanobodies_in_pdb,
    classify_structure as classify_structure_api,
)

app = typer.Typer(
    help="NbFrame: Classify nanobody CDR3 conformation as kinked or extended from sequence and/or structure.",
    add_completion=False,  # Disable --install-completion and --show-completion
)
console = Console()


def _strip_features(obj: Dict[str, object], summary_only: bool) -> Dict[str, object]:
    """Remove 'features' key from result dict when summary_only is True."""
    if not summary_only:
        return obj
    slim = dict(obj)
    slim.pop("features", None)
    return slim


def _get_label_style(label: str) -> tuple[str, str]:
    """Get color styling and symbol for a classification label."""
    if label == "kinked":
        return "[bold green]KINKED[/bold green]", "green"
    elif label == "extended":
        return "[bold blue]EXTENDED[/bold blue]", "blue"
    else:  # uncertain
        return "[bold yellow]UNCERTAIN[/bold yellow]", "yellow"


def _print_single_sequence_prediction(
    label: str,
    probability: float,
    kinked_threshold: float,
    extended_threshold: float,
    verbose: bool = False,
) -> None:
    """
    Print a single sequence prediction with unified format.

    For single predictions, we show more detail by default (verbose-like behavior).
    """
    label_styled, color = _get_label_style(label)

    console.print()
    console.print(f"Prediction:  {label_styled}")
    console.print(f"P(kinked):   [bold]{probability:.4f}[/bold]")
    console.print(f"Thresholds:  kinked >{kinked_threshold:.2f}, extended <{extended_threshold:.2f}")

    if label == "uncertain":
        explanation = UNCERTAINTY_EXPLANATION.format(
            extended=extended_threshold,
            kinked=kinked_threshold,
        )
        console.print(f"\n[dim]Note: {explanation}[/dim]")

    if verbose:
        console.print()  # Extra spacing in verbose mode


def _print_single_structure_prediction(
    chain_id: str,
    label: str,
    probability: float,
    kinked_threshold: float,
    extended_threshold: float,
    verbose: bool = False,
) -> None:
    """
    Print a single structure prediction with unified format.

    For single predictions, we show more detail by default (verbose-like behavior).
    """
    label_styled, color = _get_label_style(label)

    console.print()
    console.print(f"Chain:       {chain_id}")
    console.print(f"Prediction:  {label_styled}")
    console.print(f"P(kinked):   [bold]{probability:.4f}[/bold]")
    console.print(f"Thresholds:  kinked >{kinked_threshold:.2f}, extended <{extended_threshold:.2f}")

    if label == "uncertain":
        explanation = STRUCT_UNCERTAINTY_EXPLANATION.format(
            extended=extended_threshold,
            kinked=kinked_threshold,
        )
        console.print(f"\n[dim]Note: {explanation}[/dim]")

    if verbose:
        console.print()  # Extra spacing in verbose mode


@app.command("classify-sequence")
def classify_sequence_cmd(
    sequence: Optional[str] = typer.Option(None, "--sequence", "-s", help="Single VHH sequence string"),
    fasta: Optional[str] = typer.Option(None, "--fasta", "-f", help="FASTA file containing one or more sequences"),
    fix_cdr1: bool = typer.Option(
        True,
        "--fix-cdr1/--no-fix-cdr1",
        help="Fix common gaps in CDR1 during alignment",
    ),
    align: bool = typer.Option(
        True,
        "--align/--no-align",
        help="Perform AHo alignment (disable if input is already aligned)",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logs to stderr"),
    output_csv: Optional[str] = typer.Option(None, "--output-csv", "-o", help="Save results to CSV file"),
    batch_size: int = typer.Option(100, "--batch-size", help="Batch size for FASTA processing"),
    label: bool = typer.Option(
        True,
        "--label/--no-label",
        help="Show classification label (kinked/extended/uncertain) with probability",
    ),
    kinked_threshold: float = typer.Option(
        DEFAULT_SEQ_KINKED_THRESHOLD,
        "--kinked-threshold",
        help=f"Probability threshold above which to classify as 'kinked' (default: {DEFAULT_SEQ_KINKED_THRESHOLD}).",
    ),
    extended_threshold: float = typer.Option(
        DEFAULT_SEQ_EXTENDED_THRESHOLD,
        "--extended-threshold",
        help=f"Probability threshold below which to classify as 'extended' (default: {DEFAULT_SEQ_EXTENDED_THRESHOLD}).",
    ),
):
    """
    Classify CDR3 conformation from sequence alone.

    Predicts whether a nanobody has a kinked or extended CDR3 based on its
    amino acid sequence. Provide either a single sequence (--sequence) or
    a FASTA file with multiple sequences (--fasta).
    """
    if (sequence is None) == (fasta is None):
        typer.echo("Error: provide exactly one of --sequence or --fasta", err=True)
        raise typer.Exit(code=2)

    if sequence is not None:
        if label:
            # Use classification with labels
            result = classify_sequence_api(
                sequence=sequence,
                kinked_threshold=kinked_threshold,
                extended_threshold=extended_threshold,
                fix_cdr1_gaps=fix_cdr1,
                do_alignment=align,
                verbose=verbose,
            )

            if result.get("error"):
                typer.echo(f"Error: {result['error']}", err=True)
                raise typer.Exit(code=1)

            if result.get("probability") is not None and not output_csv:
                label_str = result.get("label", "unknown")
                prob = result.get("probability")
                _print_single_sequence_prediction(
                    label=label_str,
                    probability=prob,
                    kinked_threshold=kinked_threshold,
                    extended_threshold=extended_threshold,
                    verbose=verbose,
                )
        else:
            # Just probability, no labels
            result = predict_kink_probability(
                sequence=sequence,
                fix_cdr1_gaps=fix_cdr1,
                verbose=verbose,
                do_alignment=align,
            )

            if result.get("error"):
                typer.echo(f"Error: {result['error']}", err=True)
                raise typer.Exit(code=1)

            if result.get("probability") is not None and not output_csv:
                # Raw probability output (no label mode)
                console.print(f"\nP(kinked) = [bold]{result['probability']:.4f}[/bold]")

        if output_csv:
            from .sequence_predictor import save_results_to_csv

            try:
                save_results_to_csv([result], output_csv, names=["Input_Sequence"], do_alignment=align)
                typer.echo(f"Results saved to {output_csv}", err=True)
            except Exception as e:
                typer.echo(f"Error saving CSV file: {e}", err=True)
                raise typer.Exit(code=1)

    else:
        try:
            df = predict_from_fasta(
                fasta,
                fix_cdr1_gaps=fix_cdr1,
                verbose=verbose,
                do_alignment=align,
                batch_size=batch_size,
            )

            # Add labels to the dataframe if requested
            if label:
                df['label'] = df['nbframe_score'].apply(
                    lambda prob: assign_label(prob, kinked_threshold, extended_threshold)[0]
                    if pd.notna(prob) else None
                )

            if output_csv:
                df.to_csv(output_csv, index=False)
                typer.echo(f"Results saved to {output_csv}", err=True)
            else:
                # Render results as a Rich table for preview
                table = Table(title="NbFrame Sequence Predictions", show_lines=False)
                table.add_column("Name", no_wrap=True, style="dim")
                if label:
                    table.add_column("Prediction", justify="center")
                table.add_column("P(kinked)", justify="right")
                if 'error' in df.columns:
                    table.add_column("Error", justify="left", style="red")

                for _, row in df.iterrows():
                    name = str(row['name']) if 'name' in df.columns else ""
                    if 'error' in df.columns and pd.notna(row['error']):
                        if label:
                            table.add_row(name, "-", "-", str(row['error']))
                        else:
                            table.add_row(name, "-", str(row['error']))
                        continue

                    prob = row['nbframe_score']
                    prob_str = f"{prob:.4f}" if pd.notna(prob) else "-"

                    if label:
                        label_str = row.get('label', '-') if pd.notna(row.get('label')) else "-"

                        # Format label with color coding
                        if pd.notna(prob) and label_str != "-":
                            if label_str == "kinked":
                                label_display = "[green]KINKED[/green]"
                            elif label_str == "extended":
                                label_display = "[blue]EXTENDED[/blue]"
                            else:  # uncertain
                                label_display = "[yellow]UNCERTAIN[/yellow]"
                        else:
                            label_display = "-"

                        cells = [name, label_display, prob_str]
                    else:
                        cells = [name, prob_str]

                    if 'error' in df.columns:
                        cells.append("")

                    table.add_row(*cells)

                console.print(table)

                # Print threshold info after table
                console.print(f"\n[dim]Thresholds: kinked >{kinked_threshold:.2f}, extended <{extended_threshold:.2f}[/dim]")
        except Exception as e:
            typer.echo(f"Error processing FASTA file: {e}", err=True)
            raise typer.Exit(code=1)


@app.command("classify-structure")
def classify_structure_cmd(
    pdb: Optional[str] = typer.Option(
        None,
        "--pdb",
        "-p",
        help="Single PDB file to classify",
    ),
    pdb_dir: Optional[str] = typer.Option(
        None,
        "--pdb-dir",
        "-d",
        help="Directory containing multiple PDB files",
    ),
    chain: Optional[str] = typer.Option(
        None,
        "--chain",
        "-c",
        help="Chain ID; auto-detect nanobody chain when omitted.",
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output-json",
        help="Write full JSON result to this file",
    ),
    output_csv: Optional[str] = typer.Option(
        None,
        "--output-csv",
        help="Path to save classification results as CSV",
    ),
    output_aho_dir: Optional[str] = typer.Option(
        None,
        "--output-aho-pdb",
        help="Directory to write AHo-numbered VHH PDBs for classified chains",
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary-only",
        help="Emit minimal outputs (no per-structure features) in JSON/CSV",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        help="When using --pdb-dir, traverse subdirectories recursively for PDB files",
    ),
    progress_interval: int = typer.Option(
        50,
        "--progress-interval",
        help=(
            "How often to print progress during directory classification "
            "(in number of PDBs processed)"
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        help="Verbose logs to stderr",
    ),
    rmsd_threshold: float = typer.Option(
        2.0,
        "--rmsd-threshold",
        help="Maximum framework RMSD (Å) for quality control. Structures above this are filtered out.",
    ),
    no_rmsd_filter: bool = typer.Option(
        False,
        "--no-rmsd-filter",
        help="Disable RMSD-based quality filtering.",
    ),
    kinked_threshold: float = typer.Option(
        DEFAULT_STRUCT_KINKED_THRESHOLD,
        "--kinked-threshold",
        help=f"Probability threshold above which to classify as 'kinked' (default: {DEFAULT_STRUCT_KINKED_THRESHOLD}).",
    ),
    extended_threshold: float = typer.Option(
        DEFAULT_STRUCT_EXTENDED_THRESHOLD,
        "--extended-threshold",
        help=f"Probability threshold below which to classify as 'extended' (default: {DEFAULT_STRUCT_EXTENDED_THRESHOLD}).",
    ),
):
    """
    Classify CDR3 conformation from 3D structure.

    Analyzes nanobody PDB structures to determine CDR3 conformation (kinked or
    extended) based on structural features. Provide either a single PDB file
    (--pdb) or a directory of PDB files (--pdb-dir).

    The classifier uses structural features including dihedral angles, contact
    density, and solvent accessibility to make predictions with high accuracy.
    """
    # Exactly one of --pdb or --pdb-dir must be provided.
    if (pdb is None) == (pdb_dir is None):
        typer.echo("Error: provide exactly one of --pdb or --pdb-dir", err=True)
        raise typer.Exit(code=2)

    if pdb_dir is not None and chain is not None:
        typer.echo(
            "Error: --chain is only supported together with --pdb, "
            "not with --pdb-dir.",
            err=True,
        )
        raise typer.Exit(code=2)

    if recursive and pdb_dir is None:
        typer.echo(
            "Error: --recursive is only valid together with --pdb-dir.", err=True
        )
        raise typer.Exit(code=2)

    all_rows: List[Dict[str, object]] = []
    json_written = False
    csv_written = False

    # Single-PDB mode
    if pdb is not None:
        pdb_path = pdb
        errors_by_chain: Dict[str, str] = {}

        try:
            if chain is not None:
                # Allow comma-separated list of chains, e.g. "--chain A,B,G".
                chain_ids = [c.strip() for c in chain.split(",") if c.strip()]
                results_by_chain: Dict[str, Dict[str, object]] = {}

                filter_rmsd = not no_rmsd_filter
                for cid in chain_ids:
                    try:
                        res = classify_structure_api(
                            pdb_path=pdb_path,
                            chain_id=cid,
                            save_pdb=output_aho_dir is not None,
                            aho_output_dir=output_aho_dir,
                            filter_by_rmsd=filter_rmsd,
                            rmsd_threshold=rmsd_threshold,
                            kinked_threshold=kinked_threshold,
                            extended_threshold=extended_threshold,
                        )
                        if res is not None:
                            results_by_chain[cid] = res
                        else:
                            errors_by_chain[cid] = f"RMSD > {rmsd_threshold} Å threshold"
                    except Exception as exc:  # noqa: BLE001 - report per-chain failure
                        errors_by_chain[cid] = str(exc)

                if not results_by_chain:
                    typer.echo(
                        "Error: none of the requested chains "
                        f"({', '.join(chain_ids)}) could be classified "
                        f"for PDB {pdb_path!r}.",
                        err=True,
                    )
                    raise typer.Exit(code=1)
            else:
                # Classify all detected VHH-like nanobody chains in the PDB,
                # collapsing identical sequences by default.
                filter_rmsd = not no_rmsd_filter
                results_by_chain = classify_all_nanobodies_in_pdb(
                    pdb_path=pdb_path,
                    unique_sequences=True,
                    save_pdb=output_aho_dir is not None,
                    aho_output_dir=output_aho_dir,
                    filter_by_rmsd=filter_rmsd,
                    rmsd_threshold=rmsd_threshold,
                    kinked_threshold=kinked_threshold,
                    extended_threshold=extended_threshold,
                )
                errors_by_chain = {}
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Error during structure classification: {exc}", err=True)
            raise typer.Exit(code=1)

        # Verbose model info (using the first successful result).
        first_result = next(iter(results_by_chain.values()))
        if verbose:
            console.log(
                f"Model: {first_result['model_info'].get('model_file')} "
                f"(trained {first_result['model_info'].get('date_trained')})"
            )
            typer.echo("")  # blank line after model info

        # If requested, write full JSON result to disk for downstream tooling.
        if output_json:
            try:
                if len(results_by_chain) == 1:
                    only_res = next(iter(results_by_chain.values()))
                    payload = _strip_features(only_res, summary_only)
                else:
                    payload = {
                        cid: _strip_features(res, summary_only)
                        for cid, res in results_by_chain.items()
                    }

                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                json_written = True
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"Error writing JSON output: {exc}", err=True)
                raise typer.Exit(code=1)

        # Report any chains that could not be classified.
        for cid, msg in errors_by_chain.items():
            typer.echo(
                f"Chain {cid}: skipped (could not classify: {msg})",
                err=True,
            )

        # Always print a concise, human-readable summary to stdout.
        multi = len(results_by_chain) > 1
        if chain is not None:
            header = "User-provided chains requested for classification:"
        else:
            header = "Chains identified as VHH in the provided PDB:"
        typer.echo(header)

        for cid, result in results_by_chain.items():
            label = result.get("label")
            probs = result.get("probabilities", {})
            p_kink = probs.get("kinked")

            if p_kink is not None:
                if multi:
                    # Multiple chains: use compact format
                    label_styled, color = _get_label_style(label)
                    console.print(f"\n  Chain {cid}:")
                    console.print(f"    Prediction:  {label_styled}")
                    console.print(f"    P(kinked):   [bold]{p_kink:.4f}[/bold]")
                    if label == "uncertain":
                        explanation = STRUCT_UNCERTAINTY_EXPLANATION.format(
                            extended=extended_threshold,
                            kinked=kinked_threshold,
                        )
                        console.print(f"    [dim]Note: {explanation}[/dim]")
                else:
                    # Single chain: use full format
                    _print_single_structure_prediction(
                        chain_id=cid,
                        label=label,
                        probability=p_kink,
                        kinked_threshold=kinked_threshold,
                        extended_threshold=extended_threshold,
                        verbose=verbose,
                    )
            else:
                typer.echo(f"\tChain {cid}: {result}")

        # Print thresholds at the end if multiple chains
        if multi:
            console.print(f"\n[dim]Thresholds: kinked >{kinked_threshold:.2f}, extended <{extended_threshold:.2f}[/dim]")

        # Accumulate rows for CSV output.
        for cid, result in results_by_chain.items():
            label = result.get("label")
            probs = result.get("probabilities", {})
            p_kink = probs.get("kinked")
            p_ext = probs.get("extended")

            row: Dict[str, object] = {
                "pdb_path": pdb_path,
                "pdb_name": Path(pdb_path).name,
                "chain_id": cid,
                "label": label,
                "prob_kinked": p_kink,
                "prob_extended": p_ext,
            }
            # Optionally flatten features into the row.
            if not summary_only:
                features = result.get("features") or {}
                for k, v in features.items():
                    row[f"feature_{k}"] = v
            all_rows.append(row)

    # Directory mode
    else:
        dir_path = Path(pdb_dir or "")
        if not dir_path.is_dir():
            typer.echo(f"Error: {pdb_dir!r} is not a directory.", err=True)
            raise typer.Exit(code=1)

        file_iter = dir_path.rglob("*.pdb") if recursive else dir_path.glob("*.pdb")
        pdb_files = sorted(file_iter)
        if not pdb_files:
            typer.echo(
                f"Error: no .pdb files found in directory {pdb_dir!r}.",
                err=True,
            )
            raise typer.Exit(code=1)

        # Initial summary so the user sees immediate feedback before the first
        # progress or per-PDB output.
        filter_rmsd = not no_rmsd_filter
        typer.echo(
            f"Detected {len(pdb_files)} PDBs in {dir_path}, starting processing..."
        )
        if filter_rmsd:
            typer.echo(
                f"RMSD filtering enabled (threshold: {rmsd_threshold} Å)"
            )

        all_results_for_json: Dict[str, Dict[str, Dict[str, object]]] = {}
        any_success = False
        total_files = len(pdb_files)
        processed = 0
        success_count = 0
        failure_count = 0
        rmsd_filtered_count = 0

        for pdb_file in pdb_files:
            pdb_path = str(pdb_file)
            try:
                results_by_chain = classify_all_nanobodies_in_pdb(
                    pdb_path=pdb_path,
                    unique_sequences=True,
                    save_pdb=output_aho_dir is not None,
                    aho_output_dir=output_aho_dir,
                    filter_by_rmsd=filter_rmsd,
                    rmsd_threshold=rmsd_threshold,
                    kinked_threshold=kinked_threshold,
                    extended_threshold=extended_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                typer.echo(
                    f"[nbframe] WARNING: failed to classify {pdb_path!r}: {exc}",
                    err=True,
                )
                processed += 1
            else:
                processed += 1
                if not results_by_chain:
                    # All chains were filtered (likely by RMSD)
                    rmsd_filtered_count += 1
                    continue

                any_success = True
                success_count += 1
                all_results_for_json[pdb_path] = results_by_chain

                # Only print per-PDB summaries when verbose; otherwise rely on
                # periodic progress updates and CSV/JSON outputs.
                if verbose:
                    multi = len(results_by_chain) > 1
                    console.print(f"\n[bold]{pdb_file.name}[/bold]")
                    for cid, result in results_by_chain.items():
                        label = result.get("label")
                        probs = result.get("probabilities", {})
                        p_kink = probs.get("kinked")

                        if p_kink is not None:
                            label_styled, color = _get_label_style(label)
                            prefix = f"  Chain {cid}: " if multi else "  "
                            console.print(f"{prefix}{label_styled} (P_kinked={p_kink:.4f})")
                        else:
                            typer.echo(f"  Chain {cid}: {result}")

                # Accumulate rows for CSV output regardless of verbosity.
                for cid, result in results_by_chain.items():
                    label = result.get("label")
                    probs = result.get("probabilities", {})
                    p_kink = probs.get("kinked")
                    p_ext = probs.get("extended")

                    row: Dict[str, object] = {
                        "pdb_path": pdb_path,
                        "pdb_name": pdb_file.name,
                        "chain_id": cid,
                        "label": label,
                        "prob_kinked": p_kink,
                        "prob_extended": p_ext,
                    }
                    if not summary_only:
                        features = result.get("features") or {}
                        for k, v in features.items():
                            row[f"feature_{k}"] = v
                    all_rows.append(row)

            # Progress reporting to stderr every progress_interval files.
            if processed % progress_interval == 0 or processed == total_files:
                typer.echo(
                    f"[nbframe] Processed {processed}/{total_files} PDBs "
                    f"(successes={success_count}, rmsd_filtered={rmsd_filtered_count}, "
                    f"failures={failure_count})",
                    err=True,
                )

        if not any_success:
            msg = f"Error: no PDBs in {pdb_dir!r} could be successfully classified."
            if rmsd_filtered_count > 0:
                msg += f" ({rmsd_filtered_count} filtered by RMSD threshold)"
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        # If requested, write aggregated JSON results for the directory.
        if output_json:
            try:
                payload_dir: Dict[str, Dict[str, Dict[str, object]]] = {}
                for path_str, by_chain in all_results_for_json.items():
                    payload_dir[path_str] = {
                        cid: _strip_features(res, summary_only) for cid, res in by_chain.items()
                    }

                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(payload_dir, f, indent=2)
                json_written = True
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"Error writing JSON output: {exc}", err=True)
                raise typer.Exit(code=1)

    # Optional CSV output (both single-PDB and directory modes).
    if output_csv and all_rows:
        try:
            df = pd.DataFrame(all_rows)
            df.to_csv(output_csv, index=False)
            csv_written = True
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Error writing CSV output: {exc}", err=True)
            raise typer.Exit(code=1)

    # Final, tidy summary of output locations.
    outputs: List[str] = []
    if json_written and output_json:
        outputs.append(f"\tStructure classification JSON: {output_json}")
    if csv_written and output_csv:
        outputs.append(f"\tClassification CSV summary: {output_csv}")
    if output_aho_dir:
        outputs.append(f"\tAHo-numbered VHH PDBs: {output_aho_dir}")

    if outputs:
        typer.echo("")
        typer.echo("Outputs:")
        for line in outputs:
            typer.echo(line)


def main() -> None:
    app()