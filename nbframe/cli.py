# (c) 2026 Mateusz Jaskolowski
# Developed at Sormanni Lab at University of Cambridge
# ============================================================================

import json
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .sequence_predictor import (
    predict_kink_probability,
    predict_from_fasta,
    classify_sequence as classify_sequence_api,
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
    resolve_structure_thresholds,
)

from .validation import validate_thresholds, validate_rmsd_threshold

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
    if "sequence_group" in slim:
        slim["sequence_group"] = dict(slim["sequence_group"])
        slim["sequence_group"]["results"] = [_strip_features(r, True) for r in slim["sequence_group"]["results"]]
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
    batch_size: int = typer.Option(100, "--batch-size", min=1, help="Batch size for FASTA processing"),
    ncpu: Optional[int] = typer.Option(None, "--ncpu", min=1, help="Maximum ANARCI workers for FASTA processing"),
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

    try:
        kinked_threshold, extended_threshold = validate_thresholds(kinked_threshold, extended_threshold)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
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

        if result.get("probability") is None:
            raise typer.Exit(code=1)

    else:
        try:
            df = predict_from_fasta(
                fasta,
                fix_cdr1_gaps=fix_cdr1,
                verbose=verbose,
                do_alignment=align,
                batch_size=batch_size,
                ncpu=ncpu,
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
            if not df["nbframe_score"].notna().any():
                typer.echo("No predictions were produced; input error reports were retained.", err=True)
                raise typer.Exit(code=1)
        except typer.Exit:
            raise
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
        help="Directory for AHo-numbered structures (mmCIF for multi-character chain IDs)",
    ),
    unique_sequences: bool = typer.Option(False, "--unique-sequences", help="Group identical sequences after classifying every copy; retain all copy results."),
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
        min=1,
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
    kinked_threshold: Optional[float] = typer.Option(
        None,
        "--kinked-threshold",
        help=f"Probability threshold above which to classify as 'kinked' (default: {DEFAULT_STRUCT_KINKED_THRESHOLD}).",
    ),
    extended_threshold: Optional[float] = typer.Option(
        None,
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

    try:
        kinked_threshold, extended_threshold = resolve_structure_thresholds(kinked_threshold, extended_threshold)
        validate_rmsd_threshold(rmsd_threshold)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2)

    chain_ids = None if chain is None else [c.strip() for c in chain.split(",") if c.strip()]
    if chain_ids == []:
        typer.echo("Error: --chain must contain at least one chain ID.", err=True)
        raise typer.Exit(code=2)
    if pdb is not None:
        files = [Path(pdb)]
    else:
        directory = Path(pdb_dir)
        if not directory.is_dir():
            typer.echo(f"Error: {pdb_dir!r} is not a directory.", err=True)
            raise typer.Exit(code=1)
        candidates = directory.rglob("*") if recursive else directory.glob("*")
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in {".pdb", ".ent", ".cif", ".mmcif"})
        if not files:
            typer.echo(f"Error: no PDB/mmCIF files found in directory {pdb_dir!r}.", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Detected {len(files)} structures in {directory}, starting processing...")

    all_results = {}
    all_rows = []
    classified_count = success_count = failure_count = filtered_count = 0
    for processed, path in enumerate(files, 1):
        results = classify_all_nanobodies_in_pdb(
            str(path), chain_ids=chain_ids, unique_sequences=unique_sequences,
            strict=False, save_pdb=output_aho_dir is not None, aho_output_dir=output_aho_dir,
            filter_by_rmsd=not no_rmsd_filter, rmsd_threshold=rmsd_threshold,
            kinked_threshold=kinked_threshold, extended_threshold=extended_threshold)
        all_results[str(path)] = {cid: _strip_features(r, summary_only) for cid, r in results.items()}
        statuses = [r["status"] for r in results.values()]
        success_count += int("classified" in statuses)
        filtered_count += int(all(status == "filtered" for status in statuses))
        failure_count += int("classified" not in statuses and not all(status == "filtered" for status in statuses))
        for result in results.values():
            cid = result["chain_id_used"]
            status = result["status"]
            classified_count += int(status == "classified")
            if status != "classified":
                typer.echo(f"{path.name}, chain {cid or '(undetermined)'}: prediction withheld ({status}). {result.get('error')}", err=True)
            elif pdb is not None or verbose:
                _print_single_structure_prediction(cid, result["label"], result["prob_kinked"],
                    kinked_threshold, extended_threshold, verbose)
            if result.get("sequence_group", {}).get("label_disagreement"):
                typer.echo(f"{path.name}, chain group {result['sequence_group']['chain_ids']}: copies have different labels; inspect all copy results.", err=True)
            if verbose:
                console.log(f"Model: {result['model_info']['model_file']} (trained {result['model_info']['date_trained']})")
            row = dict(pdb_path=str(path), pdb_name=path.name, chain_id=cid,
                label=result["label"], prob_kinked=result["prob_kinked"], prob_extended=result["prob_extended"],
                status=status, error=result.get("error"), model_id=result.get("model_id_used"),
                warnings=json.dumps(result.get("warnings", [])), quality=json.dumps(result.get("quality", {})))
            if unique_sequences:
                row["sequence_group"] = json.dumps(_strip_features(result, summary_only).get("sequence_group"))
            if not summary_only:
                row.update({f"feature_{k}": v for k, v in result.get("features", {}).items()})
            all_rows.append(row)
        if pdb_dir and (processed % progress_interval == 0 or processed == len(files)):
            typer.echo(f"[nbframe] Processed {processed}/{len(files)} structures (successes={success_count}, rmsd_filtered={filtered_count}, failures={failure_count})", err=True)

    try:
        if output_json:
            payload = all_results
            if pdb is not None:
                payload = next(iter(all_results.values()))
                if len(payload) == 1:
                    payload = next(iter(payload.values()))
            Path(output_json).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            typer.echo(f"Structure classification JSON: {output_json}", err=True)
        if output_csv:
            pd.DataFrame(all_rows).to_csv(output_csv, index=False)
            typer.echo(f"Classification CSV summary: {output_csv}", err=True)
        if output_aho_dir:
            typer.echo(f"AHo-numbered VHH structures: {output_aho_dir}", err=True)
    except Exception as exc:
        typer.echo(f"Error writing output: {exc}", err=True)
        raise typer.Exit(code=1)
    if classified_count == 0:
        typer.echo("No predictions were produced; input and structural quality reports were retained.", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()
