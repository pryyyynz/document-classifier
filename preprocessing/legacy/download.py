#!/usr/bin/env python3
"""
Download SEC documents and organize them by type, removing successful downloads from filtered.jsonl.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import subprocess
import sys
import os


SEC_BASE = "https://www.sec.gov/Archives/edgar/data"


def parse_source(src: str) -> Optional[Tuple[int, str, str]]:
    """Parse source path to get CIK, accession, and filename."""
    parts = src.strip("/").split("/")
    if len(parts) < 4:
        return None
    accession = parts[-2]  # e.g. 000146715419000004
    filename = parts[-1]
    if len(accession) < 18:
        # Some feeds still work; continue but may fail
        pass
    cik_str = accession[:10]
    try:
        # drop leading zeros as EDGAR paths use non-padded CIK
        cik = int(cik_str)
    except ValueError:
        return None
    return cik, accession, filename


def build_url(cik: int, accession: str, filename: str) -> str:
    """Build the SEC URL for a file."""
    return f"{SEC_BASE}/{cik}/{accession}/{filename}"


def get_headers(user_agent: str) -> dict:
    """Get headers for SEC requests."""
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "identity",
        "Connection": "close",
    }


def try_download_html(url: str, headers: dict) -> Optional[Tuple[str, bytes]]:
    """Try to download HTML content."""
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.text, r.content
    except Exception:
        pass
    return None


def try_fallback_listing(cik: int, accession: str, target_filename: str, headers: dict) -> Optional[Tuple[str, bytes]]:
    """Query the index.json for the filing to find actual filenames."""
    idx_url = f"{SEC_BASE}/{cik}/{accession}/index.json"
    try:
        r = requests.get(idx_url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        files = data.get("directory", {}).get("item", [])

        # Case-insensitive exact match
        tname_lower = target_filename.lower()
        for it in files:
            name = it.get("name", "")
            if name.lower() == tname_lower:
                file_url = f"{SEC_BASE}/{cik}/{accession}/{name}"
                fr = requests.get(file_url, headers=headers, timeout=30)
                if fr.status_code == 200:
                    return name, fr.content

        # Heuristic fallback: first HTML-like file if exact not found
        for it in files:
            name = it.get("name", "")
            if name.lower().endswith((".htm", ".html", ".txt")):
                file_url = f"{SEC_BASE}/{cik}/{accession}/{name}"
                fr = requests.get(file_url, headers=headers, timeout=30)
                if fr.status_code == 200:
                    return name, fr.content
    except Exception:
        pass
    return None


def convert_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Convert HTML file to PDF using WeasyPrint."""
    try:
        # Import WeasyPrint
        from weasyprint import HTML

        # Convert HTML to PDF
        html_doc = HTML(filename=str(html_path))
        html_doc.write_pdf(str(pdf_path))

        return pdf_path.exists()

    except ImportError:
        print(
            f"    ⚠ WeasyPrint not available, skipping PDF conversion for {html_path.name}")
        return False
    except Exception as e:
        print(f"    ⚠ PDF conversion failed for {html_path.name}: {e}")
        return False


def save_file(content: bytes, file_path: Path, prefer_pdf: bool = True) -> bool:
    """Save file content, optionally converting HTML to PDF."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if prefer_pdf and file_path.suffix.lower() in ['.htm', '.html']:
            # Try to convert HTML to PDF
            html_content = content.decode('utf-8', errors='ignore')
            pdf_path = file_path.with_suffix('.pdf')

            if convert_html_to_pdf(file_path, pdf_path):
                # Save both HTML and PDF
                file_path.write_bytes(content)
                return True
            else:
                # PDF conversion failed, just save HTML
                file_path.write_bytes(content)
                return True
        else:
            # Save as-is
            file_path.write_bytes(content)
            return True

    except Exception:
        return False


def create_type_directories(base_dir: Path, matched_types: List[str]) -> List[Path]:
    """Create directories for each matched type and return the paths."""
    type_dirs = []
    for matched_type in matched_types:
        type_dir = base_dir / matched_type
        type_dir.mkdir(parents=True, exist_ok=True)
        type_dirs.append(type_dir)
    return type_dirs


def save_to_type_folders(line_data: dict, type_dirs: List[Path], line_number: int) -> None:
    """Save the line data to each type folder."""
    for type_dir in type_dirs:
        # Create a unique filename using line number and source
        source = line_data.get("source", "unknown")
        safe_source = source.replace("/", "_").replace("\\", "_")
        filename = f"{line_number:06d}_{safe_source}.json"
        file_path = type_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(line_data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Download SEC documents and organize by type, removing successful downloads from filtered.jsonl"
    )
    ap.add_argument("--input", default="filtered.jsonl",
                    help="Path to filtered.jsonl")
    ap.add_argument("--out", default="downloads",
                    help="Output directory for downloaded files")
    ap.add_argument("--user-agent", required=True,
                    help="SEC-compliant User-Agent with contact info")
    ap.add_argument("--rps", type=float, default=3.0,
                    help="Requests per second (throttle)")
    ap.add_argument("--prefer-pdf", action="store_true",
                    help="Prefer PDF over HTML when possible")
    ap.add_argument("--type-output", default="processed_downloads",
                    help="Directory for type-organized JSON files")

    args = ap.parse_args()

    in_path = Path(args.input)
    # Resolve outputs relative to project root (parent of preprocessing)
    base_dir = Path(__file__).parent.parent
    out_dir = (
        base_dir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    type_output_dir = (base_dir / args.type_output) if not Path(
        args.type_output).is_absolute() else Path(args.type_output)
    ua = args.user_agent
    delay = 1.0 / max(0.1, args.rps)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    headers = get_headers(ua)

    # Create output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    type_output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total_lines": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "skipped_lines": 0,
        "processed_types": set()
    }

    # Read all lines and process them
    lines_to_keep = []
    lines_to_process = []

    print("Reading filtered.jsonl...")
    with in_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            stats["total_lines"] += 1

            try:
                line_data = json.loads(line.strip())
            except json.JSONDecodeError:
                # Keep malformed lines
                lines_to_keep.append(line)
                stats["skipped_lines"] += 1
                continue

            source = line_data.get("source")
            if not source:
                # Keep lines without source
                lines_to_keep.append(line)
                stats["skipped_lines"] += 1
                continue

            # Parse source to get file location
            parsed = parse_source(source)
            if not parsed:
                # Keep lines with invalid source format
                lines_to_keep.append(line)
                stats["skipped_lines"] += 1
                continue

            cik, accession, filename = parsed
            lines_to_process.append(
                (line_num, line_data, cik, accession, filename))

    print(f"Processing {len(lines_to_process)} valid lines...")

    # Process downloads
    for line_num, line_data, cik, accession, filename in lines_to_process:
        print(f"Processing line {line_num}: {filename}")

        url = build_url(cik, accession, filename)
        dest = out_dir / str(cik) / accession / filename

        # Try direct download first
        download_success = False
        downloaded_filename = filename
        downloaded_content = None

        # Try to download the original file
        html_result = try_download_html(url, headers)
        if html_result:
            html_text, html_content = html_result
            if save_file(html_content, dest, args.prefer_pdf):
                download_success = True
                downloaded_content = html_content

        # Try fallback if direct download failed
        if not download_success:
            fallback_result = try_fallback_listing(
                cik, accession, filename, headers)
            if fallback_result:
                fallback_name, fallback_content = fallback_result
                fallback_dest = out_dir / str(cik) / accession / fallback_name
                if save_file(fallback_content, fallback_dest, args.prefer_pdf):
                    download_success = True
                    downloaded_filename = fallback_name
                    downloaded_content = fallback_content

        if download_success:
            stats["successful_downloads"] += 1
            print(f"  ✓ Downloaded: {downloaded_filename}")

            # Save to type-organized folders
            matched_types = line_data.get("_matched_types", [])
            if matched_types:
                type_dirs = create_type_directories(
                    type_output_dir, matched_types)
                save_to_type_folders(line_data, type_dirs, line_num)
                stats["processed_types"].update(matched_types)
        else:
            stats["failed_downloads"] += 1
            print(f"  ✗ Failed: {filename}")
            # Keep failed downloads in filtered.jsonl
            lines_to_keep.append(line)

        # Rate limiting
        time.sleep(delay)

    # Write back filtered.jsonl with only failed downloads
    print("Updating filtered.jsonl...")
    with in_path.open("w", encoding="utf-8") as f:
        for line in lines_to_keep:
            f.write(line)

    # Convert set to list for display
    stats["processed_types"] = list(stats["processed_types"])

    # Print final statistics
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Successful downloads: {stats['successful_downloads']}")
    print(f"Failed downloads: {stats['failed_downloads']}")
    print(f"Skipped lines: {stats['skipped_lines']}")
    print(f"Types processed: {', '.join(sorted(stats['processed_types']))}")
    print(f"Downloads directory: {out_dir.resolve()}")
    print(f"Type-organized directory: {type_output_dir.resolve()}")
    print(f"Remaining lines in filtered.jsonl: {len(lines_to_keep)}")

    # Save statistics
    stats_file = type_output_dir / "download_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
