#!/usr/bin/env python3
"""
Script to convert all text and HTML files in Datasetss folders to PDF format.
Uses reportlab instead of weasyprint for better macOS compatibility.
"""

import os
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_html_tags(text):
    """Remove HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def convert_html_to_pdf(html_path, output_path):
    """Convert HTML file to PDF using reportlab."""
    try:
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        # Clean HTML tags
        clean_text = clean_html_tags(html_content)

        # Create PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        title = Paragraph(
            f"Converted from: {Path(html_path).name}", title_style)
        story.append(title)
        story.append(Spacer(1, 12))

        # Add content
        # Split into paragraphs and add each one
        paragraphs = clean_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Handle long paragraphs by breaking them up
                if len(para) > 500:
                    # Break into smaller chunks
                    words = para.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) < 500:
                            current_line += " " + word if current_line else word
                        else:
                            if current_line:
                                p = Paragraph(current_line, styles['Normal'])
                                story.append(p)
                                story.append(Spacer(1, 6))
                            current_line = word
                    if current_line:
                        p = Paragraph(current_line, styles['Normal'])
                        story.append(p)
                        story.append(Spacer(1, 6))
                else:
                    p = Paragraph(para, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 6))

        doc.build(story)
        logger.info(f"Converted {html_path} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert {html_path}: {str(e)}")
        return False


def convert_text_to_pdf(text_path, output_path):
    """Convert text file to PDF using reportlab."""
    try:
        # Read text content
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()

        # Create PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        title = Paragraph(
            f"Converted from: {Path(text_path).name}", title_style)
        story.append(title)
        story.append(Spacer(1, 12))

        # Add content using Preformatted for better text preservation
        preformatted_style = ParagraphStyle(
            'CustomPreformatted',
            parent=styles['Code'],
            fontSize=10,
            leftIndent=20,
            rightIndent=20,
            fontName='Courier'
        )

        # Split content into manageable chunks
        lines = text_content.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk + line + '\n') < 2000:  # Keep chunks manageable
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    p = Preformatted(current_chunk, preformatted_style)
                    story.append(p)
                    story.append(Spacer(1, 12))
                current_chunk = line + '\n'

        # Add remaining content
        if current_chunk:
            p = Preformatted(current_chunk, preformatted_style)
            story.append(p)

        doc.build(story)
        logger.info(f"Converted {text_path} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert {text_path}: {str(e)}")
        return False


def convert_files_in_directory(directory_path):
    """Convert all text and HTML files in a directory to PDF."""
    directory = Path(directory_path)
    if not directory.exists():
        logger.warning(f"Directory {directory_path} does not exist")
        return 0, 0

    # Find all HTML and text files
    html_files = list(directory.glob("*.html")) + list(directory.glob("*.htm"))
    text_files = list(directory.glob("*.txt"))

    total_files = len(html_files) + len(text_files)
    converted_count = 0

    logger.info(
        f"Found {len(html_files)} HTML files and {len(text_files)} text files in {directory_path}")

    # Convert HTML files
    for html_file in html_files:
        output_path = html_file.with_suffix('.pdf')
        if convert_html_to_pdf(str(html_file), str(output_path)):
            converted_count += 1

    # Convert text files
    for text_file in text_files:
        output_path = text_file.with_suffix('.pdf')
        if convert_text_to_pdf(str(text_file), str(output_path)):
            converted_count += 1

    logger.info(
        f"Successfully converted {converted_count}/{total_files} files in {directory_path}")
    return converted_count, total_files


def main():
    """Main function to process all Datasetss folders."""
    # Get the script directory
    script_dir = Path(__file__).parent
    datasetss_dir = script_dir.parent / "Datasetss"

    if not datasetss_dir.exists():
        logger.error(f"Datasetss directory not found at {datasetss_dir}")
        return

    # Get all subdirectories
    subdirs = [d for d in datasetss_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(subdirs)} subdirectories in Datasetss")

    total_converted = 0
    total_files = 0

    # Process each subdirectory
    for subdir in subdirs:
        logger.info(f"Processing directory: {subdir.name}")
        converted, total = convert_files_in_directory(subdir)
        total_converted += converted
        total_files += total

    logger.info(
        f"Conversion complete! Total: {total_converted}/{total_files} files converted successfully")


if __name__ == "__main__":
    main()
