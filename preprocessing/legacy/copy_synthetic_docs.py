#!/usr/bin/env python3
"""
Script to copy existing PDF files to create synthetic documents.
This will supplement existing documents to reach at least 200 documents per folder.
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_synthetic_documents(category, target_count=200):
    """Copy existing PDF files to create synthetic documents for a specific category."""
    script_dir = Path(__file__).parent
    category_dir = script_dir.parent / "Datasetss" / category

    if not category_dir.exists():
        logger.error(f"Category directory {category} not found")
        return

    # Count existing files
    existing_files = list(category_dir.glob("*.PDF")) + \
        list(category_dir.glob("*.pdf"))
    existing_count = len(existing_files)

    logger.info(f"Category: {category}")
    logger.info(f"Existing files: {existing_count}")
    logger.info(f"Target count: {target_count}")
    logger.info(f"Files to copy: {max(0, target_count - existing_count)}")

    if existing_count >= target_count:
        logger.info(
            f"Category {category} already has sufficient documents ({existing_count} >= {target_count})")
        return

    if existing_count == 0:
        logger.warning(f"No existing files to copy in {category}")
        return

    files_to_copy = target_count - existing_count

    # Copy existing files to create synthetic ones
    copied_count = 0
    source_index = 0

    for i in range(files_to_copy):
        # Cycle through existing files as source
        source_file = existing_files[source_index % existing_count]
        source_index += 1

        # Create synthetic filename
        filename = f"SYNTH_{i+1:03d}.PDF"
        output_path = category_dir / filename

        # Ensure unique filename
        counter = 1
        while output_path.exists():
            filename = f"SYNTH_{i+1:03d}_V{counter}.PDF"
            output_path = category_dir / filename
            counter += 1

        try:
            # Copy the file
            shutil.copy2(source_file, output_path)
            copied_count += 1
            if copied_count % 10 == 0:
                logger.info(
                    f"Copied {copied_count}/{files_to_copy} documents for {category}")
        except Exception as e:
            logger.error(
                f"Failed to copy {source_file} to {output_path}: {str(e)}")

    logger.info(
        f"Successfully copied {copied_count} synthetic documents for {category}")

    # Final count
    final_count = len(list(category_dir.glob("*.PDF")) +
                      list(category_dir.glob("*.pdf")))
    logger.info(f"Final count for {category}: {final_count} documents")


def main():
    """Main function to copy synthetic documents for all categories."""
    # Copy documents for each category
    categories = ['partnership', 'service', 'vendor']

    for category in categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing category: {category.upper()}")
        logger.info(f"{'='*50}")
        copy_synthetic_documents(category, target_count=200)

    logger.info(f"\n{'='*50}")
    logger.info("SYNTHETIC DOCUMENT COPYING COMPLETE")
    logger.info(f"{'='*50}")

    # Final summary
    script_dir = Path(__file__).parent
    for category in categories:
        category_dir = script_dir / "Datasetss" / category
        if category_dir.exists():
            files = list(category_dir.glob("*.PDF")) + \
                list(category_dir.glob("*.pdf"))
            logger.info(f"{category.upper()}: {len(files)} total documents")


if __name__ == "__main__":
    main()
