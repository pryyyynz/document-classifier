#!/usr/bin/env python3
"""
Script to generate synthetic documents for partnership, service, and vendor categories.
This will supplement existing documents to reach at least 200 documents per folder for training and validation.
"""

import os
import random
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib import colors
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticDocumentGenerator:
    """Generate synthetic documents for different contract categories."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_company_data()

    def setup_company_data(self):
        """Setup realistic company names and data."""
        self.company_prefixes = [
            "Tech", "Global", "Advanced", "Innovative", "Strategic", "Digital", "Smart",
            "Future", "Next", "Prime", "Elite", "Core", "Main", "Central", "United",
            "International", "National", "Regional", "Local", "Metro", "Urban", "Rural"
        ]

        self.company_suffixes = [
            "Solutions", "Systems", "Technologies", "Corporation", "Inc", "LLC", "Ltd",
            "Group", "Holdings", "Partners", "Associates", "Enterprises", "Ventures",
            "Industries", "Services", "Consulting", "Advisory", "Management", "Capital",
            "Investments", "Fund", "Trust", "Foundation", "Institute", "Laboratories"
        ]

        self.industries = [
            "Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Energy",
            "Transportation", "Telecommunications", "Media", "Education", "Real Estate",
            "Biotechnology", "Pharmaceuticals", "Automotive", "Aerospace", "Defense",
            "Agriculture", "Construction", "Mining", "Oil & Gas", "Utilities", "Insurance"
        ]

        self.contract_types = {
            'partnership': [
                "Joint Venture Agreement", "Strategic Alliance Agreement", "Partnership Agreement",
                "Collaboration Agreement", "Co-Marketing Agreement", "Distribution Agreement",
                "Licensing Agreement", "Franchise Agreement", "Revenue Sharing Agreement",
                "Technology Transfer Agreement", "Research & Development Agreement",
                "Manufacturing Agreement", "Supply Chain Agreement", "Investment Agreement"
            ],
            'service': [
                "Service Agreement", "Consulting Agreement", "Professional Services Agreement",
                "Maintenance Agreement", "Support Agreement", "Implementation Agreement",
                "Training Agreement", "Outsourcing Agreement", "Managed Services Agreement",
                "SLA Agreement", "Service Level Agreement", "Operations Agreement",
                "Administrative Services Agreement", "Technical Services Agreement"
            ],
            'vendor': [
                "Vendor Agreement", "Supplier Agreement", "Procurement Agreement",
                "Purchase Agreement", "Master Services Agreement", "Statement of Work",
                "Consulting Agreement", "Professional Services Agreement", "Work Order",
                "Service Order", "Maintenance Agreement", "Support Agreement",
                "Implementation Agreement", "Training Agreement", "Outsourcing Agreement"
            ]
        }

        self.legal_terms = [
            "Confidentiality", "Non-Disclosure", "Intellectual Property", "Termination",
            "Liability", "Indemnification", "Force Majeure", "Dispute Resolution",
            "Governing Law", "Jurisdiction", "Arbitration", "Mediation", "Notices",
            "Amendments", "Waiver", "Severability", "Entire Agreement", "Counterparts"
        ]

    def generate_company_name(self):
        """Generate a realistic company name."""
        prefix = random.choice(self.company_prefixes)
        suffix = random.choice(self.company_suffixes)
        industry = random.choice(self.industries)

        # 70% chance to include industry, 30% chance for generic name
        if random.random() < 0.7:
            return f"{prefix} {industry} {suffix}"
        else:
            return f"{prefix} {suffix}"

    def generate_date(self, start_year=2010, end_year=2024):
        """Generate a realistic date within the specified range."""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        return random_date.strftime("%m_%d_%Y")

    def generate_exhibit_number(self, category):
        """Generate realistic exhibit numbers based on category."""
        if category == 'partnership':
            ex_numbers = ["EX-10.1", "EX-10.2", "EX-10.3", "EX-10.4", "EX-10.5",
                          "EX-10.11", "EX-10.12", "EX-10.13", "EX-10.14", "EX-10.15",
                          "EX-99.1", "EX-99.2", "EX-99.3", "EX-99.4", "EX-99.5"]
        elif category == 'service':
            ex_numbers = ["EX-4.25", "EX-10.3", "EX-10.8", "EX-10.11", "EX-10.12",
                          "EX-99.8.77", "EX-99.SERV", "EX-99.1", "EX-99.2", "EX-99.3"]
        else:  # vendor
            ex_numbers = ["EX-10.12", "EX-10.1", "EX-10.3", "EX-4.23", "EX-99.28.H.9",
                          "EX-99.1", "EX-99.2", "EX-99.3", "EX-99.4", "EX-99.5"]

        return random.choice(ex_numbers)

    def generate_filename(self, category, company_name, date, exhibit_num):
        """Generate a realistic filename following the existing pattern."""
        # Clean company name for filename
        clean_name = company_name.replace(
            " ", "").replace(",", "").replace(".", "")
        contract_type = random.choice(self.contract_types[category])

        # Format: COMPANYNAME_DATE-EXHIBIT-CONTRACTTYPE.PDF
        filename = f"{clean_name}_{date}-{exhibit_num}-{contract_type.upper().replace(' ', '')}.PDF"
        return filename

    def generate_partnership_content(self, company1, company2, contract_type):
        """Generate realistic partnership agreement content."""
        content = []

        # Header
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        content.append(Paragraph(f"{contract_type.upper()}", header_style))
        content.append(Spacer(1, 12))

        # Parties
        parties_style = ParagraphStyle(
            'Parties',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        content.append(Paragraph(
            f"This {contract_type} (the 'Agreement') is entered into as of {datetime.now().strftime('%B %d, %Y')} by and between:", parties_style))
        content.append(Paragraph(
            f"<b>{company1}</b>, a corporation organized under the laws of Delaware (the 'First Party')", parties_style))
        content.append(Paragraph(
            f"<b>{company2}</b>, a corporation organized under the laws of Delaware (the 'Second Party')", parties_style))
        content.append(Spacer(1, 12))

        # Recitals
        recitals = [
            "WHEREAS, the Parties wish to establish a strategic relationship for mutual benefit;",
            "WHEREAS, the Parties desire to collaborate on business opportunities;",
            "WHEREAS, the Parties have agreed to share resources and expertise;",
            "WHEREAS, the Parties intend to create value through their collaboration;"
        ]

        for recital in recitals:
            content.append(Paragraph(recital, self.styles['Normal']))
            content.append(Spacer(1, 6))

        content.append(Spacer(1, 12))

        # Agreement
        content.append(Paragraph(
            "NOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, the Parties agree as follows:", self.styles['Normal']))
        content.append(Spacer(1, 12))

        # Key terms
        terms = [
            ("1. Purpose", "The purpose of this Agreement is to establish a framework for collaboration between the Parties."),
            ("2. Term", "This Agreement shall commence on the Effective Date and continue for a period of three (3) years."),
            ("3. Scope of Collaboration", "The Parties agree to collaborate in areas including but not limited to technology development, market expansion, and resource sharing."),
            ("4. Confidentiality", "Each Party agrees to maintain the confidentiality of any proprietary information shared during the course of this Agreement."),
            ("5. Intellectual Property", "Each Party retains ownership of its pre-existing intellectual property. New IP developed jointly shall be owned as agreed upon by the Parties."),
            ("6. Termination", "Either Party may terminate this Agreement with thirty (30) days written notice to the other Party."),
            ("7. Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.")
        ]

        for title, description in terms:
            content.append(
                Paragraph(f"<b>{title}</b>", self.styles['Heading2']))
            content.append(Paragraph(description, self.styles['Normal']))
            content.append(Spacer(1, 6))

        return content

    def generate_service_content(self, company1, company2, contract_type):
        """Generate realistic service agreement content."""
        content = []

        # Header
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        content.append(Paragraph(f"{contract_type.upper()}", header_style))
        content.append(Spacer(1, 12))

        # Parties
        parties_style = ParagraphStyle(
            'Parties',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        content.append(Paragraph(
            f"This {contract_type} (the 'Agreement') is made and entered into as of {datetime.now().strftime('%B %d, %Y')} by and between:", parties_style))
        content.append(Paragraph(
            f"<b>{company1}</b>, a Delaware corporation (the 'Service Provider')", parties_style))
        content.append(Paragraph(
            f"<b>{company2}</b>, a Delaware corporation (the 'Client')", parties_style))
        content.append(Spacer(1, 12))

        # Recitals
        recitals = [
            "WHEREAS, the Client desires to engage the Service Provider to provide certain services;",
            "WHEREAS, the Service Provider is willing to provide such services on the terms and conditions set forth herein;",
            "WHEREAS, the Parties wish to establish the terms and conditions governing the provision of such services;"
        ]

        for recital in recitals:
            content.append(Paragraph(recital, self.styles['Normal']))
            content.append(Spacer(1, 6))

        content.append(Spacer(1, 12))

        # Agreement
        content.append(Paragraph(
            "NOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, the Parties agree as follows:", self.styles['Normal']))
        content.append(Spacer(1, 12))

        # Key terms
        terms = [
            ("1. Services", "The Service Provider shall provide the services described in the Statement of Work attached hereto as Exhibit A."),
            ("2. Term", "This Agreement shall commence on the Effective Date and continue for a period of one (1) year, unless earlier terminated."),
            ("3. Compensation", "The Client shall pay the Service Provider for services rendered in accordance with the fee schedule set forth in Exhibit B."),
            ("4. Performance Standards", "The Service Provider shall perform all services in a professional and workmanlike manner in accordance with industry standards."),
            ("5. Confidentiality", "Each Party agrees to maintain the confidentiality of any proprietary information shared during the course of this Agreement."),
            ("6. Independent Contractor",
             "The Service Provider is an independent contractor and not an employee, agent, or representative of the Client."),
            ("7. Termination", "Either Party may terminate this Agreement with thirty (30) days written notice to the other Party."),
            ("8. Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.")
        ]

        for title, description in terms:
            content.append(
                Paragraph(f"<b>{title}</b>", self.styles['Heading2']))
            content.append(Paragraph(description, self.styles['Normal']))
            content.append(Spacer(1, 6))

        return content

    def generate_vendor_content(self, company1, company2, contract_type):
        """Generate realistic vendor agreement content."""
        content = []

        # Header
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        content.append(Paragraph(f"{contract_type.upper()}", header_style))
        content.append(Spacer(1, 12))

        # Parties
        parties_style = ParagraphStyle(
            'Parties',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        content.append(Paragraph(
            f"This {contract_type} (the 'Agreement') is entered into as of {datetime.now().strftime('%B %d, %Y')} by and between:", parties_style))
        content.append(Paragraph(
            f"<b>{company1}</b>, a Delaware corporation (the 'Vendor')", parties_style))
        content.append(Paragraph(
            f"<b>{company2}</b>, a Delaware corporation (the 'Customer')", parties_style))
        content.append(Spacer(1, 12))

        # Recitals
        recitals = [
            "WHEREAS, the Customer desires to procure certain goods and/or services from the Vendor;",
            "WHEREAS, the Vendor is willing to provide such goods and/or services on the terms and conditions set forth herein;",
            "WHEREAS, the Parties wish to establish the terms and conditions governing the provision of such goods and/or services;"
        ]

        for recital in recitals:
            content.append(Paragraph(recital, self.styles['Normal']))
            content.append(Spacer(1, 6))

        content.append(Spacer(1, 12))

        # Agreement
        content.append(Paragraph(
            "NOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, the Parties agree as follows:", self.styles['Normal']))
        content.append(Spacer(1, 12))

        # Key terms
        terms = [
            ("1. Goods and Services", "The Vendor shall provide the goods and/or services described in the Statement of Work attached hereto as Exhibit A."),
            ("2. Term", "This Agreement shall commence on the Effective Date and continue for a period of two (2) years, unless earlier terminated."),
            ("3. Pricing and Payment", "The Customer shall pay the Vendor for goods and/or services provided in accordance with the pricing schedule set forth in Exhibit B."),
            ("4. Quality Standards", "The Vendor shall provide goods and/or services that meet or exceed industry standards and any specifications set forth in Exhibit A."),
            ("5. Delivery", "The Vendor shall deliver goods and/or services in accordance with the delivery schedule set forth in Exhibit A."),
            ("6. Warranties", "The Vendor warrants that all goods and/or services provided hereunder shall be free from defects in materials and workmanship."),
            ("7. Confidentiality", "Each Party agrees to maintain the confidentiality of any proprietary information shared during the course of this Agreement."),
            ("8. Termination", "Either Party may terminate this Agreement with thirty (30) days written notice to the other Party."),
            ("9. Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.")
        ]

        for title, description in terms:
            content.append(
                Paragraph(f"<b>{title}</b>", self.styles['Heading2']))
            content.append(Paragraph(description, self.styles['Normal']))
            content.append(Spacer(1, 6))

        return content

    def generate_document(self, category, output_path, company1, company2, contract_type):
        """Generate a complete synthetic document."""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)

            if category == 'partnership':
                content = self.generate_partnership_content(
                    company1, company2, contract_type)
            elif category == 'service':
                content = self.generate_service_content(
                    company1, company2, contract_type)
            else:  # vendor
                content = self.generate_vendor_content(
                    company1, company2, contract_type)

            doc.build(content)
            return True
        except Exception as e:
            logger.error(
                f"Failed to generate document {output_path}: {str(e)}")
            return False

    def generate_category_documents(self, category, target_count=200):
        """Generate synthetic documents for a specific category."""
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
        logger.info(
            f"Files to generate: {max(0, target_count - existing_count)}")

        if existing_count >= target_count:
            logger.info(
                f"Category {category} already has sufficient documents ({existing_count} >= {target_count})")
            return

        files_to_generate = target_count - existing_count

        # Generate synthetic documents
        generated_count = 0
        for i in range(files_to_generate):
            company1 = self.generate_company_name()
            company2 = self.generate_company_name()
            contract_type = random.choice(self.contract_types[category])
            date = self.generate_date()
            exhibit_num = self.generate_exhibit_number(category)

            filename = self.generate_filename(
                category, company1, date, exhibit_num)
            output_path = category_dir / filename

            # Ensure unique filename
            counter = 1
            while output_path.exists():
                base_name = filename.rsplit('.', 1)[0]
                filename = f"{base_name}_V{counter}.PDF"
                output_path = category_dir / filename
                counter += 1

            if self.generate_document(category, output_path, company1, company2, contract_type):
                generated_count += 1
                if generated_count % 10 == 0:
                    logger.info(
                        f"Generated {generated_count}/{files_to_generate} documents for {category}")

        logger.info(
            f"Successfully generated {generated_count} synthetic documents for {category}")

        # Final count
        final_count = len(list(category_dir.glob("*.PDF")) +
                          list(category_dir.glob("*.pdf")))
        logger.info(f"Final count for {category}: {final_count} documents")


def main():
    """Main function to generate synthetic documents for all categories."""
    generator = SyntheticDocumentGenerator()

    # Generate documents for each category
    categories = ['partnership', 'service', 'vendor']

    for category in categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing category: {category.upper()}")
        logger.info(f"{'='*50}")
        generator.generate_category_documents(category, target_count=200)

    logger.info(f"\n{'='*50}")
    logger.info("SYNTHETIC DOCUMENT GENERATION COMPLETE")
    logger.info(f"{'='*50}")

    # Final summary
    script_dir = Path(__file__).parent
    for category in categories:
        category_dir = script_dir.parent / "Datasetss" / category
        if category_dir.exists():
            files = list(category_dir.glob("*.PDF")) + \
                list(category_dir.glob("*.pdf"))
            logger.info(f"{category.upper()}: {len(files)} total documents")


if __name__ == "__main__":
    main()
