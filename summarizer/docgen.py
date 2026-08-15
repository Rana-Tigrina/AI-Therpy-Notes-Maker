from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import json
from datetime import datetime

class TherapyNotesDocumentGenerator:
    def __init__(self):
        self.document = Document()
        self._setup_document()

    def _setup_document(self):
        sections = self.document.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

    def _add_first_page(self):
        heading = self.document.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading_run = heading.add_run("CLIENT CREDENTIALS")
        heading_run.bold = True
        heading_run.font.size = Pt(12)
        self.document.add_paragraph()

        credentials = [
            ("NAME:", "AGE:"),
            ("GENDER:", "DATE OF BIRTH:"),
            ("RESIDENCE:", "CONTACT:"),
            ("E-MAIL:", "EMERGENCY CONTACT:"),
            ("CLIENT ID:", "APPOINTMENT ID:")
        ]
        for left, right in credentials:
            p = self.document.add_paragraph()
            p.paragraph_format.tab_stops.add_tab_stop(Inches(3)) 
            p.add_run(left).bold = True
            p.add_run("\t")
            p.add_run(right).bold = True

        self.document.add_paragraph()

        therapist_heading = self.document.add_paragraph()
        therapist_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        therapist_run = therapist_heading.add_run("THERAPIST CREDENTIALS")
        therapist_run.bold = True
        therapist_run.font.size = Pt(12)
        self.document.add_paragraph()

        therapist_credentials = [
            ("NAME:", "THERAPIST ID:"),
            ("RCI LICENSE NO:", "DESIGNATION:"),
            ("MENTOR:", "MENTOR LICENSE NO:"),
            ("Session No.:", ""),
            ("Session Date:", "")
        ]
        for left, right in therapist_credentials:
            p = self.document.add_paragraph()
            p.paragraph_format.tab_stops.add_tab_stop(Inches(3))
            p.add_run(left).bold = True
            if right:
                p.add_run("\t")
                p.add_run(right).bold = True

        self.document.add_page_break()

    def _add_therapy_notes_page(self, therapy_notes):
        heading = self.document.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading_run = heading.add_run("Therapy Notes")
        heading_run.bold = True
        heading_run.font.size = Pt(14)

        self.document.add_paragraph()

        for section_name, content in therapy_notes.items():
            formatted_title = section_name.replace('_', ' ').title()
            heading = self.document.add_heading(level=1)
            run = heading.add_run(formatted_title)
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0, 0, 139)

            if isinstance(content, dict):
                for key, value in content.items():
                    p = self.document.add_paragraph()
                    p.add_run(f"{key}:").bold = True
                    if isinstance(value, (list, dict)):
                        if isinstance(value, list):
                            for item in value:
                                sub_p = self.document.add_paragraph()
                                sub_p.style = 'List Bullet'
                                sub_p.add_run(str(item))
                        else:
                            for sub_key, sub_value in value.items():
                                sub_p = self.document.add_paragraph()
                                sub_p.style = 'List Bullet'
                                sub_p.add_run(f"{sub_key}: {sub_value}")
                    else:
                        p.add_run(f" {value}")
            elif isinstance(content, list):
                for item in content:
                    p = self.document.add_paragraph()
                    p.style = 'List Bullet'
                    p.add_run(str(item))

            self.document.add_paragraph()

    def generate_document(self, therapy_notes):
        self._add_first_page()
        self._add_therapy_notes_page(therapy_notes)
        return self.document

    def save_document(self, filename):
        self.document.save(filename)