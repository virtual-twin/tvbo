import pandas as pd
import pymupdf


class Publication:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.doc = pymupdf.open(pdf_file)
        self.toc = pd.DataFrame(self.doc.get_toc(), columns=["Level", "Title", "Page"])

    def get_section_pages(self, section_title="Methods"):
        # Find the page number for the 'Methods' section (3rd entry)
        methods_row = self.toc[self.toc["Title"] == section_title].iloc[0]
        methods_page_start = methods_row["Page"]
        methods_level = methods_row["Level"]

        # Find the next entry with the same level as 'Methods'
        subsequent_row = self.toc[
            (self.toc["Level"] == methods_level)
            & (self.toc["Page"] > methods_page_start)
        ].iloc[0]
        methods_page_end = subsequent_row["Page"] - 1

        return methods_page_start, methods_page_end

    def extract_section(self, section_title="Methods"):
        """
        Extract the text from a specific section of the PDF, limiting the content to start
        from the first occurrence of the section title and end at the last occurrence of the next section title.

        Args:
            section_title (str): The title of the section to extract. Default is 'Methods'.

        Returns:
            str: The extracted text from the specified section.
        """
        start, end = self.get_section_pages(section_title)
        text = ""

        for page_num in range(start - 1, end + 1):
            page = self.doc.load_page(page_num)
            text += page.get_text()

        # Find the first occurrence of the section title
        start_idx = text.find(section_title)
        if start_idx == -1:
            start_idx = 0  # Fallback to start of the text if not found

        # Find the next section title after the current one
        next_section_title = self.toc[self.toc["Page"] == end + 1]["Title"].values[0]
        end_idx = text.rfind(next_section_title)
        if end_idx == -1:
            end_idx = len(text)  # Fallback to end of the text if not found

        return text[start_idx:end_idx].strip()
