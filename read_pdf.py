import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from pypdf import PdfReader

pdf_path = os.path.join(r"c:\DATATHON", "Đề thi Vòng 1.pdf")
reader = PdfReader(pdf_path)

with open(r"c:\DATATHON\pdf_content.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        f.write(f"=== PAGE {i+1} ===\n")
        text = page.extract_text()
        if text:
            f.write(text)
        f.write("\n\n")
print("Done! Written to pdf_content.txt")
