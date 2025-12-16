from dotenv import load_dotenv
import argparse, requests, os, re, unicodedata, cloudconvert
from openai import OpenAI
import urllib.parse

UNICODE_DASHES = "‐–—−"
UNICODE_LANGLE = ["〈", "〈", "﹤", "＜"]  # U+2329, U+3008, etc.
UNICODE_RANGLE = ["〉", "〉", "﹥", "＞"]  # U+232A, U+3009, etc.

def require_api_key():
    load_dotenv()
    api_key = os.getenv("C_KEY")
    if not api_key:
        raise RuntimeError("C_KEY not set. Put it in a .env file or set env var.")
    cloudconvert.configure(api_key=api_key)
    return api_key

def _find_task(job, operation: str):
    for t in job.get("tasks", []):
        if t.get("operation") == operation:
            return t
    return None

# Job A: OCR (PDF -> OCR'ed PDF)
def _ocr_pdf_job(pdf_path: str, languages=("eng",), auto_orient: bool = True):
    """
    Upload local PDF -> pdf/ocr -> export URL(s).
    Returns: list of exported file dicts, each with {"filename": "...", "url": "..."}.
    """
    job = cloudconvert.Job.create(payload={
        "tasks": {
            "upload": {"operation": "import/upload"},
            "ocr": {
                "operation": "pdf/ocr",
                "input": "upload",
                "language": list(languages),
                "auto_orient": bool(auto_orient)
            },
            "export": {"operation": "export/url", "input": "ocr"}
        }
    })

    upload_task = _find_task(job, "import/upload")
    if not upload_task:
        raise RuntimeError("No 'import/upload' task found in OCR job.")
    upload_task = cloudconvert.Task.find(id=upload_task["id"])
    cloudconvert.Task.upload(file_name=os.path.abspath(pdf_path), task=upload_task)

    export_task = _find_task(job, "export/url")
    if not export_task:
        raise RuntimeError("No 'export/url' task found in OCR job.")
    export_task = cloudconvert.Task.wait(id=export_task["id"])

    files = (export_task.get("result") or {}).get("files") or []
    return files

# Job B: PDF (OCR'ed) -> HTML
def _pdf_to_html_job_from_url(pdf_url: str, out_dir: str, engine: str = "pdf2htmlex"):
    """
    Import OCR'ed PDF by URL -> convert (pdf->html) -> export URL(s) -> download HTML(s).
    Returns: list of local .html paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    job = cloudconvert.Job.create(payload={
        "tasks": {
            "import-pdf": {
                "operation": "import/url",
                "url": pdf_url
            },
            "convert-pdf-html": {
                "operation": "convert",
                "input": "import-pdf",
                "input_format": "pdf",
                "output_format": "html",
                "engine": engine,  # explicit engine for better academic layout
                # You can add engine-specific options here if needed, e.g.:
                # "embed_css": True,
                # "split_pages": False,
            },
            "export-html": {
                "operation": "export/url",
                "input": "convert-pdf-html"
            }
        }
    })

    export_task = _find_task(job, "export/url")
    if not export_task:
        raise RuntimeError("No 'export/url' task found in PDF->HTML job.")
    export_task = cloudconvert.Task.wait(id=export_task["id"])

    files = (export_task.get("result") or {}).get("files") or []
    html_paths = []
    for f in files:
        if not f["filename"].lower().endswith(".html"):
            # skip non-HTML assets, if any
            continue
        local_path = os.path.join(out_dir, f["filename"])
        cloudconvert.download(filename=local_path, url=f["url"])
        print(f"Downloaded file:{local_path} successfully..")
        html_paths.append(local_path)
    return html_paths

# Job C: HTML -> TXT (local upload)
def _html_local_to_txt_job(local_html_path: str, out_dir: str):
    """
    Import/upload local HTML -> convert (html->txt) -> export URL(s) -> download TXT(s).
    Returns: list of local .txt paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    job = cloudconvert.Job.create(payload={
        "tasks": {
            "upload-html": {"operation": "import/upload"},
            "convert-html-txt": {
                "operation": "convert",
                "input": "upload-html",
                "input_format": "html",
                "output_format": "txt"
            },
            "export-txt": {
                "operation": "export/url",
                "input": "convert-html-txt"
            }
        }
    })

    upload_task = _find_task(job, "import/upload")
    if not upload_task:
        raise RuntimeError("No 'import/upload' task in HTML->TXT job.")
    upload_task = cloudconvert.Task.find(id=upload_task["id"])
    cloudconvert.Task.upload(file_name=os.path.abspath(local_html_path), task=upload_task)

    export_task = _find_task(job, "export/url")
    if not export_task:
        raise RuntimeError("No 'export/url' task in HTML->TXT job.")
    export_task = cloudconvert.Task.wait(id=export_task["id"])

    files = (export_task.get("result") or {}).get("files") or []
    txt_paths = []
    for f in files:
        if not f["filename"].lower().endswith(".txt"):
            continue
        local_path = os.path.join(out_dir, f["filename"])
        cloudconvert.download(filename=local_path, url=f["url"])
        print(f"Downloaded file:{local_path} successfully..")
        txt_paths.append(local_path)
    return txt_paths

# Orchestrator: PDF -> OCR -> HTML -> TXT
def convert_pdf_to_txt_with_ocr(pdf_path: str,
                                out_dir: str,
                                languages=("eng",),
                                html_engine: str = "pdf2htmlex"):
    os.makedirs(out_dir, exist_ok=True)
    ocr_exports = _ocr_pdf_job(pdf_path, languages=languages)
    if not ocr_exports:
        print("No OCR outputs returned.")
        return []

    all_txt_paths = []

    # B) Convert each OCR'ed PDF to HTML
    for fpdf in ocr_exports:
        if not fpdf["filename"].lower().endswith(".pdf"):
            continue
        html_paths = _pdf_to_html_job_from_url(fpdf["url"], out_dir, engine=html_engine)

        # C) Convert each HTML to TXT (local re-upload simplifies flow)
        for hpath in html_paths:
            txt_paths = _html_local_to_txt_job(hpath, out_dir)
            all_txt_paths.extend(txt_paths)

    return all_txt_paths

# Extract from txt
def get_references_block(text):
    """
    Extract ref / bib from a txt file. Start with header, then end before appendix etc
    """
    t = text.replace('\r\n', '\n').replace('\r', '\n')

    # Header: references / reference / bibliography (colon/dashes optional), alone on line
    start_pat = re.compile(
        r'^[ \t]*(?:references?|bibliography)[ \t]*:?[ \t]*[-–—=]*[ \t]*$',
        re.IGNORECASE | re.MULTILINE
    )
    m_start = start_pat.search(t)
    if not m_start:
        return ""

    # Enders: next major section headers after references
    end_pat = re.compile(
        r'^[ \t]*(?:'
        r'appendix(?:\s+[A-Z0-9]+)?|appendices|'
        r'supplementary(?:\s+material)?|supplemental|'
        r'figures?|tables?'
        r')[ \t]*[:\-–—=]?\b.*$',
        re.IGNORECASE | re.MULTILINE
    )
    m_end = end_pat.search(t, pos=m_start.end())

    start_idx = m_start.end()
    end_idx = m_end.start() if m_end else len(t)

    block = t[start_idx:end_idx].strip()
    cleaned_lines = []
    for line in block.split('\n'):
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r'\d+', s):
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

def openai_response(prompt, model="gpt-4o"):
    try:
        system_prompt = """You are a text-processing assistant. 
You will receive the full content of a `.txt` file that contains many references or citations. 
Some citations may be broken across multiple lines (for example, when a reference wraps between pages). 
Your task is to reconstruct each citation so that each full reference appears on exactly ONE line.

Follow these rules carefully:
1. Combine consecutive lines that belong to the same citation into a single line.
2. Keep the original punctuation, spacing, and wording from the source text.
3. Insert a single space where line breaks were removed (unless there is already a space).
4. Ensure each complete citation ends with a period, and there is exactly one blank line between citations.
5. Do not merge two different citations together — start a new line when a new citation begins (often identified by a new author name pattern like "Lastname, Initial." or a year like "2020:").
6. Output plain text only, with one reference per line.
7. Start with (<GPT I will start>) at a separate line. When finished, put (<GPT I am done>) at the end in a separate line.
8. If you saw anything that is a header or a footer, ignore it. For example, page numbers, or repetitive doi with page number and author etc. But if the header and footer are within another reference, just remove the header and footer, keep the rest of the reference as is.

Example Input:
Aðalgeirsdóttir, G. et al., 2020: Glacier changes in Iceland from ~1890 to 2019.  
Frontiers in Earth Science, 8, 520, doi:10.3389/feart.2020.523646.  
Adler,  R.F.  et  al.,  2003:  The  Version-2  Global  Precipitation  Climatology  
Project  (GPCP)  Monthly  Precipitation  Analysis  (1979-Present).  Journal  
of  Hydrometeorology,  4(6),  1147-1167,  doi:10.1175/1525-7541(2003)  
004<1147:tvgpcp>2.0.co;2.  
Adusumilli, S., H.A. Fricker, B. Medley, L. Padman, and M.R. Siegfried, 2020:  
Interannual  variations  in  meltwater  input  to  the  Southern  Ocean  from  
Antarctic  ice  shelves.  Nature  Geoscience,  13,  616-620,  doi:10.1038/  
s41561-020-0616-z.
Park, B., Pit
ˇ
na, A., Šafránková, J., N
ˇ
eme
ˇ
cek, Z., Krupa
ˇ
rová,
O., Krupa
ˇ
r, V., Zhao, L., and Silwal, A.: Change of Spec-
tral Properties of Magnetic Field Fluctuations across Differ-
Ann. Geophys., 43, 489-510, 2025 https://doi.org/10.5194/angeo-43-489-2025
10.1086/146579,
10.1016/S0273-1177(03)90635-
16-017-001
1-z,
2018.
E. Kilpua et al.: Shock effect on turbulence parameters 509
ent Types of Interplanetary Shocks, Astrophys. J., 954, L51,
https://doi.org/10.3847/2041-8213/acf4ff, 2023.

Example Output:
(<GPT I will start>)
Aðalgeirsdóttir, G. et al., 2020: Glacier changes in Iceland from ~1890 to 2019. Frontiers in Earth Science, 8, 520, doi:10.3389/feart.2020.523646.

Adler, R.F. et al., 2003: The Version-2 Global Precipitation Climatology Project (GPCP) Monthly Precipitation Analysis (1979-Present). Journal of Hydrometeorology, 4(6), 1147-1167, doi:10.1175/1525-7541(2003)004<1147:tvgpcp>2.0.co;2.

Adusumilli, S., H.A. Fricker, B. Medley, L. Padman, and M.R. Siegfried, 2020: Interannual variations in meltwater input to the Southern Ocean from Antarctic ice shelves. Nature Geoscience, 13, 616-620, doi:10.1038/s41561-020-0616-z.

Park, B., Pitˇ na, A., Šafránková, J., Nˇemeˇ cek, Z., Krupaˇ rová,O., Krupaˇ r, V., Zhao, L., and Silwal, A.: Change of Spectral Properties of Magnetic Field Fluctuations across Different Types of Interplanetary Shocks, Astrophys. J., 954, L51, https://doi.org/10.3847/2041-8213/acf4ff, 2023.
(<GPT I am done>)
"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def normalize_reference_text(text):
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = re.sub(r'(\w)/\s+(\w)', r'\1/\2', text)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()

def has_complete_doi(text):
    normalized_text = normalize_reference_text(text)
    doi_patterns = [
        r'doi:\s*10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+',
        r'https?://doi\.org/10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+',
        r'doi.org/10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+',
    ]
    
    for pattern in doi_patterns:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            return True
    return False

def get_doi_data(doi):
    url = "https://citation.doi.org/metadata"
    params = {"doi": doi}
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", "N/A")
        if isinstance(title, list):
            title = title[0] if title else "N/A"
        title = title.replace("‐", "-")
        title = title.replace("–", "-")
        #title = title.replace("-", "")
        title = title.replace("’", "'")
        title = re.sub(r"[?!]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        
        authors = data.get("author", [])
        author_names = []
        if not authors:
            return title, None
        
        for author in authors:
            last = author.get("family", "")
            last = last.replace("’", "'")
            author_names.append(f"{last}".strip())
        
        first_author_last_name = author_names[0] if author_names else None
        return title, first_author_last_name

    except requests.RequestException as e:
        return None, None

def extract_doi_from_text(text):
    t = unicodedata.normalize("NFKC", text)
    for ch in UNICODE_DASHES: t = t.replace(ch, "-")
    for ch in UNICODE_LANGLE: t = t.replace(ch, "<")
    for ch in UNICODE_RANGLE: t = t.replace(ch, ">")
    doi_patterns = [
        r'doi:\s*10\.\d{4,9}/[-._;()/:<>#a-zA-Z0-9]+',
        r'https?://doi\.org/10\.\d{4,9}/[-._;()/:<>#a-zA-Z0-9]+',
        r'doi.org/10\.\d{4,9}/[-._;()/:<>#a-zA-Z0-9]+',
    ]
    
    normalized_text = normalize_reference_text(t)
    
    for pattern in doi_patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            doi_match = match.group(0)
            if doi_match.startswith('doi:'):
                doi = doi_match[4:]
            elif 'doi.org/' in doi_match:
                doi = doi_match.split('doi.org/')[-1]
            elif '://' in doi_match:
                doi = doi_match.split('/')[-1]
            else:
                doi = doi_match

            if doi.endswith('.'):
                doi = doi[:-1]
            
            return doi
    return None

def process_references_with_doi(references):
    no_doi_references = []
    valid_references = []
    wrong_doi_list = []
    reason_list = dict()
    
    for ref in references:
        doi = extract_doi_from_text(ref)
        if doi:
            title, first_author = get_doi_data(doi)
            if title is None and first_author is None:
                wrong_doi_list.append(ref)
                reason_list[ref] = "DOI and author not found in doi.org, likely invalid DOI"
            else:
                title_in_ref = is_text_in_reference(title, ref)
                # author_in_ref = is_author_in_reference(first_author, ref) if first_author else False
                if not title_in_ref:
                    wrong_doi_list.append(ref)
                    reason_list[ref] = "Title mismatch in reference"
                    continue
                
                valid_references.append(ref)
        else:
            no_doi_references.append(ref)
    
    return valid_references, wrong_doi_list, no_doi_references, reason_list

def is_text_in_reference(search_text, reference_text):
    """
    Check if search_text appears in reference_text (case-insensitive, partial matching)
    """
    if not search_text or search_text == 'N/A':
        return False
    
    def normalize_text(text):
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = text.replace("‐", "-").replace("–", "-").replace("—", "-").replace("−", "-")
        
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    clean_search = normalize_text(search_text.lower())
    clean_ref = normalize_text(reference_text.lower())
    
    search_words = [word for word in clean_search.split() if len(word) > 3]
    
    if not search_words:
        return False
    
    matches = sum(1 for word in search_words if word in clean_ref)
    
    # Return True if at least 70% of significant words match
    return matches / len(search_words) >= 0.7

def is_author_in_reference(author_name, reference_text):
    if not author_name or author_name == 'N/A':
        return False

    if author_name.lower() in reference_text.lower():
        return True
    patterns = [
        rf"{re.escape(author_name)},\s*[A-Z]",
        rf"^{re.escape(author_name)}",
        rf"and\s+{re.escape(author_name)}",
        rf"{re.escape(author_name)}\s+et al",
    ]
    
    for pattern in patterns:
        if re.search(pattern, reference_text, re.IGNORECASE):
            return True
    
    return False

def make_doi_clickable(ref_text, flag=True):
    """Convert DOI in reference text to clickable HTML link."""
    ref_text = ref_text.replace("‐", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    ref_text = ref_text.replace("<", "&lt;").replace(">", "&gt;")

    if flag:
        doi = extract_doi_from_text(ref_text)
    else:
        doi = ref_text

    if doi:
        doi = doi.replace("‐", "-").replace("–", "-").replace("—", "-").replace("−", "-")
        doi_url = f"https://doi.org/{doi}"

        # Include more possible characters inside DOI path
        doi_patterns = [
            (r'https?://doi\.org/10\.\d{4,9}/[^\s"\'<>]+', lambda m: f'<a href="{doi_url}" target="_blank">{m.group(0)}</a>'),
            (r'doi:\s*10\.\d{4,9}/[^\s"\'<>]+', lambda m: f'<a href="{doi_url}" target="_blank">{m.group(0)}</a>'),
            (r'doi\.org/10\.\d{4,9}/[^\s"\'<>]+', lambda m: f'<a href="{doi_url}" target="_blank">{m.group(0)}</a>'),
        ]

        for pattern, replacement in doi_patterns:
            if re.search(pattern, ref_text, re.IGNORECASE):
                return re.sub(pattern, replacement, ref_text, count=1, flags=re.IGNORECASE)

    return ref_text

def extract_title_from_reference(ref_line: str, model="gpt-4o"):
    """
    Given a single reference line, use GPT to extract the paper title only.
    Returns the title string or None if not found.
    """
    try:
        system_prompt = (
            "You are a reference text analyzer.\n"
            "Input: one bibliographic reference line.\n"
            "Output: ONLY the first author's lastname and title of the paper, exactly as it appears.\n"
            "Do not include other authors, journal names, or DOIs.\n"
            "If the title cannot be found, return 'None'.\n"
            "Return in the format of 'LASTNAME, Title'.\n"
            "Do not make up or autocomplete any doi or title, you can only infer from the existing lines.\n"
            "Do not add any extra explanation."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ref_line}
            ]
        )

        title = response.choices[0].message.content.strip().strip('"').strip()
        if title.lower() in ("none", ""):
            return None
        return title

    except Exception as e:
        return None

def query_crossref_by_title(title: str, author: str):
    q = urllib.parse.quote(title)
    a = urllib.parse.quote(author)
    url = f"https://api.crossref.org/works?query.title={q}&query.author={a}&rows={3}"
    headers = {"User-Agent": "DOI-Finder/1.0 (mailto:your.email@example.com)"}
    
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    items = data["message"]["items"]

    results = []
    for i in items:
        authors = []
        for a in i.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        results.append({
            "title": i.get("title", [""])[0],
            "doi": i.get("DOI"),
            "year": i.get("issued", {}).get("date-parts", [[None]])[0][0],
            "publisher": i.get("publisher"),
            "authors": authors,
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to TXT with OCR")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    args = parser.parse_args()
    require_api_key()
    load_dotenv()
    FILE_NAME = args.pdf_path.split("/")[-1].split(".")[0]
    #cropped_pdf_path = f"cropped_pdf/{FILE_NAME}.cropped.pdf"
    cropped_pdf_path = args.pdf_path
    print("Crop pdf to remove header footer etc begins")
    #
    # Crop pdf logic, takes in the original pdf path, save to cropped_pdf path
    #
    print("Cropped pdf ready")
    # Next is to turn the cropped pdf into txt file
    print("Conversion from cropped pdf to txt begins")
    convert_pdf_to_txt_with_ocr(cropped_pdf_path, "paper_txt")
    # get the file name from the path
    print("Conversion to txt complete")
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    #txt_path = f"paper_txt/{FILE_NAME}.cropped.txt"
    txt_path = f"paper_txt/{FILE_NAME}.txt" # debug
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    # txt = test_text
    lines = get_references_block(txt)
    # write lines to a txt file
    with open(f'reference_messy_txt/{FILE_NAME}_reference.txt', 'w') as f:
        f.write(lines)
    with open(f'reference_messy_txt/{FILE_NAME}_reference.txt', 'r') as file:
        inp = file.read()
    print("Sent to LM for cleaning")
    reply = openai_response(inp)
    print("GPT response received.")

    with open(f'reference_clean_txt/{FILE_NAME}_reference.txt', 'w') as f:
        f.write(reply)
    
    with open(f'reference_clean_txt/{FILE_NAME}_reference.txt', 'r') as file:
        test_text = file.read()
        # read everything between (<GPT I will start>) and (<GPT I am done>)
        match = re.search(r'\(<GPT I will start>\)(.*?)\(<GPT I am done>\)', test_text, re.DOTALL)
        if match:
            relevant_text = match.group(1).strip()
        else:
            relevant_text = test_text.strip()
    print("Begin comparison")
    relevant_text_list = relevant_text.split('\n')
    relevant_text_list = [line for line in relevant_text_list if line.strip()]
    relevant_text_list = [line.strip() for line in relevant_text_list]
    valid_references, wrong_doi_list, no_doi_references, reason_list = process_references_with_doi(relevant_text_list)
    print("Finished comparison")

    print("Generating HTML report...")
    title_dict = dict()
    for l in no_doi_references:
        result = extract_title_from_reference(l)
        if result and "," in result:
            author, title = result.split(",", 1)
        else:
            author, title = (None, result if result else None)

        if author is None and title is None:
            continue
        title_dict[l] = (author, title)
    
    for l in title_dict.keys():
        papers = query_crossref_by_title(title_dict[l][1], title_dict[l][0])
        doi_list = []
        for p in papers:
            doi_list.append(p["doi"])
        title_dict[l] = (title_dict[l][0], title_dict[l][1], doi_list)
    
    with open(f'html_report/{FILE_NAME}_report.html', 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Reference Validation Report</title>
            <style>
                a { color: #0066cc; text-decoration: underline; }
                a:hover { color: #004499; }
            </style>
        </head>
        <body>
        """)

        # section for wrong_doi_list
        f.write(f'<div class="section"><h1>Wrong DOI List ({len(wrong_doi_list)})</h1><ul>')
        for ref in wrong_doi_list:
            clickable_ref = make_doi_clickable(ref)
            f.write(f'<li><b>{clickable_ref}</b><br><em>Reason:</em> {reason_list[ref]}</li>')
        f.write('</ul></div>')

        # section for no_doi_references
        f.write(f'<div class="section"><h1>No DOI References ({len(no_doi_references)})</h1><ul>')
        for ref in no_doi_references:
            possible_dois = title_dict.get(ref, (None, None, []))[2]
            if possible_dois:
                possible_dois = [make_doi_clickable(f"doi:{doi}", False) for doi in possible_dois]
                ref += "<br>Possible DOIs: " + " |||| ".join(possible_dois) + "<br><br>"
            f.write(f'<li>{ref}</li>\n')
        f.write('</ul></div>')

        # section for valid_references
        f.write(f'<div class="section"><h1>Valid References ({len(valid_references)})</h1><ul>')
        for ref in valid_references:
            clickable_ref = make_doi_clickable(ref)
            f.write(f'<li>{clickable_ref}</li>')
        f.write('</ul></div>')
        f.write("</body></html>")
    
    print("DONE!")
