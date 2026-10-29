import utils
import config
import re
import pandas as pd
from tqdm import tqdm

tqdm.pandas() 

df_data = utils.get_df_data()

### remove rows with blank DOI
mask = df_data[config.COLUMN_DOI].isna() | (df_data[config.COLUMN_DOI].astype(str).str.strip() == '')
if mask.sum() > 0:
    df_data = df_data[~mask]

### convert year column from str to int
print("Converting year column to int...")
if df_data["Year Published"].dtype.kind not in {'i', 'u'}:
    df_data["Year Published"] = df_data["Year Published"].astype(int)

### convert citation count column from str to int
print("Converting citation count column to int...")
if df_data["Total Times Cited Count"].dtype.kind not in {'i', 'u'}:
    df_data["Total Times Cited Count"] = df_data["Total Times Cited Count"].astype(int)


### split author lists
print("Processing author lists...")
if "authors_list" not in df_data.columns:
    df_data["authors_list"] = df_data["Authors"].str.split(";").apply(lambda x: [a.strip() for a in x])
if "authors_full_name_list" not in df_data.columns:
    df_data["authors_full_name_list"] = df_data["Author Full Name"].str.split(";").apply(lambda x: [a.strip() for a in x])

### extract author institutions
print("Processing author institutions...")
if "authors_institution" not in df_data.columns or "authors_country" not in df_data.columns:
    COUNTRY_CORRECTIONS = {
        'United States of America': 'United States',
        "USA": "United States",
        "U Arab Emirates": "United Arab Emirates",
        "Turkiye": "Turkey",
        "Viet Nam": "Vietnam",
        "BELARUS": "Belarus",
        "DEM REP CONGO": "Democratic Republic of the Congo",
        "IL 6020 USA": "United States",
        "Indian Inst Technol Jodhpur": "India",
    }
    # Regex for generic patterns
    USA_PATTERN = re.compile(r"[A-Z]{2}\s\d{5}\sUSA$", re.IGNORECASE)    
    USA_PATTERN_2 = re.compile(r"[A-Z]{2}\sUSA$", re.IGNORECASE)    

    def split_outside_brackets(s, sep=';'):
        """Split s by sep ignoring sep occurrences inside square brackets [ ... ]."""
        if pd.isna(s) or not isinstance(s, str):
            return []
        parts = []
        buf = []
        level = 0
        for ch in s:
            if ch == '[':
                level += 1
                buf.append(ch)
            elif ch == ']':
                level = max(level - 1, 0)
                buf.append(ch)
            elif ch == sep and level == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)
        return parts

    def extract_authors_block(block):
        """Return list of authors extracted from the content inside square brackets [ ... ]."""
        m = re.search(r'\[(.*?)\]', block)
        if not m:
            return []
        authors_str = m.group(1)
        authors = [a.strip() for a in authors_str.split(';') if a.strip()]
        return authors

    def extract_institution_from_block(block):
        """
        Get the institution from the block: text after ']' up to the first comma.
        E.g.: "] Politech Univ Timisoara, Dept Comp..." -> "Politech Univ Timisoara"
        """
        if ']' in block:
            after = block.split(']', 1)[1].strip()
        else:
            after = block.strip()
        if not after:
            return "Unknown"
        inst = after.split(',', 1)[0].strip()
        inst = inst.lstrip('.,;:')
        if not inst:
            return "Unknown"
        
        if inst == "Cent S Univ":
            inst = "Cent South Univ"

        return inst

    def extract_country_from_block(block):
        """
        Get the country from the block: last element after the last comma.
        E.g.: "] ..., Timisoara, Romania" -> "Romania"
        """
        if ']' in block:
            after = block.split(']', 1)[1].strip()
        else:
            after = block.strip()
        if not after:
            return "Unknown"
        parts = [p.strip() for p in after.split(',') if p.strip()]
        country_raw = parts[-1] if parts else "Unknown"

        # Direct correction via exception list
        country = COUNTRY_CORRECTIONS.get(country_raw, country_raw)

        # Regex correction for generic patterns
        if USA_PATTERN.search(country):
            return "United States"
        if USA_PATTERN_2.search(country):
            return "United States"
    
        return country

    def normalize_name(name):
        """Normalize name for matching: remove dots, collapse spaces, and lower()"""
        if name is None:
            return name
        n = re.sub(r'\.', '', name)            
        n = re.sub(r'\s+', ' ', n)             
        n = n.strip()
        n = n.lower()
        return n

    def map_row_authors_institution_country(row):
        """
        Receives a row with:
        - 'authors_full_name_list' : list of authors
        - 'Author Address' : complex string to be parsed
        Returns two lists aligned with authors_full_name_list: institutions and countries
        """
        authors_master = row["authors_full_name_list"]
        author_address = row["Author Address"] or ""

        # CASE 1 — address without brackets -> applies to all authors
        if "[" not in author_address and "]" not in author_address:
            inst = extract_institution_from_block(author_address)
            country = extract_country_from_block(author_address)

            mapped_inst = [inst or "Unknown"] * len(authors_master)
            mapped_country = [country or "Unknown"] * len(authors_master)
        # CASE 2 — address with brackets -> follow standard parsing
        else: 

            # 1) split into blocks (respecting brackets)
            address_blocks = split_outside_brackets(author_address, sep=';')

            # 2) build author_normalized -> institution / country maps (first occurrence prevails)
            author_to_inst = {}
            author_to_country = {}

            for block in address_blocks:
                authors_in_block = extract_authors_block(block)
                inst = extract_institution_from_block(block)
                country = extract_country_from_block(block)

                for a in authors_in_block:
                    key = normalize_name(a)
                    if key not in author_to_inst:
                        author_to_inst[key] = inst
                    if key not in author_to_country:
                        author_to_country[key] = country

            # 3) map master list to institutions and countries
            mapped_inst = []
            mapped_country = []

            for a in authors_master:
                key = normalize_name(a)
                mapped_inst.append(author_to_inst.get(key, "Unknown"))
                mapped_country.append(author_to_country.get(key, "Unknown"))

        return mapped_inst, mapped_country

    # 4) Apply to DataFrame using progress_apply
    df_data[["authors_institution", "authors_country"]] = pd.DataFrame(
        df_data.progress_apply(map_row_authors_institution_country, axis=1).tolist(),
        index=df_data.index
    )

    # Check if the 4 columns have the same list length
    inconsistent = df_data[
        ~df_data.apply(
            lambda row: len(row["authors_list"]) == len(row["authors_full_name_list"]) == len(row["authors_institution"]) == len(row["authors_country"]),
            axis=1
        )
    ]
    if len(inconsistent) > 0:   
        print(inconsistent[["authors_list", "authors_full_name_list", "authors_institution", "authors_country"]].head())
    else:
        print("All rows are consistent across the 4 author, institution, and country columns.")

### first and last author, institution, and country
print("Processing first and last author, institution, and country...")
if "first_author" not in df_data.columns:
    df_data["first_author"] = df_data["authors_list"].str[0]
if "last_author" not in df_data.columns:
    df_data["last_author"] = df_data["authors_list"].str[-1]
if "first_author_full_name" not in df_data.columns:
    df_data["first_author_full_name"] = df_data["authors_full_name_list"].str[0]
if "last_author_full_name" not in df_data.columns:
    df_data["last_author_full_name"] = df_data["authors_full_name_list"].str[-1]
if "first_institution" not in df_data.columns:
    df_data["first_institution"] = df_data["authors_institution"].str[0]
if "last_institution" not in df_data.columns:
    df_data["last_institution"] = df_data["authors_institution"].str[-1]
if "first_country" not in df_data.columns:
    df_data["first_country"] = df_data["authors_country"].apply(
                        lambda x: next((c for c in x if c != "Unknown"), "Unknown")
                    )
if "last_country" not in df_data.columns:
    df_data["last_country"] = df_data["authors_country"].str[-1]


### regions for first and last author countries
print("Mapping countries to regions...")
if "first_country_region" not in df_data.columns or "last_country_region" not in df_data.columns:
    country_region_map = {
        'United States': 'North America',
        'United States of America': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'England': 'Europe',
        'Chile': 'South America',
        'Colombia': 'South America',
        'Germany': 'Europe',
        'France': 'Europe',
        'United Kingdom': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'Russia': 'Europe',
        'China': 'Asia',
        'Peoples R China': 'Asia',
        'Japan': 'Asia',
        'India': 'Asia',
        'South Korea': 'Asia',
        'Turkey': 'Europe',
        'Iran': 'Asia',
        'Egypt': 'Africa',
        'South Africa': 'Africa',
        'Australia': 'Oceania',
        'Georgia': 'Europe',
        'Taiwan': 'Asia',
        'Singapore': 'Asia',
        'Belgium': 'Europe',
        'Sweden': 'Europe',
        'Netherlands': 'Europe',
        'Denmark': 'Europe',
        'Russian Federation': 'Europe',
        'Ireland': 'Europe',
        'North Ireland': 'Europe',
        'Morocco': 'Africa',
        'Switzerland': 'Europe',
        'Saudi Arabia': 'Asia',
        'Finland': 'Europe',
        'Portugal': 'Europe',
        'Qatar': 'Asia',
        'Luxembourg': 'Europe',
        'Oman': 'Asia',
        'Serbia': 'Europe',
        'Pakistan': 'Asia',
        'Malaysia': 'Asia',
        'Chad': 'Africa',
        'Greece': 'Europe',
        'Oman': 'Asia',
        'United Arab Emirates': 'Asia',
        'Bulgaria': 'Europe',
        'Niger': 'Africa',
        "Nigeria": 'Africa',
        'Poland': 'Europe',
        'Bangladesh': 'Asia',
        'New Zealand': 'Oceania',
        'Vietnam': 'Asia',
        'Israel': 'Asia',
        'Czechia': 'Europe',
        'Czech Republic': 'Europe',
        'Namibia': 'Africa',
        'Mali': 'Africa',
        'Lithuania': 'Europe',
        'Mauritius': 'Africa',
        'Ukraine': 'Europe',
        'Anguilla': 'North America',
        'Brunei Darussalam': 'Asia',
        'Thailand': 'Asia',
        'Croatia': 'Europe',
        'Slovenia': 'Europe',
        'Norway': 'Europe',
        'Belarus': 'Europe',
        'Bahrain': 'Asia',
        'Austria': 'Europe',
        'Indonesia': 'Asia',
        'Bhutan': 'Asia',
        'Algeria': 'Asia',
        'Slovakia': 'Europe',
        'Haiti': 'North America',
        'Estonia': 'Europe',
        'Peru': 'South America',
        'Philippines': 'Asia',
        'Ethiopia': 'Africa',
        'Jordan': 'Asia',
        'Cuba': 'North America',
        'Ghana': 'Africa',
        'Hungary': 'Europe',
        'Ecuador': 'South America',
        'Iraq': 'Asia',
        'Lebanon': 'Asia',
        'Cameroon': 'Africa',
        'Puerto Rico': 'North America',
        'Jamaica': 'North America',
        'Mongolia': 'Asia',
        'Latvia': 'Europe',
        'Tunisia': 'Africa',
        'Iceland': 'Europe',
        'Kazakhstan': 'Asia',
        'Uruguay': 'South America',
        'Romania': 'Europe',
        "Scotland": 'Europe',
        "Sri Lanka": 'Asia',
        "Yemen": 'Asia',
        "Wales": 'Europe',
        "Nepal": 'Asia',
        "Uzbekistan": 'Asia',
        "Cyprus": 'Europe',
        "Trinidad Tobago": 'South America',
        "North Korea": 'Asia',
        "Myanmar": 'Asia',
        "Zimbabwe": 'Africa',
        "Rwanda": 'Africa',
        "Democratic Republic of the Congo": 'Africa',
        "Kuwait": 'Asia',
        "Palestine": 'Asia',
        "Nepal": 'Asia',
        "Afghanistan": 'Asia',
        "Armenia": 'Asia',
        "Togo": 'Africa',
        "Bolivia": 'South America',
        "Tanzania": 'Africa',
        "Uganda": 'Africa',
        "Honduras": 'North America',
        "Montenegro": 'Europe',
        "Azerbaijan": 'Asia',
        "Kenya": 'Africa',
        "Armenia": 'Asia',
        "Brunei": 'Asia',
        "Costa Rica": 'North America',
        "Zambia": 'Africa',
        "Moldova": 'Europe',
        "Syria": 'Asia',
        "Sudan": 'Africa',
        "Venezuela": 'South America',
        "Malta": 'Europe',
        "Barbados": 'North America',
    }
    df_data["first_country_region"] = df_data["first_country"].map(country_region_map).fillna("Others")
    df_data["last_country_region"] = df_data["last_country"].map(country_region_map).fillna("Others")

# Fix issue where year is recorded as e.g. 12 instead of 2012
if df_data["Year Published"].min() < 1900:
    print("Fixing invalid year values...")
    df_data.loc[df_data["Year Published"] < 1900, "Year Published"] += 2000

# Rename topic "Advanced Separators for Enhanced TS & EC Performance in Li-Ion"
# to "Thermally & Electrochemically Stable Li-ion Separators"
if len(df_data[df_data["topic_cluster_4"] == "Advanced Separators for Enhanced TS & EC Performance in Li-Ion"]) > 0:
    print("Renaming topic 'Advanced Separators for Enhanced TS & EC Performance in Li-Ion'...")
    df_data.loc[df_data["topic_cluster_4"] == "Advanced Separators for Enhanced TS & EC Performance in Li-Ion", "topic_cluster_4"] = "Thermally & Electrochemically Stable Li-ion Separators"


# Add average citations per year column
if config.COLUMN_CITATIONS_PER_YEAR not in df_data.columns:
    print("Calculating citations per year...")
    FINAL_YEAR = 2025
    # Calculate number of years since publication (minimum = 1)
    anos_desde_publicacao = (
        FINAL_YEAR - df_data[config.COLUMN_YEAR] + 1
    ).clip(lower=1)

    # Calculate average citations per year
    df_data[config.COLUMN_CITATIONS_PER_YEAR] = (
        df_data[config.COLUMN_CITATIONS_COUNT] / anos_desde_publicacao
    )

    # # Optional: replace infinite or NaN values with 0
    # df_data[config.COLUMN_CITATIONS_PER_YEAR] = (
    #     df_data[config.COLUMN_CITATIONS_PER_YEAR]
    #     .replace([np.inf, -np.inf], np.nan)
    #     .fillna(0)
    # )



# Add journal impact factor column
if config.COLUMN_JOURNAL_IF not in df_data.columns:
    print("Calculating journal impact factor...")
   
    # Include the impact factor of each journal
    from impact_factor.core import Factor
    fa = Factor()
    def obtem_fi(row):
        issn, eissn, journal = row['ISSN'], row['eISSN'], row[config.COLUMN_JOURNAL]

        if eissn != 'UNKNOWN':
            chave = eissn
        elif issn != 'UNKNOWN':
            chave = issn   
        elif journal != 'UNKNOWN':
            chave = journal.lower()
        else:
            return None 
        
        ifs = fa.search(chave)

        if len(ifs) == 0:
            return None
        
        return ifs[0]["factor"]


    # Apply function to extract impact factors
    df_data[config.COLUMN_JOURNAL_IF] = df_data.progress_apply(obtem_fi, axis=1)











print("Saving updated DataFrame...")
utils.save_df_data(df_data)
utils.export_df_to_csv(df_data)
