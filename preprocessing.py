"""
Modul Preprocessing Teks Bahasa Indonesia
Untuk klasifikasi sarkasme pada dataset Reddit dan Twitter Indonesia.
"""

import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Inisialisasi stemmer Sastrawi (Bahasa Indonesia)
_factory = StemmerFactory()
stemmer = _factory.createStemmer()

# Daftar stopwords Bahasa Indonesia
INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "tidak", "dari",
    "pada", "adalah", "ke", "dalam", "akan", "juga", "sudah", "saya", "ada",
    "bisa", "atau", "mereka", "satu", "telah", "oleh", "kita", "kami",
    "ia", "dia", "apa", "jika", "maka", "karena", "seperti", "tapi",
    "namun", "hanya", "semua", "lebih", "lain", "anda", "begitu",
    "antara", "lagi", "harus", "banyak", "bahwa", "setelah", "belum",
    "kalau", "kan", "ya", "nya", "pun", "nih", "sih", "dong", "deh",
    "loh", "kok", "tuh", "gitu", "gini", "mah", "nah", "yah", "dah",
    "udah", "aja", "doang", "banget", "bgt", "yg", "dgn", "utk", "dlm",
    "tdk", "gak", "ga", "gk", "ngga", "nggak", "enggak", "tak",
    "se", "si", "sang", "para",
}


def clean_text(text):
    """Membersihkan teks dari noise."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Hapus URL
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Hapus mention (@user) dan hashtag (#topic)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Hapus karakter HTML entity
    text = re.sub(r"&\w+;", "", text)

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Hapus extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_slang(text):
    """Normalisasi kata slang/singkatan Bahasa Indonesia."""
    slang_dict = {
        "gak": "tidak", "ga": "tidak", "gk": "tidak", "ngga": "tidak",
        "nggak": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",
        "yg": "yang", "dgn": "dengan", "utk": "untuk", "dlm": "dalam",
        "krn": "karena", "bgt": "banget", "bkn": "bukan", "blm": "belum",
        "bnr": "benar", "bs": "bisa", "dr": "dari", "dpt": "dapat",
        "emg": "memang", "gmn": "gimana", "gpp": "tidak apa apa",
        "hrs": "harus", "jd": "jadi", "jg": "juga", "klo": "kalau",
        "kyk": "kayak", "lg": "lagi", "msh": "masih", "org": "orang",
        "pd": "pada", "sdh": "sudah", "udh": "sudah", "udah": "sudah",
        "sgt": "sangat", "sm": "sama", "spy": "supaya", "sy": "saya",
        "trs": "terus", "tp": "tapi", "trm": "terima", "wkwk": "haha",
        "wkwkwk": "haha", "wkwkwkwk": "haha", "hehe": "haha",
        "hihi": "haha", "xixi": "haha", "kwkw": "haha",
        "emang": "memang", "gimana": "bagaimana", "kayak": "seperti",
        "banget": "sekali", "aja": "saja", "doang": "saja",
        "biar": "supaya", "abis": "habis", "gue": "saya", "gw": "saya",
        "lo": "kamu", "lu": "kamu", "elu": "kamu",
    }

    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized)


def remove_stopwords(text):
    """Menghapus stopwords Bahasa Indonesia."""
    words = text.split()
    filtered = [w for w in words if w not in INDONESIAN_STOPWORDS]
    return " ".join(filtered)


def stem_text(text):
    """Stemming menggunakan Sastrawi."""
    return stemmer.stem(text)


def preprocess_text(text, use_stemming=True, use_stopword_removal=True,
                    use_slang_normalization=True):
    """
    Pipeline preprocessing lengkap untuk teks Bahasa Indonesia.

    Parameters
    ----------
    text : str
        Teks mentah yang akan diproses.
    use_stemming : bool
        Apakah menggunakan stemming.
    use_stopword_removal : bool
        Apakah menghapus stopwords.
    use_slang_normalization : bool
        Apakah menormalisasi kata slang.

    Returns
    -------
    str
        Teks yang sudah diproses.
    """
    text = clean_text(text)

    if use_slang_normalization:
        text = normalize_slang(text)

    if use_stopword_removal:
        text = remove_stopwords(text)

    if use_stemming:
        text = stem_text(text)

    return text
