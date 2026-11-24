"""Parse URSI SAO-4 ionosonde files from EMBRACE (INPE) to extract scaled
characteristics, in particular foEs / fminEs / fbEs / Type Es (Group 4).

Group 4 holds 49 characteristics in F8.3 format. We locate the block by
scanning for the first run of >=45 parseable 8-char floats that begins with
a plausible foF2 (1-20 MHz). Returns dict of named characteristics.
"""
import re
import sys

NAMES = {
    1: "foF2", 2: "foF1", 3: "M(D)", 4: "MUF(D)", 5: "fmin",
    6: "foEs", 7: "fminF", 8: "fminE", 9: "foE", 10: "fxI",
    11: "h'F2", 12: "h'F1", 13: "h'E", 14: "h'F", 15: "h'E2",
    16: "foE2", 17: "M3000F2", 18: "M3000F1", 19: "hmF2", 20: "hmF1",
    21: "hmE", 22: "FF", 23: "FE", 24: "D", 25: "fMUF", 26: "h'(fMUF)",
    27: "delta_foF2", 28: "foEp", 29: "f(h'F)", 30: "f(h'F2)",
    31: "foF1p", 32: "peak_h_F2", 33: "peak_h_F1", 34: "zhalfNm",
    35: "foF2p", 36: "fminEs", 37: "yF2", 38: "yF1", 39: "TEC",
    40: "scale_h_F2", 41: "B0", 42: "B1", 43: "D1", 44: "foEa",
    45: "h'Ea", 46: "foP", 47: "h'P", 48: "fbEs", 49: "type_es",
}
NO_READING = 9999.0

FLOAT_RE = re.compile(r"^[ +-]?\d{1,4}\.\d{3}$")


def _chunks(text, width=8):
    """Yield consecutive fixed-width float fields starting at offset."""
    out = []
    i = 0
    n = len(text)
    while i + width <= n:
        chunk = text[i:i + width]
        try:
            out.append(float(chunk))
        except ValueError:
            return out  # stop at first non-float
        i += width
    return out


def find_group4(text):
    """Return list of 49 floats for Group 4, or None."""
    best = None
    # scan every offset; groups are packed, so try each char offset
    for off in range(len(text) - 392):
        seg = text[off:off + 392]
        vals = _chunks(seg)
        if len(vals) < 49:
            continue
        vals = vals[:49]
        # sanity: foF2 plausible, several 9999 sentinels typical
        if not (0.5 < vals[0] < 25.0):
            continue
        nmiss = sum(1 for v in vals if v >= 9999.0)
        if nmiss < 5:
            continue
        return vals
    return best


# URSI codes used in SAO-XML files (e.g. SMK29)
XML_URSI = {"00": "foF2", "30": "foEs", "31": "fxEs", "32": "fbEs",
            "33": "ftEs", "34": "h'Es", "36": "type_es"}


def parse_sao_xml(path):
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    rec = root.find(".//SAORecord")
    if rec is None:
        return None
    out = {"foF2": None, "foEs": None, "fminEs": None, "fbEs": None,
           "ftEs": None, "type_es": None}
    cl = rec.find(".//CharacteristicList")
    if cl is None:
        return out
    for el in cl.iter("URSI"):
        name = XML_URSI.get(el.get("ID"))
        if name is None:
            continue
        try:
            val = float(el.get("Val", "nan"))
        except ValueError:
            continue
        if np_is_missing(val):
            continue
        out[name] = val
    return out


def np_is_missing(v):
    return v >= 9999.0 or v != v


def parse_sao(path):
    if path.endswith(".XML") or path.endswith(".xml"):
        return parse_sao_xml(path)
    with open(path, "r", errors="replace") as f:
        text = f.read().replace("\r", "").replace("\n", "")
    vals = find_group4(text)
    if vals is None:
        return None
    out = {}
    for i, v in enumerate(vals, start=1):
        name = NAMES.get(i, f"char{i}")
        out[name] = None if v >= NO_READING else v
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = parse_sao(p)
        if r is None:
            print(p, "-> parse failed")
        else:
            print(p)
            for k in ("foF2", "foEs", "fminEs", "fbEs", "type_es", "foE", "h'F2"):
                print(f"   {k:8s} = {r[k]}")
