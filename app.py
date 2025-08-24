import io
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import imagehash
import xlsxwriter  # via pandas ExcelWriter engine

# ---------------------- CONFIG ----------------------
CATALOG_DIR = Path("catalog")
CATALOG_CSV = CATALOG_DIR / "catalog.csv"
IMAGES_BASE = CATALOG_DIR  # image_path in CSV is relative to /catalog
PHASH_SIZE = 16            # larger = more detail (default 8). 16 is robust.
DEFAULT_THRESHOLD = 12     # max Hamming distance to auto-accept a match
# ----------------------------------------------------

st.set_page_config(page_title="Snap → Item Sheet", page_icon="📸", layout="wide")
st.title("📸 Snap → Item Sheet (Personal)")
st.caption("Click photos, auto-recognize items against your own catalog, edit quantities, and export Excel.")

# ---------- Utilities ----------
def safe_open_image(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im

def phash_bytes(img_bytes: bytes) -> imagehash.ImageHash:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return imagehash.phash(img, hash_size=PHASH_SIZE)

def phash_image(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img.convert("RGB"), hash_size=PHASH_SIZE)

def digest_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

@st.cache_data(show_spinner=False)
def load_catalog_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["item_name", "price", "image_path"])
    df = pd.read_csv(csv_path)
    required = {"item_name", "price", "image_path"}
    missing = required - set(map(str.lower, df.columns))
    # normalize column names
    cols_map = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)
    if missing:
        st.error(f"`catalog.csv` is missing columns: {missing}. Expected: {required}")
    return df[["item_name", "price", "image_path"]].copy()

@st.cache_data(show_spinner=False)
def load_catalog_images(df: pd.DataFrame) -> Dict[str, bytes]:
    """Return {item_name -> image_bytes} for catalog images."""
    out = {}
    for _, row in df.iterrows():
        name = str(row["item_name"]).strip()
        rel = str(row["image_path"]).strip()
        full = (IMAGES_BASE / rel).resolve()
        if full.exists():
            try:
                out[name] = full.read_bytes()
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False)
def build_phash_index(df: pd.DataFrame) -> List[Tuple[str, imagehash.ImageHash]]:
    """List of (item_name, phash) for catalog images."""
    index = []
    for _, row in df.iterrows():
        name = str(row["item_name"]).strip()
        rel = str(row["image_path"]).strip()
        full = (IMAGES_BASE / rel).resolve()
        if full.exists():
            try:
                im = safe_open_image(full)
                index.append((name, phash_image(im)))
            except Exception:
                pass
    return index

def match_item_by_phash(query_bytes: bytes, index: List[Tuple[str, imagehash.ImageHash]], topk: int = 3):
    qh = phash_bytes(query_bytes)
    dists = []
    for name, h in index:
        d = qh - h  # Hamming distance
        dists.append((name, d))
    dists.sort(key=lambda x: x[1])
    return dists[:topk]

def format_money(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# ---------- Load Catalog ----------
catalog_df = load_catalog_df(CATALOG_CSV)
if catalog_df.empty:
    st.warning("Catalog is empty. Create `catalog/catalog.csv` and put images in `catalog/images/` (see example above).")
catalog_images = load_catalog_images(catalog_df)
phash_index = build_phash_index(catalog_df)

# Quick lookup maps
price_lookup = {str(r.item_name).strip(): float(r.price) for r in catalog_df.itertuples()}
image_lookup = catalog_images  # {item_name -> image_bytes}

# ---------- Session State ----------
if "cart" not in st.session_state:
    # cart is an aggregator: item_name -> {"price": float, "qty": int}
    st.session_state.cart: Dict[str, Dict[str, float | int]] = {}

if "scans" not in st.session_state:
    # keep raw scans history (optional)
    st.session_state.scans: List[Dict] = []

# ---------- Sidebar Settings ----------
with st.sidebar:
    st.header("Settings")
    currency = st.text_input("Currency symbol", "₹")
    threshold = st.slider("Match threshold (lower = stricter)", 4, 30, DEFAULT_THRESHOLD)
    st.caption("If best match distance ≤ threshold, we auto-select it. Otherwise you can pick manually.")
    if st.button("🔁 Reload catalog"):
        load_catalog_df.clear()
        load_catalog_images.clear()
        build_phash_index.clear()
        st.experimental_rerun()

# ---------- Camera / Upload ----------
st.subheader("1) Take a photo (or upload)")
cam_img = st.camera_input("Tap here to capture (mobile camera supported)")
upload_imgs = st.file_uploader("Or upload one or more photos", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True)

new_photos: List[Tuple[str, bytes]] = []
if cam_img is not None:
    raw = cam_img.getvalue()
    new_photos.append((f"camera_{digest_bytes(raw)[:10]}.png", raw))

if upload_imgs:
    for f in upload_imgs:
        new_photos.append((f.name, f.read()))

# ---------- Matching & Add to Cart ----------
st.subheader("2) Recognize & add to cart")
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    if new_photos:
        for fname, raw in new_photos:
            st.write(f"**Photo:** {fname}")
            try:
                st.image(Image.open(io.BytesIO(raw)).convert("RGB"), use_container_width=True)
            except Exception:
                st.warning("Preview failed")

            if len(phash_index) == 0:
                st.error("No catalog images indexed. Add items to catalog first.")
                continue

            top_matches = match_item_by_phash(raw, phash_index, topk=3)
            best_name, best_dist = top_matches[0]

            st.caption(f"Top matches (phash distance): " + ", ".join([f"{n} ({d})" for n, d in top_matches]))
            auto_pick = best_dist <= threshold

            if auto_pick:
                st.success(f"Auto-selected: **{best_name}**  (distance {best_dist})")
                default_item = best_name
            else:
                st.warning("Low confidence. Please select the correct item.")
                default_item = None

            # let user confirm / override
            chosen = st.selectbox("Item", options=list(price_lookup.keys()), index=(list(price_lookup.keys()).index(default_item) if (default_item in price_lookup) else 0))
            qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"qty_{fname}")
            unit_price = st.number_input("Unit price", min_value=0.0, step=0.5, value=float(price_lookup.get(chosen, 0.0)), key=f"price_{fname}")

            if st.button("➕ Add to cart", key=f"add_{fname}"):
                # aggregate into cart
                if chosen not in st.session_state.cart:
                    st.session_state.cart[chosen] = {"price": unit_price, "qty": 0}
                # If catalog had a different price but user edited, keep user's latest
                st.session_state.cart[chosen]["price"] = float(unit_price)
                st.session_state.cart[chosen]["qty"] += int(qty)

                # record scan
                st.session_state.scans.append({"source": fname, "item": chosen, "qty": qty, "price": unit_price})
                st.success(f"Added {qty} × {chosen}")

with col_right:
    st.markdown("### Cart (aggregated)")
    if st.session_state.cart:
        items = []
        for name, rec in st.session_state.cart.items():
            unit = float(rec["price"])
            q = int(rec["qty"])
            items.append({
                "Item Name": name,
                "Price per Unit": unit,
                "Total Quantity": q,
                "Total Price": round(unit * q, 2),
            })
        cart_df = pd.DataFrame(items).sort_values("Item Name")
        # allow edits
        edited = st.data_editor(
            cart_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Item Name": st.column_config.TextColumn(disabled=True),
                "Price per Unit": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.2f"),
                "Total Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                "Total Price": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            }
        )
        # persist edits back to cart
        for _, row in edited.iterrows():
            name = row["Item Name"]
            st.session_state.cart[name]["price"] = float(row["Price per Unit"])
            st.session_state.cart[name]["qty"] = int(row["Total Quantity"])

        grand_total = sum(float(v["price"]) * int(v["qty"]) for v in st.session_state.cart.values())
        st.metric("Grand Total", f"{currency}{grand_total:,.2f}")
        if st.button("🧹 Clear cart"):
            st.session_state.cart.clear()
            st.info("Cart cleared.")
    else:
        st.info("Cart is empty. Scan or upload photos and add items.")

st.divider()

# ---------- Export to Excel ----------
def build_excel(cart: Dict[str, Dict[str, float | int]], image_lookup: Dict[str, bytes]) -> bytes:
    """
    Creates an .xlsx with:
    - 'Items' sheet: Item Name, Price per Unit, Total Quantity, Total Price, Grand Total row
    - 'Images' sheet: Item Name + Thumbnail
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # Items sheet
        rows = []
        for name, rec in cart.items():
            unit = float(rec["price"])
            qty = int(rec["qty"])
            rows.append([name, unit, qty, round(unit * qty, 2)])
        items_df = pd.DataFrame(rows, columns=["Item Name", "Price per Unit", "Total Quantity", "Total Price"])
        items_df.to_excel(writer, sheet_name="Items", index=False)

        ws = writer.sheets["Items"]
        bold = wb.add_format({"bold": True})
        money = wb.add_format({"num_format": "#,##0.00"})
        # widths + header bold
        widths = [40, 18, 16, 18]
        for i, col in enumerate(items_df.columns):
            ws.write(0, i, col, bold)
            ws.set_column(i, i, widths[i])
        ws.set_column(1, 1, widths[1], money)
        ws.set_column(3, 3, widths[3], money)

        # Grand total row
        n = len(items_df)
        ws.write(n + 1, 2, "Grand Total:", bold)
        ws.write_formula(n + 1, 3, f"=SUM(D2:D{n+1})", money)

        # Images sheet
        img_ws = wb.add_worksheet("Images")
        img_ws.write(0, 0, "Item Name", bold)
        img_ws.write(0, 1, "Preview", bold)
        img_ws.set_column(0, 0, 40)
        img_ws.set_column(1, 1, 45)

        row = 1
        for name in cart.keys():
            img_ws.write(row, 0, name)
            if name in image_lookup:
                try:
                    im = Image.open(io.BytesIO(image_lookup[name])).convert("RGB")
                    im.thumbnail((320, 240))
                    bio = io.BytesIO()
                    im.save(bio, format="PNG")
                    bio.seek(0)
                    img_ws.insert_image(row, 1, f"{name}.png", {"image_data": bio})
                except Exception:
                    img_ws.write(row, 1, "Preview failed")
            else:
                img_ws.write(row, 1, "No catalog image")
            row += 10
    return output.getvalue()

st.subheader("3) Export")
if st.session_state.cart:
    file_name = f"items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_bytes = build_excel(st.session_state.cart, image_lookup)
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Excel includes: Items sheet (with totals) + Images sheet (thumbnails).")
else:
    st.info("Add items to cart to enable export.")
