import io
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib.pyplot as plt
from html2image import Html2Image


# --- OpenCV ORB ---
import cv2

# --- CLIP (fallback) ---
import torch
from sklearn.metrics.pairwise import cosine_similarity
from open_clip import create_model_and_transforms, get_tokenizer

# ---------------------- CONFIG ----------------------
CATALOG_DIR = Path("catalog")
CATALOG_CSV = CATALOG_DIR / "catalog.csv"
BILL_LOG_PATH = Path("generated_bills/billing_log.csv")

st.set_page_config(page_title="CLIP/ORB Billing", page_icon="📸", layout="wide")
tab1, tab2 = st.tabs(["🧾 Billing", "📊 Analytics"])

# ---------------------- CLIP Setup (fallback) ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def load_clip_model_and_preprocess():
    return create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)

model, _, preprocess = load_clip_model_and_preprocess()
tokenizer = get_tokenizer("ViT-B-32")

def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

@st.cache_data(show_spinner=False)
def embed_image_clip(image: Image.Image) -> np.ndarray:
    # Fix EXIF orientation then preprocess
    image = ImageOps.exif_transpose(image)
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(img_tensor).float()
    emb = feats.cpu().numpy().astype("float32")
    return _l2norm(emb)

def best_query_clip_embs(image: Image.Image) -> Tuple[np.ndarray, List[int]]:
    """4 rotations (0,90,180,270); returns stacked normalized embeddings (4,D)."""
    angles = [0, 90, 180, 270]
    embs = []
    for ang in angles:
        im = image if ang == 0 else image.rotate(ang, expand=True)
        embs.append(embed_image_clip(im))
    return np.vstack(embs), angles

# ---------------------- Catalog Loading ----------------------
@st.cache_data(show_spinner=False)
def load_catalog_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["item_name", "price", "image_path"])
    df = pd.read_csv(csv_path)
    # allow optional cost_price column
    cols = ["item_name", "price", "image_path"] + (["cost_price"] if "cost_price" in df.columns else [])
    return df[cols].copy()

@st.cache_resource(show_spinner=False)
def load_and_embed_catalog_clip(df: pd.DataFrame):
    """Return maps + names + CLIP embedding matrix (normalized)."""
    price_map, embeddings, raw_images = {}, {}, {}
    for _, row in df.iterrows():
        name = str(row["item_name"]).strip()
        price = float(row["price"])
        image_path = CATALOG_DIR / row["image_path"]
        if not image_path.exists():
            continue
        try:
            img = Image.open(image_path).convert("RGB")
            price_map[name] = price
            raw_images[name] = image_path.read_bytes()
            embeddings[name] = embed_image_clip(img)  # normalized
        except Exception:
            continue

    names = list(embeddings.keys())
    if names:
        emb_matrix = np.vstack([embeddings[n] for n in names])  # (N,D)
    else:
        emb_matrix = np.zeros((0, 512), dtype="float32")
    return price_map, embeddings, raw_images, names, emb_matrix

# ---------------------- ORB Catalog Index ----------------------
@st.cache_resource(show_spinner=False)
def build_orb_index(df: pd.DataFrame):
    """
    Build ORB descriptors for each catalog image.
    Returns: dict[name] = {"desc": np.ndarray, "kp": list, "shape": (h,w)}
    """
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
    index = {}
    for _, row in df.iterrows():
        name = str(row["item_name"]).strip()
        image_path = CATALOG_DIR / row["image_path"]
        if not image_path.exists():
            continue
        try:
            # robust read (handles non-ascii paths)
            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)
            if desc is not None and len(kp) >= 10:
                index[name] = {"desc": desc, "kp": kp, "shape": gray.shape}
        except Exception:
            pass
    return index

def _rotate_pil(img: Image.Image, angle: int) -> Image.Image:
    return img if angle == 0 else img.rotate(angle, expand=True)

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def match_with_orb(query_pil: Image.Image, orb_index: Dict[str, dict],
                   min_inliers: int = 12, top_k: int = 5) -> List[Tuple[str, int]]:
    """
    Try orientations {0,90,180,270}; for each, compute ORB and match to catalog with FLANN+RANSAC.
    Returns sorted list of (name, inlier_count).
    """
    if not orb_index:
        return []

    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    index_items = list(orb_index.items())
    FLANN_INDEX_LSH = 6
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2),
        dict(checks=64)
    )

    best_scores = {}
    for ang in (0, 90, 180, 270):
        q_img = _rotate_pil(query_pil, ang)
        q_bgr = _pil_to_cv2(q_img)
        q_gray = cv2.cvtColor(q_bgr, cv2.COLOR_BGR2GRAY)
        q_kp, q_desc = orb.detectAndCompute(q_gray, None)
        if q_desc is None or len(q_kp) < 10:
            continue

        for name, data in index_items:
            c_desc = data["desc"]
            c_kp   = data["kp"]

            # kNN match (k=2) + Lowe ratio
            matches = flann.knnMatch(q_desc, c_desc, k=2)
            good = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            # RANSAC homography to count inliers (geometric consistency)
            src_pts = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([c_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0

            if (name not in best_scores) or (inliers > best_scores[name]):
                best_scores[name] = inliers

    ranked = sorted(best_scores.items(), key=lambda x: -x[1])[:top_k]
    return ranked

# ---------------------- CLIP Fallback Top-K ----------------------
def find_top_matches_clip(query_img: Image.Image,
                          catalog_names: List[str],
                          catalog_matrix: np.ndarray,
                          top_k: int = 5) -> List[Tuple[str, float]]:
    q_embs, _ = best_query_clip_embs(query_img)  # (4,D)
    if len(catalog_matrix) == 0:
        return []
    sims = q_embs @ catalog_matrix.T  # dot product == cosine (normalized)
    sims = sims.max(axis=0)           # best over rotations
    idxs = np.argsort(-sims)[:top_k]
    return [(catalog_names[i], float(sims[i])) for i in idxs]

# ---------------------- Billing App ----------------------
with tab1:
    st.title("📸 AI Billing (ORB + CLIP fallback)")
    st.caption("Upload/capture a photo → match to catalog. ORB is robust to rotation/perspective; CLIP is used as fallback.")

    catalog_df = load_catalog_df(CATALOG_CSV)
    if catalog_df.empty:
        st.warning("Catalog is empty. Fill catalog/catalog.csv and put images in catalog/images/")
    price_lookup, catalog_embeddings, catalog_images, catalog_names, catalog_matrix = load_and_embed_catalog_clip(catalog_df)
    orb_index = build_orb_index(catalog_df)

    if "cart" not in st.session_state:
        st.session_state.cart: Dict[str, Dict[str, float | int]] = {}

    st.subheader("1) Upload/Scan Image")
    cam_img = st.camera_input("📷 Capture")
    uploaded_imgs = st.file_uploader("Or upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    latest_photo: Optional[Tuple[str, bytes]] = None
    if cam_img:
        latest_photo = ("camera.png", cam_img.getvalue())
    elif uploaded_imgs:
        latest_photo = (uploaded_imgs[-1].name, uploaded_imgs[-1].read())

    st.subheader("2) AI Matching + Add to Cart")
    if latest_photo:
        fname, raw = latest_photo
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image = image.resize((256, 256))  # Resize for speed

            # Match logic
            matched_name: Optional[str] = None
            with st.spinner("Matching item with AI..."):
                orb_matches = match_with_orb(image, orb_index, min_inliers=12, top_k=5)

            if orb_matches:
                best_name, inliers = orb_matches[0]
                if inliers >= 15:
                    matched_name = best_name
                    st.success(f"✅ ORB match: **{best_name}** (inliers: {inliers})")
                else:
                    st.warning(f"Low ORB confidence (inliers={inliers}). Pick from suggestions or let CLIP help:")
                    choice = st.radio(
                        "ORB suggestions",
                        [f"{n} (inliers={s})" for n, s in orb_matches],
                        index=0, key=f"orb_sugg_{fname}"
                    )
                    matched_name = choice.split(" (")[0]

            if matched_name is None:
                clip_top = find_top_matches_clip(image, catalog_names, catalog_matrix, top_k=5)
                if clip_top:
                    c_best, c_score = clip_top[0]
                    if c_score >= 0.40:
                        matched_name = c_best
                        st.info(f"ℹ️ CLIP fallback: **{c_best}** (score: {c_score:.2f})")
                    else:
                        st.warning(f"No confident CLIP match (best={c_score:.2f}). Pick from top suggestions:")
                        suggestion = st.radio(
                            "CLIP suggestions",
                            [f"{n} ({s:.2f})" for n, s in clip_top],
                            index=0, key=f"clip_sugg_{fname}"
                        )
                        matched_name = suggestion.split(" (")[0]
                else:
                    st.warning("No match candidates. Choose manually.")
                    matched_name = st.selectbox("Choose manually", options=list(price_lookup.keys()),
                                                key=f"manual_{fname}")

            # Show matched item image only (not uploaded image)
            if matched_name:
                catalog_map = catalog_df.set_index("item_name").to_dict(orient="index")
                catalog_image_path = CATALOG_DIR / catalog_map[matched_name]["image_path"]
                if catalog_image_path.exists():
                    matched_img = Image.open(catalog_image_path).convert("RGB")
                    st.image(matched_img, caption=f"Matched: {matched_name}", width=160)

            default_price = float(price_lookup.get(matched_name, 0.0))
            qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"qty_{fname}")
            price = st.number_input("Selling Price", min_value=0.0, value=default_price, key=f"price_{fname}")
            cost_price = catalog_df.set_index("item_name").get("cost_price", pd.Series()).get(matched_name, 0.0)

            if st.button("➕ Add to Cart", key=f"add_{fname}"):
                st.session_state.cart[matched_name] = {
                    "price": price,
                    "qty": qty,
                    "cost_price": float(cost_price) if pd.notna(cost_price) else 0.0
                }
                st.success(f"Added {qty} × {matched_name}")

        except Exception as e:
            st.error(f"Failed to process image. {e}")
    else:
        st.info("Please capture or upload an image first.")

    st.divider()
    st.subheader("🛒 Cart")

    if st.session_state.cart:
        items = []
        for name, rec in st.session_state.cart.items():
            qty = int(rec["qty"])
            price = float(rec["price"])
            cost = float(rec.get("cost_price", 0.0))
            profit = (price - cost) * qty

            # Editable fields
            rec["qty"] = st.number_input(f"Qty - {name}", min_value=1, value=qty, key=f"edit_qty_{name}")
            rec["price"] = st.number_input(f"Price - {name}", min_value=0.0, value=price, key=f"edit_price_{name}")

            items.append({
                "Item": name,
                "Unit Price": price,
                "Cost Price": cost,
                "Qty": qty,
                "Total": round(price * qty, 2),
                "Profit": round(profit, 2)
            })

        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True)
        grand_total = float(df["Total"].sum())
        total_profit = float(df["Profit"].sum())

        st.metric("Total (Before Discount)", f"₹{grand_total:,.2f}")
        st.metric("Estimated Profit", f"₹{total_profit:,.2f}")

        # Discount
        discount_percent = st.slider("Apply Discount (%)", 0, 50, 0)
        discount_amt = (grand_total * discount_percent) / 100.0
        final_total = grand_total - discount_amt

        if discount_percent > 0:
            st.metric("Discount", f"-₹{discount_amt:,.2f}")

        st.metric("Grand Total (After Discount)", f"₹{final_total:,.2f}")

        st.subheader("🧾 Final Details")
        billed_to = st.text_input("Customer Name", value="Customer", key="billed_to")
        mobile = st.text_input("Mobile Number", key="billed_mobile")

        col1, col2 = st.columns(2)
        if col1.button("✅ Save & Generate Invoice"):
            if not billed_to or not mobile:
                st.error("Please fill both customer name and mobile number.")
            else:
                # Save CSV
                now = datetime.now()
                output_dir = Path("generated_bills")
                output_dir.mkdir(exist_ok=True)
                csv_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.csv")
                csv_path = output_dir / csv_filename

                df["Billed To"] = billed_to
                df["Mobile"] = mobile
                df["timestamp"] = now
                df["Grand Total"] = final_total
                if discount_percent > 0:
                    df["Discount %"] = discount_percent

                customer_df = df.drop(columns=["Cost Price", "Profit"], errors="ignore").copy()
                customer_df.to_csv(csv_path, index=False)


                # Save HTML invoice
                html_invoice = customer_df.to_html(index=False)

                # Determine discount line only if discount is applied
                discount_line = f"Discount: ₹{discount_amt:,.2f}<br/>" if discount_percent > 0 else ""

                html_template = f"""
                <html>
                <head>
                    <title>Invoice - {billed_to}</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{
                            font-family: 'Segoe UI', sans-serif;
                            padding: 30px;
                            background-color: #f9f9f9;
                            max-width: 900px;
                            margin: auto;
                            border: 2px solid #ccc;
                            border-radius: 8px;
                        }}
                        .header {{
                            text-align: center;
                            margin-bottom: 30px;
                        }}
                        .header h1 {{
                            margin: 0;
                            font-size: 2.2rem;
                            color: #2c3e50;
                        }}
                        .header p {{
                            margin: 4px 0;
                            font-size: 1.1rem;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin-top: 10px;
                        }}
                        th, td {{
                            border: 1px solid #ccc;
                            padding: 10px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #2c3e50;
                            color: white;
                        }}
                        .totals {{
                            margin-top: 20px;
                            font-size: 1.2rem;
                            font-weight: bold;
                            text-align: right;
                        }}
                        .footer {{
                            margin-top: 40px;
                            text-align: center;
                            font-style: italic;
                            color: #555;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🧾 Shyam FC Palwal</h1>
                        <p><b>Customer:</b> {billed_to} | <b>Mobile:</b> {mobile}</p>
                        <p><b>Date:</b> {now.strftime('%d-%m-%Y')}</p>
                    </div>
                    {html_invoice}
                    <p class="totals">
                           {discount_line}
                       Grand Total: ₹{final_total:,.2f}</p>
                    <div class="footer">
                        <p>Thank you for choosing Shyam FC Palwal!</p>
                    </div>
                </body>
                </html>
                """

                html_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.html")
                html_path = output_dir / html_filename
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_template)

                # WhatsApp link (use your current ngrok/domain)
                # --- Convert the just-saved HTML invoice to a PNG image and show download buttons ---
                try:
                    # Ensure output dir exists (it already does above; this is just defensive)
                    output_dir.mkdir(exist_ok=True)

                    # Initialize html2image to render into the same folder
                    hti = Html2Image(output_path=str(output_dir))

                    # Build a PNG file name next to the HTML
                    png_filename = now.strftime(f"{billed_to}_%Y%m%d_%H%M%S.png")
                    png_path = output_dir / png_filename

                    # Render HTML → PNG
                    hti.screenshot(html_file=str(html_path), save_as=png_filename)

                    # Show download buttons
                    with open(png_path, "rb") as img_f:
                        st.download_button(
                            label="📥 Download Invoice Image (PNG)",
                            data=img_f,
                            file_name=png_filename,
                            mime="image/png"
                        )

                    with open(html_path, "rb") as html_f:
                        st.download_button(
                            label="⬇️ Download Invoice (HTML)",
                            data=html_f,
                            file_name=html_filename,
                            mime="text/html"
                        )

                except Exception as ex:
                    st.warning(f"Could not render PNG from HTML automatically: {ex}")

                # Append to billing_log.csv for analytics
                if BILL_LOG_PATH.exists():
                    df_log = pd.read_csv(BILL_LOG_PATH)
                    df_combined = pd.concat([df_log, df], ignore_index=True)
                else:
                    df_combined = df
                df_combined.to_csv(BILL_LOG_PATH, index=False)

        if col2.button("🗑️ Clear Cart"):
            st.session_state.cart.clear()
            st.experimental_rerun()
    else:
        st.info("No items in cart.")

# ---------------------- Analytics ----------------------
with tab2:
    st.subheader("📊 Billing Analytics")
    if not BILL_LOG_PATH.exists():
        st.warning("No billing history found yet.")
    else:
        @st.cache_data(ttl=600)
        def load_billing_log():
            return pd.read_csv(BILL_LOG_PATH, parse_dates=["timestamp"], dayfirst=True, infer_datetime_format=True)


        df_log = load_billing_log()

        if "timestamp" in df_log.columns:
            df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors='coerce')
            df_log.dropna(subset=["timestamp"], inplace=True)

        total_bills = len(df_log)
        total_sales = df_log["Grand Total"].sum()
        total_profit = df_log.get("Profit", pd.Series([0]*len(df_log))).sum()

        top_customers = df_log.groupby(["Billed To", "Mobile"]).agg({"Grand Total": "sum", "Profit": "sum"})
        top_customers = top_customers.sort_values(by="Grand Total", ascending=False).head(5)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Bills", total_bills)
        col2.metric("Total Sales", f"₹{total_sales:,.2f}")
        col3.metric("Total Profit", f"₹{total_profit:,.2f}")

        st.markdown("### 🧍‍♂️ Top Customers by Sales")
        st.dataframe(top_customers.rename(columns={"Grand Total": "Total Spend", "Profit": "Total Profit"}))

        df_log["Month"] = df_log["timestamp"].dt.to_period("M").astype(str)
        monthly_profit = df_log.groupby("Month").agg({"Grand Total": "sum", "Profit": "sum"}).reset_index()

        st.markdown("### 📅 Monthly Summary")
        fig, ax = plt.subplots()
        ax.plot(monthly_profit["Month"], monthly_profit["Grand Total"], marker="o", label="Sales")
        ax.plot(monthly_profit["Month"], monthly_profit["Profit"], marker="x", label="Profit", linestyle="--")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        ax.set_title("Monthly Sales & Profit Trend")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.download_button(
            "⬇️ Download Full Billing Log",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="billing_log.csv",
            mime="text/csv"
        )
