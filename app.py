import os
import json
import streamlit as st
from PIL import Image

from utils import (
    ensure_dirs,
    init_metadata_csv,
    save_found_item,
    load_found_items,
    filter_items_text,
    build_embedding_index,
    search_by_image,
)

APP_TITLE = "RAVE SEARCH"

ensure_dirs()
init_metadata_csv()

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")

# --- Session ---
if "role" not in st.session_state:
    st.session_state.role = "landing"
if "finder_cart" not in st.session_state:
    st.session_state.finder_cart = []
if "finder_info" not in st.session_state:
    st.session_state.finder_info = {
        "full_name": "",
        "contact": "",
        "person_type": "Student",
        "student_id": "",
        "purpose": "Visitor",
        "purpose_other": "",
    }

def go(role: str):
    st.session_state.role = role

def logout():
    st.session_state.role = "landing"
    st.session_state.finder_cart = []
    st.rerun()

# --- Header ---
st.markdown(
    f"""
    <div style="padding:14px;border-radius:12px;background:#0f172a;border:1px solid #1f2937;">
      <h2 style="margin:0;color:white;">🔎 {APP_TITLE}</h2>
      <p style="margin:6px 0 0 0;color:#cbd5e1;">
        Lost & Found system with Computer Vision matching (Python + ML model).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --- Landing ---
if st.session_state.role == "landing":
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Finder")
        st.write("If you found an item, register it here.")
        if st.button("I Found Something", use_container_width=True):
            go("finder")
            st.rerun()

    with c2:
        st.subheader("Owner")
        st.write("If you lost an item, search here.")
        if st.button("I Lost Something", use_container_width=True):
            go("owner")
            st.rerun()

    st.info("This version uses a real CV model (ResNet50 embeddings) for image similarity search.")
    st.stop()

# --- Sidebar ---
st.sidebar.success(f"View: {st.session_state.role.upper()}")
st.sidebar.button("Logout", on_click=logout)

# ======================================================================
# FINDER
# ======================================================================
if st.session_state.role == "finder":
    st.subheader("🧾 Finder Dashboard")

    st.caption(
        "Fill your details (required), then add one or more found items. "
        "For each item, place is required. Photo and item name are optional, but at least one must be provided."
    )

    # Finder info (required)
    with st.container(border=True):
        st.markdown("### Finder details (required)")
        fi = st.session_state.finder_info

        colA, colB = st.columns(2)
        with colA:
            fi["full_name"] = st.text_input("Full name *", value=fi["full_name"])
            fi["contact"] = st.text_input("Contact number *", value=fi["contact"], placeholder="+44... or digits")
        with colB:
            fi["person_type"] = st.selectbox("Person type *", ["Student", "Non-student"], index=0 if fi["person_type"] == "Student" else 1)

            if fi["person_type"] == "Student":
                fi["student_id"] = st.text_input("Student ID *", value=fi["student_id"])
                fi["purpose"] = "Visitor"
                fi["purpose_other"] = ""
            else:
                fi["purpose"] = st.selectbox("Purpose *", ["Visitor", "Cleaner", "Crew", "Other"], index=["Visitor","Cleaner","Crew","Other"].index(fi["purpose"]) if fi["purpose"] in ["Visitor","Cleaner","Crew","Other"] else 0)
                if fi["purpose"] == "Other":
                    fi["purpose_other"] = st.text_input("Purpose (Other) *", value=fi["purpose_other"])
                else:
                    fi["purpose_other"] = ""
                fi["student_id"] = ""

        st.session_state.finder_info = fi

    st.write("")

    # Add item form
    with st.form("add_item", clear_on_submit=True):
        st.markdown("### Add found item")

        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
        with col1:
            item_name = st.text_input("Item name (optional)", placeholder="e.g., wallet, keys")
        with col2:
            found_place = st.text_input("Found place *", placeholder="e.g., Library, Science Lab")
        with col3:
            photo = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png", "webp"])

        add_btn = st.form_submit_button("➕ Add to list", use_container_width=True)

        if add_btn:
            # Validate finder details
            fi = st.session_state.finder_info
            if not fi["full_name"].strip():
                st.error("Full name is required.")
            elif not fi["contact"].strip():
                st.error("Contact number is required.")
            elif fi["person_type"] == "Student" and not fi["student_id"].strip():
                st.error("Student ID is required for students.")
            elif fi["person_type"] == "Non-student":
                if not fi["purpose"].strip():
                    st.error("Purpose is required for non-students.")
                elif fi["purpose"] == "Other" and not fi["purpose_other"].strip():
                    st.error("Please enter your purpose.")
            # Validate item fields
            elif not found_place.strip():
                st.error("Found place is required.")
            elif (photo is None) and (not item_name.strip()):
                st.error("Please provide at least a photo or an item name.")
            else:
                st.session_state.finder_cart.append({
                    "item_name": item_name.strip(),
                    "found_place": found_place.strip(),
                    "photo_bytes": photo.getvalue() if photo else None,
                    "photo_ext": photo.name.split(".")[-1] if (photo and "." in photo.name) else None,
                })
                st.success("Added. You can add more items.")

    st.write("")
    st.markdown("### 📦 Items to save")

    if len(st.session_state.finder_cart) == 0:
        st.info("No items added yet.")
    else:
        for idx, it in enumerate(st.session_state.finder_cart):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    if it["photo_bytes"]:
                        st.image(it["photo_bytes"], use_container_width=True)
                    else:
                        st.caption("No photo")
                with c2:
                    st.write(f"**Name:** {it['item_name'] or 'Unnamed item'}")
                    st.write(f"**Found place:** {it['found_place']}")
                with c3:
                    if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                        st.session_state.finder_cart.pop(idx)
                        st.rerun()

        colS, colI = st.columns(2)
        with colS:
            if st.button("💾 Save all items", use_container_width=True):
                fi = st.session_state.finder_info
                saved = 0
                for it in st.session_state.finder_cart:
                    save_found_item(
                        item_name=it["item_name"],
                        found_place=it["found_place"],
                        photo_bytes=it["photo_bytes"],
                        photo_ext=it["photo_ext"],
                        finder_full_name=fi["full_name"],
                        finder_contact=fi["contact"],
                        finder_person_type=fi["person_type"],
                        finder_student_id=fi["student_id"],
                        finder_purpose=fi["purpose"],
                        finder_purpose_other=fi["purpose_other"],
                    )
                    saved += 1
                st.session_state.finder_cart = []
                st.success(f"Saved {saved} item(s).")

        with colI:
            if st.button("🧠 Build / Update image index", use_container_width=True):
                try:
                    build_embedding_index()
                    st.success("Image index built successfully.")
                except Exception as e:
                    st.error(str(e))

    st.write("")
    items = load_found_items()
    st.caption(f"Total items in database: {len(items)}")
    if items:
        st.dataframe(items, use_container_width=True)

# ======================================================================
# OWNER
# ======================================================================
if st.session_state.role == "owner":
    st.subheader("🧍 Owner Dashboard")

    st.caption(
        "Search by item name and place lost. You can also upload a photo to find visually similar items."
    )

    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        q_name = st.text_input("Item name *", placeholder="e.g., wallet")
    with col2:
        q_place = st.text_input("Place lost *", placeholder="e.g., Library")
    with col3:
        q_photo = st.file_uploader("Optional: upload photo", type=["jpg", "jpeg", "png", "webp"])

    if st.button("🔎 Search", use_container_width=True):
        if not q_name.strip() or not q_place.strip():
            st.error("Item name and place are required.")
            st.stop()

        all_items = load_found_items()
        if not all_items:
            st.warning("No items in the database yet.")
            st.stop()

        # Text filter first (required fields)
        text_matches = filter_items_text(all_items, q_name, q_place)

        if not text_matches:
            st.warning("Sorry, no item we got.")
            st.stop()

        # If photo provided, rank by CV similarity (ResNet50 embeddings)
        if q_photo is not None:
            try:
                img = Image.open(q_photo).convert("RGB")
                ranked = search_by_image(img, top_k=20)

                # Keep only those also matching text/place
                allowed_ids = {m["item_id"] for m in text_matches}
                ranked = [(m, s) for (m, s) in ranked if m["item_id"] in allowed_ids]
                ranked = ranked[:5] if ranked else [(m, None) for m in text_matches[:5]]
            except Exception as e:
                st.error(f"Image search failed: {e}")
                ranked = [(m, None) for m in text_matches[:5]]
        else:
            ranked = [(m, None) for m in text_matches[:5]]

        st.success("Item found!")
        st.info("We received your item. Please contact Lost and Found, bring evidence that it is yours, and collect it.")

        cols = st.columns(min(5, len(ranked)))
        for i, (m, score) in enumerate(ranked):
            with cols[i]:
                if m.get("image_path"):
                    st.image(m["image_path"], use_container_width=True)
                st.write(f"**{m.get('item_name') or 'Unnamed item'}**")
                st.caption(f"Found place: {m.get('found_place')}")
                st.caption(f"Date: {m.get('date_found')}")
                st.caption(f"ID: {m.get('item_id')}")
                if score is not None:
                    st.caption(f"Image similarity: {score:.3f}")
