# info.py====================3====Final
import streamlit as st

# Different document categories
categories = {
    "🏠 Real Estate": [
        "Lease Agreements",
        "Property Sale Deeds",
        "Rent Contracts",
        "Mortgage Documents",
        "Title Deeds"
    ],
    "🏢 Corporate & Commercial": [
        "Shareholder Agreements",
        "NDAs",
        "Articles of Incorporation",
        "Board Meeting Minutes",
        "M&A Contracts"
    ],
    "🚗 Insurance & Claims": [
        "Motor Claims",
        "Health Insurance Claims",
        "Property Insurance",
        "Claim Rejection Letters",
        "Settlement Agreements"
    ],
    "👨‍👩‍⚖️ Family & Personal Law": [
        "Divorce Papers",
        "Custody Agreements",
        "Prenuptial Agreements",
        "Adoption Forms"
    ],
    "🧾 Contracts & Agreements": [
        "Employment Contracts",
        "Consulting Agreements",
        "Licensing Agreements",
        "Vendor Agreements"
    ],
    "⚖️ Litigation & Dispute Resolution": [
        "Case Filings",
        "Court Orders",
        "Affidavits",
        "Legal Notices",
        "Arbitration Documents"
    ],
    "📑 Compliance & Regulatory": [
        "GDPR/Privacy Policies",
        "Audit Reports",
        "Financial Disclosures",
        "Environmental Compliance Docs"
    ],
    "🧠 Intellectual Property": [
        "Patent Filings",
        "Trademark Applications",
        "Copyright Licenses",
        "IP Assignment Agreements"
    ]
}

# Function to display the information about categories and subcategories
def show_info():
    with st.expander("ℹ️ Click here to see document categories and subcategories"):
        for category, subcategories in categories.items():
            # Displaying category and subcategories in one line, separated by commas
            category_text = f"**{category}**: " + ", ".join(subcategories)
            st.markdown(category_text)