# RAVE SEARCH (Python)

RAVE SEARCH is a Lost & Found application.

It supports two roles:
- **Finder**: registers found items (place is required). Finder must provide name and contact details and indicate whether they are a student or a non-student.
- **Owner**: searches using item name + place lost. Owners can optionally upload a photo to find visually similar items.

This project includes a working CV model:
- A pretrained **ResNet50** model extracts image embeddings.
- **Cosine similarity** is used to return the Top-5 most similar items.

## Features
- Finder can add multiple items before saving.
- Required validation: finder name/contact, student ID or purpose, and found place.
- CSV storage for metadata + local folder storage for images.
- Image similarity search using ResNet50 embeddings.

## How to run
1) Install dependencies:
```bash
pip install -r requirements.txt
