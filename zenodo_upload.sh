#!/bin/bash
set -e
cd "$(dirname "$0")"

: "${ZENODO_TOKEN:?Set ZENODO_TOKEN first (https://zenodo.org/account/settings/applications/tokens/new/)}"
GH_URL="https://github.com/mablue/sphere-noise"
AUTHOR="Azizi, Masoud"
ORCID="0000-0002-8864-0533"
AFFIL="University of Tabriz, Tabriz, Iran"
TITLE="Rotation as Invertible Noise: Geometry, Stabilizers, and Multi-view Identification on the Sphere"

[ -f paper.pdf ] || { echo "paper.pdf missing — run ./build.sh first"; exit 1; }
command -v jq >/dev/null || { echo "jq missing — apt/brew install jq"; exit 1; }

echo "==> Creating deposition"
RESP=$(curl -s -X POST "https://zenodo.org/api/depositions" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/json" -d '{}')
ID=$(echo "$RESP" | jq -r .id)
BUCKET=$(echo "$RESP" | jq -r .links.bucket)
HTML=$(echo "$RESP" | jq -r .links.html)
[ -n "$ID" ] && [ "$ID" != "null" ] || { echo "Failed to create: $RESP"; exit 1; }
echo "    id=$ID  draft=$HTML"

echo "==> Uploading paper.pdf"
curl -s -X PUT "$BUCKET/paper.pdf" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/pdf" \
  --data-binary @paper.pdf >/dev/null

echo "==> Setting metadata"
DESC="We study a generative model in which \"noise\" is a random rotation R ∈ SO(3) applied to a band-limited field f on S². The cumulative effect of any number of local strikes on S² collapses to a single rotation with only 3 degrees of freedom. With R known, recovery is information-theoretically optimal (slope 1.04 against the Cramér–Rao proxy). With R unknown, a single view leaves an SO(3)-torsor gauge that no amount of data can resolve; with two views, identification is exact up to the stabilizer of the field, Stab(f) ⊆ SO(3). This contrasts with Gaussian diffusion: rotations preserve information but induce a group-theoretic gauge.

Source code & experimental scripts: $GH_URL
Author website: https://masoudsoft.ir"

METADATA=$(jq -n \
  --arg title "$TITLE" \
  --arg desc "$DESC" \
  --arg author "$AUTHOR" \
  --arg orcid "$ORCID" \
  --arg affil "$AFFIL" \
  --arg gh "$GH_URL" \
  '{
    metadata: {
      title: $title,
      upload_type: "publication",
      publication_type: "preprint",
      publication_date: (now | strftime("%Y-%m-%d")),
      description: $desc,
      creators: [{name: $author, orcid: $orcid, affiliation: $affil}],
      keywords: ["SO(3)","rotation","spherical harmonics","diffusion models","torsor","stabilizer","equivariant networks","invertible noise"],
      license: "cc-by-4.0",
      related_identifiers: [
        {relation: "isSupplementedBy", identifier: $gh, scheme: "url"}
      ]
    }
  }')

curl -s -X PUT "https://zenodo.org/api/depositions/$ID" \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$METADATA" >/dev/null

echo ""
echo "==> DRAFT ready at: $HTML"
read -p "Publish permanently? (y/N) " yn
if [ "$yn" != "y" ]; then
  echo "Left as draft. Publish manually at: $HTML"
  exit 0
fi

echo "==> Publishing"
curl -s -X POST "https://zenodo.org/api/depositions/$ID/actions/publish" \
  -H "Authorization: Bearer $ZENODO_TOKEN" >/dev/null

DOI=$(curl -s "https://zenodo.org/api/depositions/$ID" \
  -H "Authorization: Bearer $ZENODO_TOKEN" | jq -r .doi)
echo ""
echo "=============================================================="
echo "  ✅ Published"
echo "  DOI:     https://doi.org/$DOI"
echo "  Record:  https://zenodo.org/record/$ID"
echo "  GitHub:  $GH_URL"
echo "=============================================================="
