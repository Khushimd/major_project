"""Patient-friendly educational copy. Not treatment advice."""

EDUCATION_TOPICS = [
    {
        "slug": "melanoma",
        "title": "What is melanoma?",
        "body": (
            "Melanoma is a type of skin cancer that can start in pigment-producing cells. "
            "It may look like a changing mole or unusual spot. Only a qualified clinician "
            "can diagnose melanoma. This app is a decision-support tool, not a diagnosis."
        ),
    },
    {
        "slug": "dermoscopy",
        "title": "What is a dermoscopic image?",
        "body": (
            "A dermoscopic image is a close-up photo taken with a dermatoscope — a lighted "
            "magnifier used on the skin. It shows pigment patterns that are hard to see with "
            "the naked eye. This system is designed for those images, not ordinary phone selfies."
        ),
    },
    {
        "slug": "dermoscopy-why",
        "title": "Why dermoscopic images are used",
        "body": (
            "Dermoscopy helps clinicians look at colour, borders, and structure inside a lesion. "
            "The AI model was trained on similar clinical photographs. Using a matching image "
            "type makes the assessment more meaningful, but it still needs a doctor’s review."
        ),
    },
    {
        "slug": "lesion",
        "title": "What is a skin lesion?",
        "body": (
            "A skin lesion is any mark, mole, patch, or growth that looks different from the "
            "surrounding skin. Most lesions are harmless. A clinician decides whether further "
            "tests (such as a biopsy) are needed."
        ),
    },
    {
        "slug": "breslow",
        "title": "What is Breslow thickness?",
        "body": (
            "Breslow thickness is how deep a melanoma grows into the skin, measured in millimetres "
            "on a pathology sample. It is entered by a clinician from lab results — the photo AI "
            "does not measure it. Thicker tumours are generally staged higher."
        ),
    },
    {
        "slug": "ulceration",
        "title": "What is ulceration?",
        "body": (
            "Ulceration means the surface of the tumour is broken when examined under the microscope. "
            "It is a pathology finding used in staging, not something the image model diagnoses."
        ),
    },
    {
        "slug": "ajcc",
        "title": "What is AJCC staging?",
        "body": (
            "AJCC (American Joint Committee on Cancer) staging describes how far melanoma has "
            "progressed using T (tumour), N (nodes), and M (metastasis) categories. In this system, "
            "stage is calculated from information your doctor enters, not from the photo alone."
        ),
    },
    {
        "slug": "stage-0",
        "title": "Understanding Stage 0",
        "body": (
            "Stage 0 (melanoma in situ) means abnormal cells are only in the top layer of skin. "
            "Your doctor will explain what this means for you. This is educational information only."
        ),
    },
    {
        "slug": "stage-i",
        "title": "Understanding Stage I",
        "body": (
            "Stage I usually means a thinner tumour without spread to lymph nodes or distant organs. "
            "Exact grouping (IA/IB) depends on thickness and ulceration from pathology."
        ),
    },
    {
        "slug": "stage-ii",
        "title": "Understanding Stage II",
        "body": (
            "Stage II generally describes thicker tumours still without known node or distant spread. "
            "Sub-stages (IIA–IIC) depend on thickness and ulceration."
        ),
    },
    {
        "slug": "stage-iii",
        "title": "Understanding Stage III",
        "body": (
            "Stage III means melanoma has reached nearby lymph nodes or nearby skin/in-transit sites. "
            "Treatment plans are individual and must come from your care team."
        ),
    },
    {
        "slug": "stage-iv",
        "title": "Understanding Stage IV",
        "body": (
            "Stage IV means melanoma has spread to distant sites. Only your oncology/dermatology team "
            "can interpret this for your situation."
        ),
    },
    {
        "slug": "review",
        "title": "Why dermatologist review is important",
        "body": (
            "AI can miss cancers and can also flag harmless spots. A dermatologist reviews the image, "
            "your history, and (if needed) pathology before a report is finalized. Do not start, stop, "
            "or change treatment based on an AI score alone."
        ),
    },
]


def topic_by_slug(slug: str) -> dict | None:
    for item in EDUCATION_TOPICS:
        if item["slug"] == slug:
            return item
    return None
