# ======================================================================
# food_agent_ol.py
# 100% OLLAMA AGENTIC CONSTRAINT-AWARE HYBRID-RAG DIETARY CHATBOT
#
# Routes / agents:
#   DIRECT         -> Local Ollama
#   RAG            -> Constraint-aware Hybrid RAG -> Local Ollama
#   PLANNING_AGENT -> Nutrition Planning Agent -> Python/Pandas -> Ollama
#   ANALYSIS_AGENT -> Nutrition Analysis Agent -> Python/Pandas -> Ollama
#   BMI            -> Deterministic Python BMI tool
#   MCP            -> GitHub MCP Server -> Local Ollama
#
# RAG strategies:
#   1. Semantic FAISS retrieval
#   2. Pandas numerical nutrient ranking
#   3. Diversified dietary retrieval
#   4. Deterministic calorie-budget meal planning
# ======================================================================

import os
import re
import json
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import requests

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
top_k = 50

# ======================================================================
# 1. BASIC CONFIGURATION
# ======================================================================

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

CSV_FILE = (
    BASE_DIR /
    "daily_food_nutrition_representative_120.csv"
)

OLLAMA_BASE_URL = "http://localhost:11434"

# ----------------------------------------------------------
# IMPORTANT:
# Change this if `ollama list` shows a different model name.
# Examples:
#   llama3
#   llama3.2
#   gemma3:4b
#   qwen3:4b
# ----------------------------------------------------------

OLLAMA_MODEL = "llama3"

GITHUB_TOKEN = None


# ======================================================================
# 2. ENVIRONMENT CHECK
# ======================================================================

print("\n" + "=" * 78)
print("ENVIRONMENT CHECK")
print("=" * 78)

print(f"Program folder : {BASE_DIR}")
print(f".env file      : {ENV_FILE}")
print(f"CSV file       : {CSV_FILE}")
print(f".env exists    : {ENV_FILE.exists()}")
print(f"CSV exists     : {CSV_FILE.exists()}")
print(f"Ollama URL     : {OLLAMA_BASE_URL}")
print(f"Ollama model   : {OLLAMA_MODEL}")


if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"\nFood CSV not found:\n{CSV_FILE}"
    )


# ======================================================================
# 3. LOAD ENVIRONMENT VARIABLES
# ======================================================================

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if (
    GITHUB_TOKEN
    and GITHUB_TOKEN != "YOUR_REAL_GITHUB_TOKEN"
    and GITHUB_TOKEN != "YOUR_ACTUAL_GITHUB_TOKEN"
):
    print("GitHub token   : FOUND")
else:
    print("GitHub token   : NOT CONFIGURED")


# ======================================================================
# 4. OLLAMA CONNECTION TEST
# ======================================================================

def check_ollama():

    print("\n" + "=" * 78)
    print("CHECKING OLLAMA")
    print("=" * 78)

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        models = [
            item.get("name", "")
            for item in data.get("models", [])
        ]

        print("Ollama server  : RUNNING")

        if models:

            print("Installed models:")

            for model in models:
                print(f" - {model}")

        else:

            print("Installed models: NONE")

        # --------------------------------------------------------------
        # Model matching
        # --------------------------------------------------------------

        model_found = any(
            model == OLLAMA_MODEL
            or model.startswith(OLLAMA_MODEL + ":")
            for model in models
        )

        if not model_found:

            print(
                f"\nWARNING: '{OLLAMA_MODEL}' "
                "was not found in Ollama."
            )

            print(
                f"Run:\n"
                f"    ollama pull {OLLAMA_MODEL}"
            )

        else:

            print(
                f"\nSelected model : {OLLAMA_MODEL}"
            )

        return True

    except requests.exceptions.ConnectionError:

        print("Ollama server  : NOT RUNNING")

        print(
            "\nStart Ollama first."
        )

        print(
            "If necessary run:\n"
            "    ollama serve"
        )

        return False

    except Exception as e:

        print(
            f"Ollama check error: {e}"
        )

        return False


OLLAMA_AVAILABLE = check_ollama()




# ======================================================================
# 6. LOAD FOOD DATABASE
# ======================================================================

print("\n" + "=" * 78)
print("LOADING FOOD DATABASE")
print("=" * 78)

df = pd.read_csv(
    CSV_FILE
).reset_index(drop=True)

print(f"Foods loaded : {len(df)}")
print(f"Columns      : {len(df.columns)}")

print("\nColumns:")

for column in df.columns:
    print(f" - {column}")


# ======================================================================
# 7. CLEAN DATABASE
# ======================================================================

NUMERIC_COLUMNS = [
    "Calories (kcal)",
    "Protein (g)",
    "Carbohydrates (g)",
    "Fat (g)",
    "Fiber (g)",
    "Sugars (g)",
    "Sodium (mg)",
    "Cholesterol (mg)",
    "Water_Intake (ml)"
]

for column in NUMERIC_COLUMNS:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


TEXT_COLUMNS = [
    "Food_Item",
    "Category",
    "Meal_Type"
]

for column in TEXT_COLUMNS:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )


# ======================================================================
# 8. CREATE FOOD CHUNKS
# ======================================================================

print("\n" + "=" * 78)
print("CREATING FOOD CHUNKS")
print("=" * 78)


def row_to_chunk(row):

    parts = []

    for column in df.columns:

        value = row[column]

        if pd.notna(value):

            parts.append(
                f"{column}: {value}"
            )

    return " | ".join(parts)


food_chunks = [
    row_to_chunk(row)
    for _, row in df.iterrows()
]

print(
    f"Created {len(food_chunks)} food chunks."
)


# ======================================================================
# 9. EMBEDDING MODEL
# ======================================================================

print("\n" + "=" * 78)
print("LOADING EMBEDDING MODEL")
print("=" * 78)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Embedding model loaded.")
print("Creating embeddings...")

food_embeddings = embedding_model.encode(
    food_chunks,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

food_embeddings = np.asarray(
    food_embeddings,
    dtype="float32"
)

EMBEDDING_DIMENSION = (
    food_embeddings.shape[1]
)

print(
    f"Embedding matrix: "
    f"{food_embeddings.shape}"
)


# ======================================================================
# 10. FAISS VECTOR DATABASE
# ======================================================================

print("\n" + "=" * 78)
print("CREATING FAISS VECTOR DATABASE")
print("=" * 78)

faiss_index = faiss.IndexFlatIP(
    EMBEDDING_DIMENSION
)

faiss_index.add(
    food_embeddings
)

print(
    f"FAISS contains "
    f"{faiss_index.ntotal} vectors."
)


# ======================================================================
# 11. EXACT TERM MATCHING
# ======================================================================

def contains_term(text, term):

    pattern = (
        r"(?<!\w)"
        + re.escape(term.lower())
        + r"(?!\w)"
    )

    return (
        re.search(
            pattern,
            text.lower()
        )
        is not None
    )


# ======================================================================
# 12. QUERY UNDERSTANDING
# ======================================================================

def detect_dietary_constraint(query):

    q = query.lower()

    if "vegan" in q:
        return "VEGAN"

    if "vegetarian" in q:
        return "VEGETARIAN"

    return None


def detect_meal_type(query):

    q = query.lower()

    for meal in [
        "breakfast",
        "lunch",
        "dinner",
        "snack"
    ]:

        if meal in q:
            return meal

    return None


# ======================================================================
# 13. CALORIE-BUDGET DETECTION
# ======================================================================

def detect_calorie_budget(query):

    q = query.lower()

    patterns = [

        # maximum 3000 calories
        r"(?:maximum|max|under|below|less than|up to|within)"
        r"\s*(?:of\s*)?(\d{3,5})\s*(?:kcal|calories|calorie)",

        # 3000 calorie diet
        r"(\d{3,5})\s*[- ]?(?:kcal|calorie|calories)"
        r"\s*(?:diet|meal plan|plan)",

        # budget of 3000 calories
        r"(?:budget|limit|target)"
        r"\s*(?:of|is|=|:)?\s*(\d{3,5})"
        r"\s*(?:kcal|calories|calorie)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
            re.IGNORECASE
        )

        if match:

            value = int(
                match.group(1)
            )

            # Practical guard against accidental numbers
            if 500 <= value <= 10000:
                return value

    return None


# ======================================================================
# 14. DETECT PLAN REQUEST
# ======================================================================

def is_diet_plan_request(query):

    q = query.lower()

    phrases = [

        "diet plan",
        "meal plan",
        "daily diet",
        "daily meal",
        "prepare a diet",
        "prepare diet",
        "plan my diet",
        "plan my meals",
        "suitable diet",
        "diet for me",
        "what should i eat",
        "what can i eat"
    ]

    return any(
        phrase in q
        for phrase in phrases
    )


# ======================================================================
# 15. NUTRIENT OBJECTIVE DETECTION
# ======================================================================

def detect_nutrient_objective(query):

    q = query.lower()

    rules = [

        (
            [
                "high protein",
                "highest protein",
                "most protein",
                "protein rich",
                "protein-rich"
            ],
            "Protein (g)",
            False,
            "highest protein"
        ),

        (
            [
                "low protein",
                "lowest protein"
            ],
            "Protein (g)",
            True,
            "lowest protein"
        ),

        (
            [
                "low calorie",
                "low calories",
                "lowest calorie",
                "lowest calories"
            ],
            "Calories (kcal)",
            True,
            "lowest calories"
        ),

        (
            [
                "high calorie",
                "high calories",
                "highest calorie",
                "highest calories"
            ],
            "Calories (kcal)",
            False,
            "highest calories"
        ),

        (
            [
                "high fiber",
                "high fibre",
                "highest fiber",
                "highest fibre",
                "fiber rich",
                "fibre rich"
            ],
            "Fiber (g)",
            False,
            "highest fiber"
        ),

        (
            [
                "low fiber",
                "low fibre"
            ],
            "Fiber (g)",
            True,
            "lowest fiber"
        ),

        (
            [
                "low fat",
                "lowest fat"
            ],
            "Fat (g)",
            True,
            "lowest fat"
        ),

        (
            [
                "high fat",
                "highest fat"
            ],
            "Fat (g)",
            False,
            "highest fat"
        ),

        (
            [
                "low sugar",
                "lowest sugar"
            ],
            "Sugars (g)",
            True,
            "lowest sugar"
        ),

        (
            [
                "high sugar",
                "highest sugar"
            ],
            "Sugars (g)",
            False,
            "highest sugar"
        ),

        (
            [
                "low sodium",
                "lowest sodium"
            ],
            "Sodium (mg)",
            True,
            "lowest sodium"
        ),

        (
            [
                "high sodium",
                "highest sodium"
            ],
            "Sodium (mg)",
            False,
            "highest sodium"
        ),

        (
            [
                "low cholesterol",
                "lowest cholesterol"
            ],
            "Cholesterol (mg)",
            True,
            "lowest cholesterol"
        ),

        (
            [
                "high cholesterol",
                "highest cholesterol"
            ],
            "Cholesterol (mg)",
            False,
            "highest cholesterol"
        ),

        (
            [
                "low carb",
                "low carbs",
                "low carbohydrate",
                "low carbohydrates"
            ],
            "Carbohydrates (g)",
            True,
            "lowest carbohydrates"
        ),

        (
            [
                "high carb",
                "high carbs",
                "high carbohydrate",
                "high carbohydrates"
            ],
            "Carbohydrates (g)",
            False,
            "highest carbohydrates"
        )
    ]

    for (
        phrases,
        column,
        ascending,
        description
    ) in rules:

        if any(
            phrase in q
            for phrase in phrases
        ):

            return {
                "column": column,
                "ascending": ascending,
                "description": description
            }

    return None


# ======================================================================
# 16. DIETARY ELIGIBILITY RULES
# ======================================================================

VEGETARIAN_FORBIDDEN = [

    "beef",
    "steak",
    "pork",
    "ham",
    "bacon",
    "chicken",
    "turkey",
    "lamb",
    "mutton",
    "sausage",
    "meat",

    "fish",
    "seafood",
    "salmon",
    "tuna",
    "shrimp",
    "prawn",
    "oyster",
    "calamari",
    "squid",
    "crab",
    "lobster",
    "sardine",
    "anchovy",
    "cod",
    "ceviche"
]


VEGAN_EXTRA_FORBIDDEN = [

    "egg",
    "eggs",
    "milk",
    "cheese",
    "yogurt",
    "yoghurt",
    "cream",
    "butter",
    "dairy",
    "whey",
    "honey"
]


# ======================================================================
# 17. FOOD IDENTITY
# ======================================================================

def food_identity_text(row):

    return (
        str(
            row.get(
                "Food_Item",
                ""
            )
        )
        + " "
        +
        str(
            row.get(
                "Category",
                ""
            )
        )
    ).lower()


def vegetarian_allowed(row):

    text = food_identity_text(row)

    return not any(
        contains_term(
            text,
            term
        )
        for term in VEGETARIAN_FORBIDDEN
    )


def vegan_allowed(row):

    if not vegetarian_allowed(row):
        return False

    text = food_identity_text(row)

    return not any(
        contains_term(
            text,
            term
        )
        for term in VEGAN_EXTRA_FORBIDDEN
    )


# ======================================================================
# 18. DIET FILTER
# ======================================================================

def apply_diet_filter(
    dataframe,
    diet
):

    if diet is None:
        return dataframe.copy()

    if diet == "VEGETARIAN":

        mask = dataframe.apply(
            vegetarian_allowed,
            axis=1
        )

        return dataframe[
            mask
        ].copy()

    if diet == "VEGAN":

        mask = dataframe.apply(
            vegan_allowed,
            axis=1
        )

        return dataframe[
            mask
        ].copy()

    return dataframe.copy()


# ======================================================================
# 19. MEAL FILTER
# ======================================================================

def apply_meal_filter(
    dataframe,
    meal
):

    if meal is None:
        return dataframe.copy()

    if "Meal_Type" not in dataframe.columns:
        return dataframe.copy()

    mask = (
        dataframe["Meal_Type"]
        .astype(str)
        .str.contains(
            meal,
            case=False,
            na=False
        )
    )

    return dataframe[
        mask
    ].copy()


# ======================================================================
# 20. SEMANTIC SEARCH
# ======================================================================

def semantic_search(
    query,
    candidates,
    k=top_k
):

    if candidates.empty:
        return []

    original_indices = (
        candidates.index
        .to_numpy(dtype=int)
    )

    candidate_vectors = (
        food_embeddings[
            original_indices
        ]
    )

    local_index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    local_index.add(
        np.asarray(
            candidate_vectors,
            dtype="float32"
        )
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    n_results = min(
        k,
        len(candidates)
    )

    scores, indices = (
        local_index.search(
            query_embedding,
            n_results
        )
    )

    results = []

    for rank, local_idx in enumerate(
        indices[0],
        start=1
    ):

        if local_idx < 0:
            continue

        original_idx = int(
            original_indices[
                local_idx
            ]
        )

        row = df.loc[
            original_idx
        ]

        results.append(
            {
                "rank": rank,
                "row_index": original_idx,
                "score": float(
                    scores[0][rank - 1]
                ),
                "text": row_to_chunk(row)
            }
        )

    return results


# ======================================================================
# 21. NUMERICAL RETRIEVAL
# ======================================================================

def numerical_search(
    candidates,
    objective,
    k=top_k
):

    if candidates.empty:
        return []

    column = objective[
        "column"
    ]

    if column not in candidates.columns:
        return []

    working = (
        candidates
        .dropna(
            subset=[column]
        )
        .sort_values(
            column,
            ascending=objective[
                "ascending"
            ]
        )
        .head(k)
    )

    results = []

    for rank, (
        index,
        row
    ) in enumerate(
        working.iterrows(),
        start=1
    ):

        results.append(
            {
                "rank": rank,
                "row_index": int(index),
                "score": None,
                "text": row_to_chunk(row)
            }
        )

    return results


# ======================================================================
# 22. DIVERSIFIED SEMANTIC RETRIEVAL
# ======================================================================

def diversified_search(
    query,
    candidates,
    k=8
):

    if candidates.empty:
        return []

    results = []
    used_indices = set()

    for meal in [
        "Breakfast",
        "Lunch",
        "Dinner",
        "Snack"
    ]:

        meal_candidates = (
            apply_meal_filter(
                candidates,
                meal
            )
        )

        if meal_candidates.empty:
            continue

        meal_results = (
            semantic_search(
                query,
                meal_candidates,
                k=2
            )
        )

        for item in meal_results:

            if (
                item["row_index"]
                in used_indices
            ):
                continue

            item[
                "meal_group"
            ] = meal

            results.append(item)

            used_indices.add(
                item["row_index"]
            )

            if len(results) >= k:
                break

        if len(results) >= k:
            break

    # ----------------------------------------------------------
    # Fill remaining positions
    # ----------------------------------------------------------

    if len(results) < k:

        extra_results = (
            semantic_search(
                query,
                candidates,
                k=min(
                    len(candidates),
                    k * 3
                )
            )
        )

        for item in extra_results:

            if (
                item["row_index"]
                in used_indices
            ):
                continue

            results.append(item)

            used_indices.add(
                item["row_index"]
            )

            if len(results) >= k:
                break

    for rank, item in enumerate(
        results,
        start=1
    ):
        item["rank"] = rank

    return results


# ======================================================================
# 23. FINAL DIETARY VALIDATION
# ======================================================================

def final_dietary_validation(
    results,
    diet
):

    if diet is None:
        return results

    validated = []

    for item in results:

        row = df.loc[
            item["row_index"]
        ]

        if diet == "VEGETARIAN":

            allowed = vegetarian_allowed(
                row
            )

        elif diet == "VEGAN":

            allowed = vegan_allowed(
                row
            )

        else:
            allowed = True

        if allowed:

            validated.append(item)

        else:

            print(
                "\n[FINAL FILTER] "
                "Removed incompatible food:"
            )

            print(
                row.get(
                    "Food_Item",
                    "Unknown"
                )
            )

    for rank, item in enumerate(
        validated,
        start=1
    ):
        item["rank"] = rank

    return validated


# ======================================================================
# 24. MEAL-PLANNER HELPER:
#     REMOVE NON-SUBSTANTIAL ITEMS
# ======================================================================

def is_substantial_food(row):

    """
    Prevents a diet plan being made mainly from:
    salt, spices, tiny garnishes, zero-calorie beverages, etc.

    This is only used for PLAN GENERATION.
    Such foods remain searchable through normal RAG.
    """

    calories = row.get(
        "Calories (kcal)",
        np.nan
    )

    protein = row.get(
        "Protein (g)",
        np.nan
    )

    carbs = row.get(
        "Carbohydrates (g)",
        np.nan
    )

    fat = row.get(
        "Fat (g)",
        np.nan
    )

    try:
        calories = float(calories)
    except Exception:
        calories = 0

    try:
        protein = float(protein)
    except Exception:
        protein = 0

    try:
        carbs = float(carbs)
    except Exception:
        carbs = 0

    try:
        fat = float(fat)
    except Exception:
        fat = 0

    # ----------------------------------------------------------
    # Exclude extremely trivial items from being the core of
    # an automatically generated meal plan.
    # ----------------------------------------------------------

    if calories < 20:
        return False

    if (
        protein <= 0
        and carbs <= 1
        and fat <= 0
    ):
        return False

    category = str(
        row.get(
            "Category",
            ""
        )
    ).lower()

    food_name = str(
        row.get(
            "Food_Item",
            ""
        )
    ).lower()

    trivial_terms = [
        "salt",
        "garnish",
        "saffron",
        "club soda"
    ]

    combined = (
        food_name
        + " "
        + category
    )

    if any(
        term in combined
        for term in trivial_terms
    ):
        return False

    return True


# ======================================================================
# 25. MEAL PLAN TARGET DISTRIBUTION
# ======================================================================

MEAL_BUDGET_RATIOS = {

    "Breakfast": 0.25,

    "Lunch": 0.30,

    "Dinner": 0.30,

    "Snack": 0.15
}


# ======================================================================
# 26. SELECT FOODS FOR ONE MEAL
# ======================================================================

def select_foods_for_meal(
    meal_candidates,
    target_calories,
    max_items=3
):

    """
    Deterministic heuristic planner.

    Goals:
    - stay within meal calorie allocation
    - avoid tiny condiment-only items
    - prefer nutritionally useful foods
    - select multiple foods where possible
    """

    if meal_candidates.empty:
        return []

    working = (
        meal_candidates
        .copy()
    )

    working = working[
        working.apply(
            is_substantial_food,
            axis=1
        )
    ].copy()

    if working.empty:
        return []

    # ----------------------------------------------------------
    # Nutrient utility score
    #
    # This is NOT a medical health score.
    # It is only a simple deterministic selection heuristic.
    # ----------------------------------------------------------

    working[
        "_utility"
    ] = (

        working[
            "Protein (g)"
        ].fillna(0) * 2.0

        +

        working[
            "Fiber (g)"
        ].fillna(0) * 1.5

        -

        working[
            "Sugars (g)"
        ].fillna(0) * 0.15
    )

    # ----------------------------------------------------------
    # Prefer foods that fit comfortably inside meal budget
    # ----------------------------------------------------------

    working = working[
        working[
            "Calories (kcal)"
        ].fillna(0) <= target_calories
    ].copy()

    if working.empty:
        return []

    # Sort by utility first
    working = working.sort_values(
        [
            "_utility",
            "Protein (g)",
            "Fiber (g)"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )

    selected = []

    total_calories = 0.0

    used_categories = set()

    # ----------------------------------------------------------
    # First pass:
    # encourage category diversity
    # ----------------------------------------------------------

    for index, row in working.iterrows():

        calories = float(
            row.get(
                "Calories (kcal)",
                0
            )
        )

        category = str(
            row.get(
                "Category",
                ""
            )
        )

        if category in used_categories:
            continue

        if (
            total_calories
            + calories
            <= target_calories
        ):

            selected.append(
                int(index)
            )

            total_calories += calories

            used_categories.add(
                category
            )

        if len(selected) >= max_items:
            break

    # ----------------------------------------------------------
    # Second pass:
    # fill remaining slots
    # ----------------------------------------------------------

    if len(selected) < max_items:

        for index, row in working.iterrows():

            if int(index) in selected:
                continue

            calories = float(
                row.get(
                    "Calories (kcal)",
                    0
                )
            )

            if (
                total_calories
                + calories
                <= target_calories
            ):

                selected.append(
                    int(index)
                )

                total_calories += calories

            if len(selected) >= max_items:
                break

    return selected


# ======================================================================
# 27. DETERMINISTIC CALORIE-BUDGET MEAL PLANNER
# ======================================================================

def create_calorie_budget_plan(
    candidates,
    calorie_budget
):

    """
    Builds a deterministic daily meal plan from the local CSV.

    Important:
    - Python chooses foods.
    - Python calculates totals.
    - Ollama does NOT invent the plan.
    """

    if candidates.empty:

        return {
            "success": False,
            "message": (
                "No eligible foods are available "
                "for meal planning."
            )
        }

    selected_indices = []
    meal_plan = {}

    # ----------------------------------------------------------
    # Allocate calories across meals
    # ----------------------------------------------------------

    for meal, ratio in (
        MEAL_BUDGET_RATIOS.items()
    ):

        target = (
            calorie_budget
            * ratio
        )

        meal_candidates = (
            apply_meal_filter(
                candidates,
                meal
            )
        )

        # Avoid selecting same food twice
        meal_candidates = (
            meal_candidates[
                ~meal_candidates.index.isin(
                    selected_indices
                )
            ]
        )

        chosen = (
            select_foods_for_meal(
                meal_candidates,
                target_calories=target,
                max_items=3
            )
        )

        selected_indices.extend(
            chosen
        )

        meal_plan[
            meal
        ] = chosen

    # ----------------------------------------------------------
    # Build selected dataframe
    # ----------------------------------------------------------

    if not selected_indices:

        return {
            "success": False,
            "message": (
                "The database did not contain sufficient "
                "substantial foods for the requested plan."
            )
        }

    selected_df = df.loc[
        selected_indices
    ].copy()

    # ----------------------------------------------------------
    # Calculate totals deterministically
    # ----------------------------------------------------------

    totals = {}

    for column in NUMERIC_COLUMNS:

        if column in selected_df.columns:

            totals[column] = float(
                selected_df[
                    column
                ]
                .fillna(0)
                .sum()
            )

    total_calories = totals.get(
        "Calories (kcal)",
        0.0
    )

    # ----------------------------------------------------------
    # Hard calorie validation
    # ----------------------------------------------------------

    if total_calories > calorie_budget:

        raise RuntimeError(
            "Meal planner exceeded calorie budget. "
            "This should not occur."
        )

    # ----------------------------------------------------------
    # Create structured plan
    # ----------------------------------------------------------

    structured_plan = {}

    for meal, indices in (
        meal_plan.items()
    ):

        structured_plan[
            meal
        ] = []

        for index in indices:

            row = df.loc[
                index
            ]

            structured_plan[
                meal
            ].append(
                {
                    "row_index":
                        int(index),

                    "food":
                        str(
                            row.get(
                                "Food_Item",
                                ""
                            )
                        ),

                    "category":
                        str(
                            row.get(
                                "Category",
                                ""
                            )
                        ),

                    "calories":
                        float(
                            row.get(
                                "Calories (kcal)",
                                0
                            )
                        ),

                    "protein":
                        float(
                            row.get(
                                "Protein (g)",
                                0
                            )
                        ),

                    "carbohydrates":
                        float(
                            row.get(
                                "Carbohydrates (g)",
                                0
                            )
                        ),

                    "fat":
                        float(
                            row.get(
                                "Fat (g)",
                                0
                            )
                        ),

                    "fiber":
                        float(
                            row.get(
                                "Fiber (g)",
                                0
                            )
                        )
                }
            )

    return {

        "success":
            True,

        "calorie_budget":
            calorie_budget,

        "total_calories":
            total_calories,

        "remaining_calories":
            calorie_budget
            - total_calories,

        "totals":
            totals,

        "plan":
            structured_plan,

        "selected_indices":
            selected_indices
    }


# ======================================================================
# 28. DISPLAY MEAL PLAN
# ======================================================================

def print_meal_plan(plan):

    print(
        "\n" + "-" * 78
    )

    print(
        "DETERMINISTIC CALORIE-BUDGET MEAL PLAN"
    )

    print(
        "-" * 78
    )

    for meal, foods in (
        plan["plan"].items()
    ):

        print(
            f"\n{meal.upper()}"
        )

        if not foods:

            print(
                "  No suitable food selected."
            )

            continue

        meal_total = 0

        for item in foods:

            meal_total += (
                item["calories"]
            )

            print(
                f"  - {item['food']}"
                f" | {item['calories']:.0f} kcal"
                f" | Protein {item['protein']:.1f} g"
                f" | Fiber {item['fiber']:.1f} g"
            )

        print(
            f"  Meal calories: "
            f"{meal_total:.0f} kcal"
        )

    print(
        "\nTOTAL CALORIES : "
        f"{plan['total_calories']:.0f} kcal"
    )

    print(
        "CALORIE BUDGET : "
        f"{plan['calorie_budget']:.0f} kcal"
    )

    print(
        "REMAINING      : "
        f"{plan['remaining_calories']:.0f} kcal"
    )

    print(
        "-" * 78
    )


# ======================================================================
# 29. FORMAT MEAL PLAN FOR OLLAMA
# ======================================================================

def meal_plan_to_context(plan):

    lines = []

    for meal, foods in (
        plan["plan"].items()
    ):

        lines.append(
            f"\n{meal}:"
        )

        for item in foods:

            lines.append(

                f"- {item['food']} | "
                f"{item['calories']:.0f} kcal | "
                f"Protein: {item['protein']:.1f} g | "
                f"Carbohydrates: "
                f"{item['carbohydrates']:.1f} g | "
                f"Fat: {item['fat']:.1f} g | "
                f"Fiber: {item['fiber']:.1f} g"
            )

    lines.append(
        f"\nTotal calories: "
        f"{plan['total_calories']:.0f} kcal"
    )

    lines.append(
        f"Maximum calorie budget: "
        f"{plan['calorie_budget']:.0f} kcal"
    )

    lines.append(
        f"Remaining calories: "
        f"{plan['remaining_calories']:.0f} kcal"
    )

    return "\n".join(
        lines
    )


# ======================================================================
# 30. CALORIE-BUDGET RAG ANSWER
# ======================================================================

def calorie_plan_answer(
    question,
    diet,
    calorie_budget
):

    print(
        "\n[MEAL PLANNER] "
        "Calorie-budget planning selected."
    )

    print(
        f"[MEAL PLANNER] "
        f"Maximum calories: {calorie_budget}"
    )

    print(
        "[MEAL PLANNER] "
        f"Dietary constraint: "
        f"{diet or 'NONE'}"
    )

    candidates = df.copy()

    candidates = (
        apply_diet_filter(
            candidates,
            diet
        )
    )

    print(
        "[MEAL PLANNER] "
        f"Eligible foods: {len(candidates)}"
    )

    plan = (
        create_calorie_budget_plan(
            candidates,
            calorie_budget
        )
    )

    if not plan[
        "success"
    ]:

        return plan[
            "message"
        ]

    # ----------------------------------------------------------
    # Final dietary validation of selected foods
    # ----------------------------------------------------------

    if diet:

        for index in (
            plan[
                "selected_indices"
            ]
        ):

            row = df.loc[
                index
            ]

            if (
                diet == "VEGETARIAN"
                and
                not vegetarian_allowed(
                    row
                )
            ):

                raise RuntimeError(
                    "Vegetarian validation failed."
                )

            if (
                diet == "VEGAN"
                and
                not vegan_allowed(
                    row
                )
            ):

                raise RuntimeError(
                    "Vegan validation failed."
                )

    print_meal_plan(
        plan
    )

    context = (
        meal_plan_to_context(
            plan
        )
    )

    prompt = f"""
The deterministic Python meal-planning tool has created
and mathematically validated the following meal plan.

USER REQUEST:
{question}

DIETARY CONSTRAINT:
{diet or 'None'}

MAXIMUM CALORIE BUDGET:
{calorie_budget} kcal

VERIFIED PLAN:
{context}

INSTRUCTIONS:

1. Explain the verified plan clearly.
2. Do not add foods that are not present in the verified plan.
3. Do not change the nutritional values.
4. Do not recalculate or modify the calorie total.
5. Respect the dietary constraint exactly.
6. State that the total is within the requested calorie maximum.
7. Do not claim that this is a medically personalized diet.
8. Mention that the plan is limited by the foods available
   in the local 120-food database.
9. Keep the answer concise and practical.

Return the final user-facing response.
"""

    print(
        "\n[OLLAMA] Explaining "
        "verified meal plan..."
    )

    return ask_ollama(
        prompt,
        system_prompt=(
            "You are a dietary information assistant. "
            "You must strictly follow the verified food "
            "records supplied by the deterministic Python tool."
        ),
        temperature=0.1
    )


# ======================================================================
# 31. COMPLETE HYBRID RETRIEVAL
# ======================================================================

def retrieve_foods(
    query,
    k=top_k
):

    diet = (
        detect_dietary_constraint(
            query
        )
    )

    meal = (
        detect_meal_type(
            query
        )
    )

    objective = (
        detect_nutrient_objective(
            query
        )
    )

    candidates = df.copy()

    initial_count = len(
        candidates
    )

    # ----------------------------------------------------------
    # Hard dietary filter
    # ----------------------------------------------------------

    candidates = (
        apply_diet_filter(
            candidates,
            diet
        )
    )

    after_diet = len(
        candidates
    )

    # ----------------------------------------------------------
    # Meal filter
    # ----------------------------------------------------------

    candidates = (
        apply_meal_filter(
            candidates,
            meal
        )
    )

    after_meal = len(
        candidates
    )

    # ----------------------------------------------------------
    # Numerical retrieval
    # ----------------------------------------------------------

    if objective is not None:

        results = numerical_search(
            candidates,
            objective,
            k=k
        )

        method = (
            "CONSTRAINT-AWARE "
            "STRUCTURED RETRIEVAL"
        )

        logic = (
            "dietary filtering + "
            "meal filtering + "
            "numerical nutrient ranking"
        )

    # ----------------------------------------------------------
    # Broad diet retrieval
    # ----------------------------------------------------------

    elif is_diet_plan_request(
        query
    ):

        results = (
            diversified_search(
                query,
                candidates,
                k=8
            )
        )

        method = (
            "CONSTRAINT-AWARE "
            "DIVERSIFIED RAG"
        )

        logic = (
            "dietary filtering + "
            "meal diversity + "
            "semantic similarity"
        )

    # ----------------------------------------------------------
    # Semantic retrieval
    # ----------------------------------------------------------

    else:

        results = semantic_search(
            query,
            candidates,
            k=k
        )

        method = (
            "CONSTRAINT-AWARE "
            "SEMANTIC FAISS"
        )

        logic = (
            "dietary filtering + "
            "meal filtering + "
            "semantic similarity"
        )

    # ----------------------------------------------------------
    # Second validation
    # ----------------------------------------------------------

    results = (
        final_dietary_validation(
            results,
            diet
        )
    )

    return {

        "diet":
            diet,

        "meal":
            meal,

        "objective":
            objective,

        "method":
            method,

        "logic":
            logic,

        "initial_count":
            initial_count,

        "after_diet":
            after_diet,

        "after_meal":
            after_meal,

        "results":
            results
    }


# ======================================================================
# 32. RAG ANSWER
# ======================================================================

def rag_answer(question):

    diet = (
        detect_dietary_constraint(
            question
        )
    )

    calorie_budget = (
        detect_calorie_budget(
            question
        )
    )

    plan_request = (
        is_diet_plan_request(
            question
        )
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    # Diet plan + explicit calorie budget
    # goes to deterministic planner BEFORE normal low-calorie
    # ranking.
    # ----------------------------------------------------------

    if (
        plan_request
        and
        calorie_budget is not None
    ):

        return calorie_plan_answer(
            question,
            diet,
            calorie_budget
        )

    # ----------------------------------------------------------
    # Standard Hybrid RAG
    # ----------------------------------------------------------

    print(
        "\n[RAG] Searching local "
        "food knowledge base..."
    )

    retrieval = (
        retrieve_foods(
            question,
            k=top_k
        )
    )

    print(
        f"[RAG] Retrieval method : "
        f"{retrieval['method']}"
    )

    print(
        f"[RAG] Retrieval logic  : "
        f"{retrieval['logic']}"
    )

    print(
        "[RAG] Dietary constraint:",
        retrieval["diet"]
        or "NONE"
    )

    print(
        "[RAG] Meal constraint   :",
        retrieval["meal"]
        or "NONE"
    )

    if retrieval[
        "objective"
    ]:

        print(
            "[RAG] Nutrient objective:",
            retrieval[
                "objective"
            ][
                "description"
            ]
        )

    print(
        "\n[RAG] Candidate filtering:"
    )

    print(
        "      Original database :",
        retrieval[
            "initial_count"
        ]
    )

    print(
        "      After diet filter :",
        retrieval[
            "after_diet"
        ]
    )

    print(
        "      After meal filter :",
        retrieval[
            "after_meal"
        ]
    )

    results = retrieval[
        "results"
    ]

    print(
        f"\n[RAG] Final retrieved "
        f"records: {len(results)}"
    )

    if not results:

        return (
            "No eligible food records "
            "were found in the local "
            "food database."
        )

    # ----------------------------------------------------------
    # Display verified evidence
    # ----------------------------------------------------------

    print(
        "\n" + "-" * 78
    )

    print(
        "FINAL RAG CONTEXT "
        "(AFTER CONSTRAINT VALIDATION)"
    )

    print(
        "-" * 78
    )

    for item in results:

        print(
            f"\nRank {item['rank']}"
        )

        if item.get(
            "meal_group"
        ):

            print(
                "Meal group:",
                item[
                    "meal_group"
                ]
            )

        if (
            item["score"]
            is not None
        ):

            print(
                "Similarity score: "
                f"{item['score']:.4f}"
            )

        print(
            item["text"]
        )

    print(
        "\n" + "-" * 78
    )

    context = "\n\n".join(
        item["text"]
        for item in results
    )

    objective_text = (

        retrieval[
            "objective"
        ][
            "description"
        ]

        if retrieval[
            "objective"
        ]

        else "None"
    )

    prompt_context = f"""
                You are given verified records retrieved from a local
                food database.

                USER QUESTION:
                {question}

                DIETARY CONSTRAINT:
                {retrieval['diet'] or 'None'}

                MEAL CONSTRAINT:
                {retrieval['meal'] or 'None'}

                NUTRITIONAL OBJECTIVE:
                {objective_text}

                RETRIEVAL METHOD:
                {retrieval['method']}

                VERIFIED FOOD RECORDS:

                {context}

                INSTRUCTIONS:

                1. Base database-specific recommendations only on the
                verified food records above.

                2. Respect the detected dietary constraint exactly.

                3. Never introduce meat or seafood into a vegetarian answer.

                4. Never introduce animal-derived foods into a vegan answer.

                5. Do not invent nutritional values.

                6. Use the numerical values from the records.

                7. Explain the retrieved results clearly.

                8. If these records are insufficient for a complete diet,
                state that limitation.

                9. Do not claim that the response is a medically personalized
                diet.

                10. Keep the answer practical and concise.
                """

    return prompt_context
# ======================================================================
# 33. BMI LOCAL TOOL
# ======================================================================

def calculate_bmi(
    weight_kg,
    height_m
):

    print(
        "\n[LOCAL TOOL] "
        "calculate_bmi() called."
    )

    if height_m <= 0:

        return {
            "success": False,
            "error": (
                "Height must be greater than zero."
            )
        }

    bmi = (
        weight_kg /
        (height_m ** 2)
    )

    if bmi < 18.5:
        category = "Underweight"

    elif bmi < 25:
        category = "Normal weight"

    elif bmi < 30:
        category = "Overweight"

    else:
        category = "Obesity"

    return {

        "success": True,

        "weight_kg":
            weight_kg,

        "height_m":
            height_m,

        "bmi":
            round(
                bmi,
                2
            ),

        "category":
            category
    }


# ======================================================================
# 34. EXTRACT BMI PARAMETERS
# ======================================================================

def extract_bmi_parameters(question):

    q = question.lower()

    # ----------------------------------------------------------
    # Weight
    # ----------------------------------------------------------

    weight_match = re.search(
        r"(\d+(?:\.\d+)?)\s*kg",
        q
    )

    # ----------------------------------------------------------
    # Height in metres
    # ----------------------------------------------------------

    height_match = re.search(
        r"(\d+(?:\.\d+)?)\s*m(?:eter|eters|etre|etres)?\b",
        q
    )

    if (
        weight_match
        and
        height_match
    ):

        return (
            float(
                weight_match.group(1)
            ),
            float(
                height_match.group(1)
            )
        )

    return None, None


# ======================================================================
# 35. BMI ANSWER
# ======================================================================

def bmi_answer(question):

    print(
        "\n[LOCAL TOOL] "
        "BMI calculator selected."
    )

    weight, height = (
        extract_bmi_parameters(
            question
        )
    )

    if (
        weight is None
        or
        height is None
    ):

        return (
            "Please provide both weight in kilograms "
            "and height in metres, for example: "
            "'Calculate my BMI for 80 kg and 1.8 m.'"
        )

    result = calculate_bmi(
        weight,
        height
    )

    if not result[
        "success"
    ]:

        return result[
            "error"
        ]

    # ----------------------------------------------------------
    # Python calculated BMI.
    # Ollama only explains the deterministic result.
    # ----------------------------------------------------------

    prompt = f"""
A deterministic Python BMI calculator returned:

Weight: {result['weight_kg']} kg
Height: {result['height_m']} m
BMI: {result['bmi']}
Standard BMI category: {result['category']}

Explain this result briefly.

Do not recalculate or change the numerical result.
Mention that BMI is a general screening measure and does not
by itself provide a complete assessment of health.
"""

    print(
        "[LOCAL TOOL] BMI calculated."
    )

    print(
        "[OLLAMA] Explaining "
        "tool result..."
    )

    return ask_ollama(
        prompt,
        temperature=0.1
    )


# ======================================================================
# 36. GITHUB MCP
# ======================================================================

async def github_mcp_search(
    user_question
):

    print(
        "\n[MCP] Preparing "
        "GitHub MCP server..."
    )

    if (
        not GITHUB_TOKEN
        or
        GITHUB_TOKEN
        in [
            "YOUR_REAL_GITHUB_TOKEN",
            "YOUR_ACTUAL_GITHUB_TOKEN"
        ]
    ):

        return (
            "GitHub MCP is not configured. "
            "Add a valid GITHUB_TOKEN "
            "to your .env file."
        )

    server_parameters = (
        StdioServerParameters(

            command="docker",

            args=[
                "run",
                "-i",
                "--rm",

                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",

                "-e",
                "GITHUB_READ_ONLY",

                "-e",
                "GITHUB_TOOLSETS",

                "ghcr.io/github/"
                "github-mcp-server"
            ],

            env={
                **os.environ,

                "GITHUB_PERSONAL_ACCESS_TOKEN":
                    GITHUB_TOKEN,

                "GITHUB_READ_ONLY":
                    "1",

                "GITHUB_TOOLSETS":
                    "repos"
            }
        )
    )

    try:

        async with stdio_client(
            server_parameters
        ) as (
            read_stream,
            write_stream
        ):

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                print(
                    "[MCP] Initializing session..."
                )

                await session.initialize()

                print(
                    "[MCP] Discovering tools..."
                )

                response = (
                    await session.list_tools()
                )

                tools = response.tools

                print(
                    f"[MCP] Discovered "
                    f"{len(tools)} tools."
                )

                for tool in tools:

                    print(
                        f"  - {tool.name}"
                    )

                # ------------------------------------------------------
                # Find repository search tool
                # ------------------------------------------------------

                search_tool = None

                preferred_names = [
                    "search_repositories",
                    "searchRepositories",
                    "search_repos"
                ]

                for tool in tools:

                    if (
                        tool.name
                        in preferred_names
                    ):

                        search_tool = tool
                        break

                if search_tool is None:

                    for tool in tools:

                        name = (
                            tool.name.lower()
                        )

                        if (
                            "search" in name
                            and
                            (
                                "repo" in name
                                or
                                "repository"
                                in name
                            )
                        ):

                            search_tool = tool
                            break

                if search_tool is None:

                    available = ", ".join(
                        tool.name
                        for tool in tools
                    )

                    return (
                        "Repository-search MCP "
                        "tool was not found.\n\n"
                        "Available tools:\n"
                        f"{available}"
                    )

                print(
                    "[MCP] Selected tool: "
                    f"{search_tool.name}"
                )

                arguments = {
                    "query":
                        user_question
                }

                # ------------------------------------------------------
                # Check tool schema for page-size argument
                # ------------------------------------------------------

                try:

                    schema = (
                        search_tool.inputSchema
                    )

                    properties = (

                        schema.get(
                            "properties",
                            {}
                        )

                        if isinstance(
                            schema,
                            dict
                        )

                        else {}
                    )

                    if (
                        "perPage"
                        in properties
                    ):

                        arguments[
                            "perPage"
                        ] = 5

                    elif (
                        "per_page"
                        in properties
                    ):

                        arguments[
                            "per_page"
                        ] = 5

                except Exception:

                    pass

                print(
                    "[MCP] Calling GitHub "
                    "repository-search tool..."
                )

                result = (
                    await session.call_tool(
                        search_tool.name,
                        arguments=arguments
                    )
                )

                # ------------------------------------------------------
                # Convert MCP content to text
                # ------------------------------------------------------

                raw_parts = []

                for part in result.content:

                    if hasattr(
                        part,
                        "text"
                    ):

                        raw_parts.append(
                            part.text
                        )

                    else:

                        raw_parts.append(
                            str(part)
                        )

                raw_result = "\n".join(
                    raw_parts
                )

                print(
                    "[MCP] GitHub data received."
                )

                # ------------------------------------------------------
                # Ollama explains MCP output
                # ------------------------------------------------------

                prompt = f"""
A real GitHub MCP server returned the following repository
search information.

USER REQUEST:
{user_question}

MCP TOOL:
{search_tool.name}

MCP OUTPUT:
{raw_result}

INSTRUCTIONS:

1. Summarize the most relevant repository results.
2. Do not invent repository names.
3. Do not invent stars, descriptions, URLs or metadata.
4. Use only the MCP output supplied above.
5. Mention that the repository information was retrieved
   through the GitHub MCP server.
6. Keep the response concise.
"""

                print(
                    "[MCP] Sending MCP "
                    "results to local Ollama..."
                )

                return ask_ollama(
                    prompt,
                    system_prompt=(
                        "You summarize verified external "
                        "tool results without inventing data."
                    ),
                    temperature=0.1
                )

    except FileNotFoundError:

        return (
            "Docker could not be found. "
            "Start Docker Desktop and verify "
            "'docker --version' works."
        )

    except Exception as e:

        return (
            "GitHub MCP error:\n"
            f"{e}"
        )



# ======================================================================
# 37. MULTI-AGENT EXTENSION
#     - Nutrition Planning Agent
#     - Nutrition Analysis Agent
# ======================================================================

# Stores the most recently generated deterministic meal plan so that
# the Analysis Agent can inspect it in the same CLI session.
LAST_PLAN = None


# ----------------------------------------------------------------------
# FOOD-PREFERENCE UNDERSTANDING
# ----------------------------------------------------------------------

SEAFOOD_TERMS = [
    "seafood", "sea food", "fish", "salmon", "tuna", "shrimp",
    "prawn", "oyster", "calamari", "squid", "crab", "lobster",
    "sardine", "anchovy", "cod", "ceviche"
]

FOOD_PREFERENCE_RULES = {
    "SEAFOOD": SEAFOOD_TERMS,
    "CHICKEN": ["chicken"],
    "MEAT": [
        "beef", "steak", "pork", "ham", "bacon", "chicken",
        "turkey", "lamb", "mutton", "sausage", "meat"
    ],
    "FRUIT": ["fruit", "fruits"],
    "VEGETABLE": ["vegetable", "vegetables", "veggie", "veggies"],
    "EGG": ["egg", "eggs"],
    "DAIRY": ["dairy", "milk", "cheese", "yogurt", "yoghurt"]
}


def detect_food_preference(query):
    q = query.lower()

    # Seafood gets priority because "fish" and seafood names are specific.
    if any(contains_term(q, term) for term in SEAFOOD_TERMS):
        return "SEAFOOD"

    for preference, terms in FOOD_PREFERENCE_RULES.items():
        if preference == "SEAFOOD":
            continue
        if any(contains_term(q, term) for term in terms):
            return preference

    return None


def seafood_allowed(row):
    text = food_identity_text(row)
    return any(contains_term(text, term) for term in SEAFOOD_TERMS)


def food_preference_allowed(row, preference):
    if preference is None:
        return True

    text = food_identity_text(row)

    if preference == "SEAFOOD":
        return seafood_allowed(row)

    if preference == "CHICKEN":
        return contains_term(text, "chicken")

    if preference == "MEAT":
        return any(
            contains_term(text, term)
            for term in FOOD_PREFERENCE_RULES["MEAT"]
        )

    if preference == "FRUIT":
        return (
            contains_term(text, "fruit")
            or "fruit" in str(row.get("Category", "")).lower()
        )

    if preference == "VEGETABLE":
        return (
            contains_term(text, "vegetable")
            or "vegetable" in str(row.get("Category", "")).lower()
        )

    if preference == "EGG":
        return any(
            contains_term(text, term)
            for term in FOOD_PREFERENCE_RULES["EGG"]
        )

    if preference == "DAIRY":
        return any(
            contains_term(text, term)
            for term in FOOD_PREFERENCE_RULES["DAIRY"]
        )

    return True


def apply_food_preference_filter(dataframe, preference):
    if preference is None:
        return dataframe.copy()

    mask = dataframe.apply(
        lambda row: food_preference_allowed(row, preference),
        axis=1
    )
    return dataframe[mask].copy()


def validate_selected_indices(indices, diet=None, preference=None):
    problems = []

    for index in indices:
        row = df.loc[index]
        name = str(row.get("Food_Item", "Unknown"))

        if diet == "VEGETARIAN" and not vegetarian_allowed(row):
            problems.append(f"{name}: violates VEGETARIAN constraint")

        if diet == "VEGAN" and not vegan_allowed(row):
            problems.append(f"{name}: violates VEGAN constraint")

        if preference and not food_preference_allowed(row, preference):
            problems.append(f"{name}: violates {preference} preference")

    return problems


# ----------------------------------------------------------------------
# GENERIC DETERMINISTIC PLAN (NO CALORIE LIMIT REQUIRED)
# ----------------------------------------------------------------------

def select_foods_without_budget(meal_candidates, max_items=2):
    if meal_candidates.empty:
        return []

    working = meal_candidates.copy()
    working = working[
        working.apply(is_substantial_food, axis=1)
    ].copy()

    if working.empty:
        return []

    working["_utility"] = (
        working["Protein (g)"].fillna(0) * 2.0
        + working["Fiber (g)"].fillna(0) * 1.5
        - working["Sugars (g)"].fillna(0) * 0.15
    )

    working = working.sort_values(
        ["_utility", "Protein (g)", "Fiber (g)"],
        ascending=[False, False, False]
    )

    selected = []
    used_categories = set()

    # First pass encourages category diversity.
    for index, row in working.iterrows():
        category = str(row.get("Category", ""))
        if category in used_categories:
            continue
        selected.append(int(index))
        used_categories.add(category)
        if len(selected) >= max_items:
            return selected

    # Second pass fills any remaining slot.
    for index, _ in working.iterrows():
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= max_items:
            break

    return selected


def create_general_meal_plan(candidates, requested_meal=None):
    if candidates.empty:
        return {
            "success": False,
            "message": "No eligible foods are available for the requested plan."
        }

    meals = (
        [requested_meal.capitalize()]
        if requested_meal
        else ["Breakfast", "Lunch", "Dinner", "Snack"]
    )

    meal_plan = {}
    selected_indices = []

    for meal in meals:
        meal_candidates = apply_meal_filter(candidates, meal)
        meal_candidates = meal_candidates[
            ~meal_candidates.index.isin(selected_indices)
        ]

        chosen = select_foods_without_budget(
            meal_candidates,
            max_items=2
        )

        selected_indices.extend(chosen)
        meal_plan[meal] = chosen

    if not selected_indices:
        return {
            "success": False,
            "message": (
                "The local database does not contain substantial eligible "
                "foods for the requested meal-plan constraints."
            )
        }

    selected_df = df.loc[selected_indices].copy()

    totals = {}
    for column in NUMERIC_COLUMNS:
        if column in selected_df.columns:
            totals[column] = float(
                selected_df[column].fillna(0).sum()
            )

    structured_plan = {}

    for meal, indices in meal_plan.items():
        structured_plan[meal] = []

        for index in indices:
            row = df.loc[index]
            structured_plan[meal].append({
                "row_index": int(index),
                "food": str(row.get("Food_Item", "")),
                "category": str(row.get("Category", "")),
                "calories": float(row.get("Calories (kcal)", 0) or 0),
                "protein": float(row.get("Protein (g)", 0) or 0),
                "carbohydrates": float(
                    row.get("Carbohydrates (g)", 0) or 0
                ),
                "fat": float(row.get("Fat (g)", 0) or 0),
                "fiber": float(row.get("Fiber (g)", 0) or 0),
                "sugars": float(row.get("Sugars (g)", 0) or 0),
                "sodium": float(row.get("Sodium (mg)", 0) or 0),
                "cholesterol": float(
                    row.get("Cholesterol (mg)", 0) or 0
                )
            })

    return {
        "success": True,
        "calorie_budget": None,
        "total_calories": totals.get("Calories (kcal)", 0.0),
        "remaining_calories": None,
        "totals": totals,
        "plan": structured_plan,
        "selected_indices": selected_indices
    }


def multiagent_plan_to_context(plan):
    lines = []

    for meal, foods in plan["plan"].items():
        lines.append(f"\n{meal}:")

        if not foods:
            lines.append(
                "- No eligible local-database food was available "
                "for this meal."
            )
            continue

        for item in foods:
            lines.append(
                f"- {item['food']} | {item['calories']:.0f} kcal | "
                f"Protein {item['protein']:.1f} g | "
                f"Carbohydrates {item['carbohydrates']:.1f} g | "
                f"Fat {item['fat']:.1f} g | "
                f"Fiber {item['fiber']:.1f} g"
            )

    lines.append(
        f"\nVerified total calories: "
        f"{plan['total_calories']:.0f} kcal"
    )

    if plan.get("calorie_budget") is not None:
        lines.append(
            f"Maximum calorie budget: "
            f"{plan['calorie_budget']:.0f} kcal"
        )
        lines.append(
            f"Remaining calories: "
            f"{plan['remaining_calories']:.0f} kcal"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# AGENT 1: NUTRITION PLANNING AGENT
# ----------------------------------------------------------------------

def nutrition_planning_agent(question):
    global LAST_PLAN

    print("\n" + "=" * 78)
    print("NUTRITION PLANNING AGENT")
    print("=" * 78)

    diet = detect_dietary_constraint(question)
    preference = detect_food_preference(question)
    meal = detect_meal_type(question)
    calorie_budget = detect_calorie_budget(question)

    print(f"[PLANNING AGENT] Diet constraint : {diet or 'NONE'}")
    print(f"[PLANNING AGENT] Food preference : {preference or 'NONE'}")
    print(f"[PLANNING AGENT] Meal constraint : {meal or 'NONE'}")
    print(
        "[PLANNING AGENT] Calorie budget  : "
        + (
            f"{calorie_budget} kcal"
            if calorie_budget is not None
            else "NONE"
        )
    )

    candidates = df.copy()
    original_count = len(candidates)

    candidates = apply_diet_filter(candidates, diet)
    after_diet = len(candidates)

    candidates = apply_food_preference_filter(
        candidates,
        preference
    )
    after_preference = len(candidates)

    # If a specific meal was requested, planning is restricted to it.
    if meal:
        candidates = apply_meal_filter(candidates, meal)

    after_meal = len(candidates)

    print(f"[PLANNING AGENT] Original foods   : {original_count}")
    print(f"[PLANNING AGENT] After diet      : {after_diet}")
    print(f"[PLANNING AGENT] After preference: {after_preference}")
    print(f"[PLANNING AGENT] After meal      : {after_meal}")

    if candidates.empty:
        return (
            "The local 120-food database has no eligible records "
            "for all of the requested constraints. I did not fill the "
            "plan with unrelated foods."
        )

    if calorie_budget is not None:
        # Existing deterministic calorie planner is reused.
        plan = create_calorie_budget_plan(
            candidates,
            calorie_budget
        )

        # The original planner creates all four meals. If the user asked
        # for one meal, use the general single-meal planner instead so
        # unrelated meal groups are not introduced.
        if meal:
            plan = create_general_meal_plan(
                candidates,
                requested_meal=meal
            )

            # Enforce the user's calorie maximum on the selected meal.
            if (
                plan.get("success")
                and plan["total_calories"] > calorie_budget
            ):
                # Use the existing budget-aware selector for this meal.
                chosen = select_foods_for_meal(
                    candidates,
                    target_calories=calorie_budget,
                    max_items=3
                )

                if not chosen:
                    return (
                        "No substantial eligible foods in the local "
                        "database could satisfy the requested meal and "
                        "calorie constraint."
                    )

                selected_df = df.loc[chosen]
                totals = {
                    col: float(selected_df[col].fillna(0).sum())
                    for col in NUMERIC_COLUMNS
                    if col in selected_df.columns
                }

                foods = []
                for index in chosen:
                    row = df.loc[index]
                    foods.append({
                        "row_index": int(index),
                        "food": str(row.get("Food_Item", "")),
                        "category": str(row.get("Category", "")),
                        "calories": float(
                            row.get("Calories (kcal)", 0) or 0
                        ),
                        "protein": float(row.get("Protein (g)", 0) or 0),
                        "carbohydrates": float(
                            row.get("Carbohydrates (g)", 0) or 0
                        ),
                        "fat": float(row.get("Fat (g)", 0) or 0),
                        "fiber": float(row.get("Fiber (g)", 0) or 0),
                        "sugars": float(row.get("Sugars (g)", 0) or 0),
                        "sodium": float(row.get("Sodium (mg)", 0) or 0),
                        "cholesterol": float(
                            row.get("Cholesterol (mg)", 0) or 0
                        )
                    })

                total_cal = totals.get("Calories (kcal)", 0.0)
                plan = {
                    "success": True,
                    "calorie_budget": calorie_budget,
                    "total_calories": total_cal,
                    "remaining_calories": calorie_budget - total_cal,
                    "totals": totals,
                    "plan": {meal.capitalize(): foods},
                    "selected_indices": chosen
                }
        else:
            # Existing planner already validates total calories.
            pass
    else:
        plan = create_general_meal_plan(
            candidates,
            requested_meal=meal
        )

    if not plan.get("success"):
        return plan.get(
            "message",
            "No valid plan could be generated."
        )

    # For plans created without a budget, attach no invented target.
    if calorie_budget is None:
        plan["calorie_budget"] = None
        plan["remaining_calories"] = None

    problems = validate_selected_indices(
        plan["selected_indices"],
        diet=diet,
        preference=preference
    )

    if problems:
        raise RuntimeError(
            "Final planning validation failed:\n"
            + "\n".join(problems)
        )

    if (
        calorie_budget is not None
        and plan["total_calories"] > calorie_budget
    ):
        raise RuntimeError(
            "Final calorie validation failed."
        )

    LAST_PLAN = plan

    print("[FINAL VALIDATION] Dietary constraint : PASSED")
    print("[FINAL VALIDATION] Food preference    : PASSED")
    if calorie_budget is not None:
        print("[FINAL VALIDATION] Calorie budget     : PASSED")

    context = multiagent_plan_to_context(plan)

    budget_instruction = (
        f"The verified total is within the requested maximum "
        f"of {calorie_budget} kcal."
        if calorie_budget is not None
        else (
            "No calorie limit was requested. Do not invent or imply "
            "a calorie target."
        )
    )

    prompt = f"""
A deterministic Nutrition Planning Agent created and validated
the following plan from a local 120-food database.

USER REQUEST:
{question}

DIETARY CONSTRAINT:
{diet or 'None'}

FOOD PREFERENCE:
{preference or 'None'}

MEAL CONSTRAINT:
{meal or 'None'}

VERIFIED PLAN:
{context}

VALIDATION:
- Dietary compatibility: PASSED
- Food-preference compatibility: PASSED
- Numerical totals: calculated by Python/Pandas
- {budget_instruction}

INSTRUCTIONS:
1. Explain only the verified plan above.
2. Do not add or substitute foods.
3. Do not change any nutritional values or totals.
4. Respect the diet and food-preference constraints.
5. If a meal has no eligible food, state that the local database
   did not provide an eligible option; do not invent one.
6. Do not claim this is medically personalized nutrition.
7. Mention that the plan is limited to the local 120-food dataset.
8. Keep the response concise and practical.
"""

    print("[OLLAMA] Explaining verified Planning Agent output...")

    return ask_ollama(
        prompt,
        system_prompt=(
            "You explain deterministic dietary-planning results. "
            "Never add foods or numerical values that were not "
            "provided by the planning tool."
        ),
        temperature=0.1
    )


# ----------------------------------------------------------------------
# AGENT 2: NUTRITION ANALYSIS AGENT
# ----------------------------------------------------------------------

def detect_food_rows_in_question(question):
    q = question.lower()
    matches = []

    # Match complete Food_Item names from the local database.
    for index, row in df.iterrows():
        name = str(row.get("Food_Item", "")).strip()

        if not name:
            continue

        if contains_term(q, name.lower()):
            matches.append(int(index))

    return matches


def rows_from_last_plan():
    if not LAST_PLAN:
        return []

    return list(
        dict.fromkeys(
            LAST_PLAN.get("selected_indices", [])
        )
    )


def nutrition_analysis_agent(question):
    print("\n" + "=" * 78)
    print("NUTRITION ANALYSIS AGENT")
    print("=" * 78)

    q = question.lower()

    indices = detect_food_rows_in_question(question)
    source = "foods explicitly named in the question"

    refers_to_plan = any(
        phrase in q
        for phrase in [
            "this plan",
            "the plan",
            "my plan",
            "meal plan",
            "diet plan",
            "previous plan",
            "last plan"
        ]
    )

    if not indices and refers_to_plan:
        indices = rows_from_last_plan()
        source = "the most recent Planning Agent result"

    if not indices:
        return (
            "I can analyze foods that are explicitly named in your "
            "question, or the most recent meal plan generated during "
            "this session. For example: 'Compare salmon and chicken' "
            "or 'Analyze the last plan.'"
        )

    working = df.loc[indices].copy()

    # Deterministic totals.
    totals = {}
    for column in NUMERIC_COLUMNS:
        if column in working.columns:
            totals[column] = float(
                working[column].fillna(0).sum()
            )

    # Structured evidence table for the LLM.
    records = []

    for index, row in working.iterrows():
        records.append({
            "Food_Item": str(row.get("Food_Item", "")),
            "Category": str(row.get("Category", "")),
            "Calories_kcal": float(
                row.get("Calories (kcal)", 0) or 0
            ),
            "Protein_g": float(row.get("Protein (g)", 0) or 0),
            "Carbohydrates_g": float(
                row.get("Carbohydrates (g)", 0) or 0
            ),
            "Fat_g": float(row.get("Fat (g)", 0) or 0),
            "Fiber_g": float(row.get("Fiber (g)", 0) or 0),
            "Sugars_g": float(row.get("Sugars (g)", 0) or 0),
            "Sodium_mg": float(row.get("Sodium (mg)", 0) or 0),
            "Cholesterol_mg": float(
                row.get("Cholesterol (mg)", 0) or 0
            )
        })

    # Deterministic leaders for common comparison dimensions.
    leaders = {}

    leader_columns = {
        "highest protein": "Protein (g)",
        "lowest calories": "Calories (kcal)",
        "highest fiber": "Fiber (g)",
        "lowest sodium": "Sodium (mg)",
        "lowest sugar": "Sugars (g)"
    }

    for label, column in leader_columns.items():
        if column not in working.columns or working[column].dropna().empty:
            continue

        if label.startswith("lowest"):
            idx = working[column].astype(float).idxmin()
        else:
            idx = working[column].astype(float).idxmax()

        leaders[label] = {
            "food": str(df.loc[idx].get("Food_Item", "")),
            "value": float(df.loc[idx].get(column, 0) or 0),
            "column": column
        }

    print(f"[ANALYSIS AGENT] Evidence source : {source}")
    print(f"[ANALYSIS AGENT] Foods analyzed  : {len(working)}")
    print("[ANALYSIS AGENT] Arithmetic      : Python/Pandas")
    print("[ANALYSIS AGENT] LLM role        : explanation only")

    evidence_json = json.dumps(
        {
            "source": source,
            "records": records,
            "totals": totals,
            "deterministic_comparison_leaders": leaders
        },
        indent=2
    )

    prompt = f"""
A deterministic Nutrition Analysis Agent retrieved and calculated
the following evidence from the local food database.

USER QUESTION:
{question}

VERIFIED ANALYSIS DATA:
{evidence_json}

INSTRUCTIONS:
1. Answer the user's comparison or analysis question using only
   the verified data above.
2. Do not invent nutritional values.
3. Do not recalculate or alter Python/Pandas totals.
4. If comparing foods, clearly identify relevant differences.
5. A "highest" or "lowest" result is only among the foods shown,
   not a universal health ranking.
6. Do not turn the analysis into medical or personalized dietary advice.
7. Keep the response concise and practical.
"""

    print("[OLLAMA] Explaining verified Analysis Agent output...")

    return ask_ollama(
        prompt,
        system_prompt=(
            "You explain verified nutrition calculations. "
            "Do not invent, modify, or independently calculate values."
        ),
        temperature=0.1
    )


# ----------------------------------------------------------------------
# ANALYSIS-INTENT DETECTION
# ----------------------------------------------------------------------

def is_nutrition_analysis_request(question):
    q = question.lower()

    analysis_phrases = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "analyze",
        "analyse",
        "analysis",
        "evaluate this plan",
        "evaluate the plan",
        "total calories",
        "total protein",
        "total carbohydrate",
        "total carbs",
        "total fat",
        "total fiber",
        "total fibre",
        "total sodium",
        "nutritional totals",
        "nutrition totals",
        "which has more",
        "which has less",
        "which is higher",
        "which is lower",
        "how many calories in this plan",
        "how much protein in this plan",
        "last plan",
        "previous plan"
    ]

    return any(
        phrase in q
        for phrase in analysis_phrases
    )



# ======================================================================
# 37. INTENT ROUTER
# ======================================================================

def classify_request(question):
    """
    Deterministic orchestrator.

    Priority matters:
    MCP -> BMI -> Planning Agent -> Analysis Agent -> RAG -> DIRECT
    """

    q = question.lower().strip()

    # 1. External GitHub MCP
    MCP_KEYWORDS = [
        "github",
        "repository",
        "repositories",
        "repo",
        "search github"
    ]

    if any(keyword in q for keyword in MCP_KEYWORDS):
        return "MCP"

    # 2. Deterministic BMI tool
    if "bmi" in q or "body mass index" in q:
        return "BMI"

    # 3. Specialized meal-planning agent
    if is_diet_plan_request(question):
        return "PLANNING_AGENT"

    # 4. Specialized deterministic nutrition-analysis agent
    if is_nutrition_analysis_request(question):
        return "ANALYSIS_AGENT"

    # 5. Standard local Hybrid RAG
    RAG_KEYWORDS = [
        "food", "foods", "diet", "dietary", "nutrition", "nutrient",
        "protein", "calorie", "calories", "kcal",
        "carbohydrate", "carbohydrates", "carb", "carbs",
        "fat", "fiber", "fibre", "sugar", "sodium", "cholesterol",
        "vegetarian", "vegan", "seafood", "sea food", "fish",
        "meat", "chicken", "beef", "egg", "eggs", "dairy",
        "fruit", "fruits", "vegetable", "vegetables",
        "grain", "grains", "breakfast", "lunch", "dinner",
        "snack", "meal", "meals", "eat", "eating"
    ]

    if any(keyword in q for keyword in RAG_KEYWORDS):
        return "RAG"

    # 6. No tool required
    return "DIRECT"


# ======================================================================
# 38. CHATBOT
# ======================================================================

def chatbot(question):

    print("\n" + "=" * 78)
    print("ORCHESTRATOR AGENT")
    print("=" * 78)
    print(f"[ORCHESTRATOR] User query: {question}")

    route = classify_request(question)

    print(f"[ORCHESTRATOR] Selected capability: {route}")

    if route == "DIRECT":
        print("[DIRECT] No RAG or external tool required.")
        return ask_ollama(question)

    if route == "RAG":
        print("[RAG] Hybrid local food retrieval selected.")
        return rag_answer(question)

    if route == "PLANNING_AGENT":
        print("[ORCHESTRATOR] Delegating to Nutrition Planning Agent.")
        return nutrition_planning_agent(question)

    if route == "ANALYSIS_AGENT":
        print("[ORCHESTRATOR] Delegating to Nutrition Analysis Agent.")
        return nutrition_analysis_agent(question)

    if route == "BMI":
        print("[ORCHESTRATOR] Delegating to deterministic BMI tool.")
        return bmi_answer(question)

    if route == "MCP":
        print("[ORCHESTRATOR] Delegating to GitHub MCP capability.")
        return asyncio.run(
            github_mcp_search(question)
        )

    return "Unable to determine request type."


# ======================================================================
# 39. LOCAL RETRIEVAL TEST
# ======================================================================

def test_retrieval():

    queries = [

        "high protein foods",

        "seafood",

        "low calorie vegetarian lunch",

        "high protein vegetarian dinner",

        "vegan dinner",

        "What is a suitable diet "
        "for me as I am vegetarian?"
    ]

    print(
        "\n" + "=" * 78
    )

    print(
        "LOCAL RAG TEST "
        "(NO OLLAMA GENERATION)"
    )

    print(
        "=" * 78
    )

    for query in queries:

        print(
            f"\nQUERY: {query}"
        )

        print(
            "-" * 78
        )

        retrieval = (
            retrieve_foods(
                query,
                k=top_k
            )
        )

        print(
            "Method:",
            retrieval[
                "method"
            ]
        )

        print(
            "Diet:",
            retrieval[
                "diet"
            ]
        )

        print(
            "Meal:",
            retrieval[
                "meal"
            ]
        )

        for item in (
            retrieval[
                "results"
            ]
        ):

            print(
                f"\nRank "
                f"{item['rank']}"
            )

            print(
                item["text"]
            )


# ======================================================================
# 40. MAIN
# ======================================================================

def main():


    answer = rag_answer(input())


    print("answer:")
    print("-"*55)
    print(answer)
    

if __name__ == "__main__":

    main()

