from datetime import datetime

# Time stamp
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


input_cols = [
    "name",
    "sort_by",
    "hl",
    "url",
    "n_reviews_max",
    "state",
    "region",
    "stars",
]

agg_dict = {
    # Input
    "name_input": "first",
    "sort_by": "first",
    "hl": "first",
    "url": "first",
    "n_reviews_max": "first",
    "state": "first",
    "region": "first",
    "stars": "first",
    "feature_id": "first",
    "retrieval_date_metadata": "first",
    "place_name": "first",
    "address": "first",
    "overall_rating": "first",
    "n_reviews": "first",
    "topics": "first",
    "file_name": "first",
    # Review
    "review_id": "count",
    "retrieval_date": "max",
    "rating": "mean",
    "rating_max": "mean",
    "relative_date": "count",
    "likes": "sum",
    "other_ratings": "count",
    "trip_type_travel_group": "count",
    "user_name": "count",
    "user_is_local_guide": "sum",
    "user_reviews": "count",
    "user_photos": "count",
    "user_url": "count",
    "text": "count",
    "response_text": "count",
    "response_relative_date": "count",
}

text_agg_dict = {
    # Input
    "place_name": "first",
    # Review
    "review_id": "count",
    "text": "count",
    "response_text": "count",
    "words_len": "sum",
    "tokens_len": "sum",
}

feature_cols = [
    # Input
    "hl",
    "state",
    "region",
    "stars",
    # Metadata
    "place_name",
    "topics",
    # Reviews
    "review_id",
    "retrieval_date",
    "rating",
    "relative_date",
    "likes",
    "other_ratings",
    "trip_type_travel_group",
    "user_name",
    "user_is_local_guide",
    "user_reviews",
    "user_photos",
    "text",
    "response_text",
    "response_relative_date",
]

text_cols = [
    # Input
    "hl",
    # Metadata
    "place_name",
    "topics",
    # Reviews
    "review_id",
    "retrieval_date",
    "relative_date",
    "other_ratings",
    "trip_type_travel_group",
    "text",
    "response_text",
    "response_relative_date",
]


full_cols = [
    # Input
    "name_input",
    "sort_by",
    "hl",
    "url",
    "n_reviews_max",
    "state",
    "region",
    "stars",
    # Metadata
    "name",
    "feature_id",
    "retrieval_date_metadata",
    "place_name",
    "address",
    "overall_rating",
    "n_reviews",
    "topics",
    "file_name",
    # Reviews
    "token",
    "review_id",
    "retrieval_date",
    "rating",
    "rating_max",
    "relative_date",
    "likes",
    "other_ratings",
    "trip_type_travel_group",
    "user_name",
    "user_is_local_guide",
    "user_reviews",
    "user_photos",
    "user_url",
    "text",
    "response_text",
    "response_relative_date",
    "errors",
]

relative_date_maps = {
    "pt-br": {
        "one_regex": "^uma?",
        "ago_regex": "\satras",
        "time_unit": {
            "ano": "years",
            "anos": "years",
            "mes": "months",
            "meses": "months",
            "semana": "weeks",
            "semanas": "weeks",
            "dia": "days",
            "dias": "days",
            "hora": "hours",
            "horas": "hours",
            "minuto": "minutes",
            "minutos": "minutes",
            "segundo": "seconds",
            "segundos": "seconds",
        },
    },
    "en": {
        "one_regex": "^a",
        "ago_regex": "\sago",
        "time_unit": {
            "year": "years",
            "years": "years",
            "month": "months",
            "months": "months",
            "week": "weeks",
            "weeks": "weeks",
            "day": "days",
            "days": "days",
            "hour": "hours",
            "hours": "hours",
            "minute": "minutes",
            "minutes": "minutes",
            "second": "seconds",
            "seconds": "seconds",
        },
    },
}

translated_text_maps = {
    "pt-br": {
        "flag": "Tradução do Google",
        "regex": "\(Tradução do Google\)|\(Original\).*?$",
    },
    "en": {
        "flag": "Translated by Google",
        "regex": "\(Translated by Google\)|\(Original\).*?$",
    },
}
