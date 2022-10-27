sort_by_enum = {
    "most_relevant": "qualityScore",  # the most relevant reviews
    "newest": "newestFirst",  # the most recent reviews
    "highest_rating": "ratingHigh",  # the highest rating reviews
    "lowest_rating": "ratingLow",  # the lowest rating reviews
}


review_default_result = {
    "token": "",
    "review_id": "",
    "retrieval_date": "",
    "rating": 0,
    "rating_max": 0,
    "other_ratings": "",
    "relative_date": "",
    "user_name": "",
    "user_url": "",
    "user_is_local_guide": None,
    "user_comments": 0,
    "user_photos": 0,
    "text": "",
}


metadata_default = {
    "feature_id": "",
    "retrieval_date": "",
    "place_name": "",
    "address": "",
    "overall_rating": 0,
    "n_reviews": -1,
    "topics": "",
    "url": "",
}
