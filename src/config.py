sort_by_enum = {
    "most_relevant": "qualityScore",  # the most relevant reviews
    "newest": "newestFirst",  # the most recent reviews
    "highest_rating": "ratingHigh",  # the highest rating reviews
    "lowest_rating": "ratingLow",  # the lowest rating reviews
}


review_default_result = {
    "token": "",  # pagination token
    "review_id": "",  # review unique id
    "retrieval_date": "",
    "rating": 0,  # float usually 1-5
    "rating_max": 0,  # float usually 5
    "other_ratings": "",  # other ratings such as rooms, service, placing, etc
    "relative_date": "",
    "user_name": "",
    "user_url": "",
    "user_is_local_guide": None,
    "user_comments": 0,  # number of user comments
    "user_photos": 0,  # number of user photos
    "text": "",  # review text if exists
    "errors": [],  # list of errors parsing review
}


metadata_default = {
    "feature_id": "",  # hotel unique id
    "retrieval_date": "",
    "place_name": "",
    "address": "",
    "overall_rating": 0,  # float usually 1-5
    "n_reviews": -1,  # number of reviews
    "topics": "",  # topics separated by number of reviews
    "url": "",  # hotel url
}
