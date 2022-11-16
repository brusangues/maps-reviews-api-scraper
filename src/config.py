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
    "relative_date": "",  # string containing the localized relative date
    "likes": -1,  # review likes if exists
    "other_ratings": "",  # other ratings such as rooms, service, placing, etc
    "trip_type_travel_group": "",
    "user_name": "",
    "user_is_local_guide": None,
    "user_reviews": "",  # total number of reviews made by the user
    "user_photos": "",  # total number of photos added by the user
    "user_url": "",
    "text": "",  # review text if exists
    "response_text": "",  # owner response text if exists
    "response_relative_date": "",  # string containing the localized relative date
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
