def format_datetime(raw_string):
    date, time = raw_string.split(" ")
    date = date.replace("-", "_")
    time = time.replace(":", "_")
    time = time.split(".")[0]
    return date + "_" + time