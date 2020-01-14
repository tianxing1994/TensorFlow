import pandas as pd

license_plates = pd.read_csv("dataset/indian_license_plates.csv")

for index, row in license_plates.iterrows():
    print(index)
    print(row)
    print(type(row))
    print(row["image_name"])
    print(row["image_width"])
    print(row["image_height"])
    print(row["top_x"])
    print(row["top_y"])
    print(row["bottom_x"])
    print(row["bottom_y"])
    break
