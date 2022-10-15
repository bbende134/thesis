import csv
a = [[0,1,3,4,2,2.1],[2,3.1,3.3,2.2,4,3]]
with open("out.csv", "w", newline="") as f:
    print("opened")
    writer = csv.writer(f)
    writer.writerows(a)