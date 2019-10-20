import pickle


result = {}
counter = 0
with open("categories-hierarchy.csv") as tree:
    curr_line = tree.readline()
    while curr_line:
        keys = curr_line.split(",")
        curr_key = ""
        for key in keys:
            if not key:
                break
            curr_key += "/" + key
        print(curr_key)
        curr_line = tree.readline()
        if curr_line:
            result[curr_key] = counter
        counter += 1
tree.close()
with open("categoriesMap", "ab") as map:
    pickle.dump(result, map)
map.close()
print(len(result))
