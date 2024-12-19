import json

# Open and read the JSON file
with open("log/log-1-0-mini.json", "r") as json_file:
    data = json.load(json_file)  # Read and parse the JSON content

# Now you can work with the 'data' variable, which is a Python dictionary
data

def get_input_vector(data):
    for item in data["0"]["Backprop"]["a_list"][0]:
        print(item[0])

def get_l2B_vector(data):
    for item in data["0"]["Pre-update"]["biases"][0]:
        print(item[0])

def get_l2W_vector(data):

    for item in data["0"]["Pre-update"]["weights"][0][0]:
        print(item)

def get_l3B_vector(data):
    for item in data["0"]["Pre-update"]["biases"][1]:
        print(item[0])

def get_l3W_vector(data):

    for item in data["0"]["Pre-update"]["weights"][1]:
        print(item[0])

def get_RealOut_vector(data):

    for item in data["0"]["Pre-update"]["mini_batch"][0][1]:
        print(item[0])

#efter backprobagation

def get_post_l2B_vector(data):
    for item in data["0"]["post-update"]["biases"][0]:
        print(item[0])

def get_post_l2W_vector(data):

    for item in data["0"]["post-update"]["weights"][0][0]:
        print(item)

def get_post_l3B_vector(data):
    for item in data["0"]["post-update"]["biases"][1]:
        print(item[0])

def get_post_l3W_vector(data):

    for item in data["0"]["post-update"]["weights"][1][0]:
        print(item)



get_post_l2W_vector(data)