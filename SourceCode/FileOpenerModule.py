def ReadLines(name):
    with open(name, "r") as f:
        return f.readlines()

def WriteLines(name, lines):
    with open(name, "w") as f:
        f.writelines(lines)

def ReadLineWithKey(name, key):
    data = ReadLines(name)
    for i, line in enumerate(data):
        if key in line:
            return line.strip("\n").strip(" ")

def WriteLineWithKey(name, key, new_val):
    data = ReadLines(name)
    for i, line in enumerate(data):
        if key in line:
            data[i] = key + " " + str(new_val) + "\n"
            break
    WriteLines(name, data)
