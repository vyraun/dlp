import sys
f = open(sys.argv[1])

val = []

for line in f:
    #print("Hi")
    if "Test acc" in line:
        #print(line)
        #print(line.split())
        val.append(line.split()[7])

print(val)
