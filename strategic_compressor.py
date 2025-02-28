import zlib

model = open("model.tflite", "rb")
contents = model.read()
model.close()

print(len(contents))

num_files = 3
max_size = 25_000_000
n=1
while len(contents)>max_size:
    with open(f"meow{n}", 'wb') as file:
        file.write(contents[:max_size])
        print(f"wrote meow {n}", end="\r")
        contents = contents[max_size:]
    n+=1

if contents:
    with open(f"meow{n}", 'wb') as file:
        file.write(contents)
        print(f"wrote meow {n}")

print("Done")

