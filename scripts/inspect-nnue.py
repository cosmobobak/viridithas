import struct

# read ../nnue/output_weights.bin as i16s and print the values

weights = []
with open("../nnue/output_weights.bin", "rb") as f:
    while True:
        b = f.read(2)
        if not b:
            break
        w = struct.unpack("h", b)[0]
        weights.append(w)

print(f"found {len(weights)} weights")
print(f"min: {min(weights)}")
print(f"max: {max(weights)}")
print(f"avg: {sum(weights) / len(weights)}")
print(f"sum: {sum(weights)}")
print(f"sum of abs: {sum(abs(w) for w in weights)}")
print(f"first 10: {weights[:10]}")
