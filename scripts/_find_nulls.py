import pathlib
import sys

root = pathlib.Path(sys.argv[1])
hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix not in {".py", ".pth"}:
        continue
    data = path.read_bytes()
    if b"\x00" in data:
        hits.append((data.count(b"\x00"), path))
print("files", len(hits))
for count, path in hits[:30]:
    print(count, path)
