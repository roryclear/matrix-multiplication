import os
print("yo")

os.remove("ffs.py")
f = open("ffs.py", "a")
f.write("print(\"HELLO WORLD\")\n")
f.close()

import ffs
print("2 ffs")