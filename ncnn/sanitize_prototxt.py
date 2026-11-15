import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
# strip comments
s = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
# fork-specific field
s = re.sub(r"^\s*phase\s*:\s*PREDICT\s*$", "", s, flags=re.M)
# drop in-place 5D->4D Reshape layer (depth dim is 1)
s = re.sub(r'layer\s*\{[^{}]*type\s*:\s*"Reshape"[^{}]*(\{[^{}]*(\{[^{}]*\})?[^{}]*\})?[^{}]*\}', "", s)
# 3-D kernel/pad/stride -> 2-D (drop the trailing depth value)
for key in ("kernel_size", "pad", "stride", "dilation"):
    s = re.sub(rf"({key}\s*:\s*\d+)\s+({key}\s*:\s*\d+)\s+{key}\s*:\s*\d+", r"\1 \2", s)
# 5-D input shape -> 4-D
s = re.sub(r"(input_shape\s*\{\s*dim:\s*1\s*dim:\s*1\s*dim:\s*1024\s*dim:\s*1024)\s*dim:\s*1\s*\}", r"\1 }", s)
open(dst, "w").write(s)
