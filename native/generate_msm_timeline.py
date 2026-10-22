"""Extract diagnostic host functions from pinned ICICLE; reuse existing kernels.

This build-time adapter does not rewrite arithmetic or change API order. Fail
closed if the source layout no longer matches reviewed v2.8.0 boundaries.
"""
import pathlib
import re
import sys


def mask_comments(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                  lambda match: " " * len(match.group()), text, flags=re.S)


def closing(text, begin):
    masked = mask_comments(text)
    depth = 0
    left, right = ("{", "}") if text[begin] == "{" else ("(", ")")
    for index in range(begin, len(text)):
        if masked[index] == left:
            depth += 1
        elif masked[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unclosed source expression")


def arguments(text):
    result = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    result.append(text[start:].strip())
    return result


def extract(text, anchor):
    index = text.index(anchor)
    start = text.rfind("template <", 0, index)
    body = text.index("{", index)
    return text[start:closing(text, body) + 1]


def generate(source):
    bucket = extract(source, "cudaError_t bucket_method_msm(")
    entry = extract(source, "cudaError_t msm(")
    bucket = bucket.replace("bucket_method_msm(", "timeline_bucket_method_msm(")
    entry = entry.replace("cudaError_t msm(", "cudaError_t timeline_msm(")
    entry = entry.replace("bucket_method_msm(", "timeline_bucket_method_msm(")
    bucket = bucket.replace("DynaverseMsmAllocations resources;", "DynaverseTimelineAllocations resources;")
    for original, replacement in {
        "cudaMemcpyAsync(": "dvgpu_timeline_copy(",
        "cudaStreamSynchronize(": "dvgpu_timeline_synchronize(",
        "cudaStreamWaitEvent(": "dvgpu_timeline_wait_event(",
        "cudaStreamCreate(": "dvgpu_timeline_stream_create(",
        "cudaStreamDestroy(": "dvgpu_timeline_stream_destroy(",
        "side_resources.create_stream(": "dvgpu_timeline_stream_create(side_resources, ",
        "side_resources.destroy_stream(": "dvgpu_timeline_stream_destroy(side_resources, ",
        "cudaMemset(": "dvgpu_timeline_memset(",
        "cudaMemsetAsync(": "dvgpu_timeline_memset_async(",
    }.items():
        bucket = bucket.replace(original, replacement)
    kernels = {
        "split_scalars_kernel": 0, "initialize_buckets_kernel": 4,
        "accumulate_buckets_kernel": 6, "big_triangle_sum_kernel": 7,
        "final_accumulation_kernel": 8,
    }
    launch_pattern = re.compile(r"(\w+(?:<[^;\n]*?>)?)<<<(.*?)>>>\s*\(", re.S)
    matches = list(launch_pattern.finditer(bucket))
    if len(matches) != bucket.count("<<<"):
        raise ValueError("uncovered ICICLE kernel launch")
    for match in reversed(matches):
        begin = match.end() - 1
        end = closing(bucket, begin) + 1
        if bucket[end:].lstrip()[0] != ";":
            raise ValueError("unexpected kernel expression")
        config = arguments(match.group(2))
        stream = config[3] if len(config) == 4 else "nullptr"
        name = match.group(1).split("<")[0]
        original = bucket[match.start():end]
        replacement = f'dvgpu_timeline_kernel("{name}", {kernels.get(name, 9)}, {stream}, [&] {{ {original}; }})'
        bucket = bucket[:match.start()] + replacement + bucket[end:]
    cub_pattern = re.compile(r"cub::(Device\w+)::(\w+)\(")
    for match in reversed(list(cub_pattern.finditer(bucket))):
        begin = match.end() - 1
        end = closing(bucket, begin) + 1
        args = arguments(bucket[begin + 1:end - 1])
        stage = {"DeviceRadixSort": 1, "DeviceRunLengthEncode": 2, "DeviceScan": 3}[match.group(1)]
        # Size queries with null storage launch no kernels; they are preparation.
        original = bucket[match.start():end]
        replacement = f'dvgpu_timeline_cub("{match.group(1)}/{match.group(2)}", {stage}, {args[-1]}, {args[0]} != nullptr, [&] {{ return {original}; }})'
        bucket = bucket[:match.start()] + replacement + bucket[end:]
    marker = "const unsigned nof_scalars = batch_size * single_msm_size;"
    bucket = bucket.replace(marker, "dvgpu_timeline().configure(bitsize, c, precompute_factor, batch_size);\n      " + marker)
    if "DynaverseTimelineAllocations resources;" not in bucket:
        raise ValueError("missing allocation cleanup guard")
    return "// Generated from pinned ICICLE host orchestration; original kernels reused.\nnamespace msm {\nnamespace {\n" + bucket + "\n}\n" + entry + "\n}\n"


if __name__ == "__main__":
    source_path, output_path = map(pathlib.Path, sys.argv[1:])
    output_path.write_text(generate(source_path.read_text()), encoding="utf-8")
